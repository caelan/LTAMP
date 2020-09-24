#!/usr/bin/env python3

from __future__ import print_function

import os
import sys

# NOTE(caelan): must come before other imports
from learn_tools.common import map_general, get_max_cores

cwd = os.getcwd()
sys.path.extend([
    os.path.join(cwd, 'pddlstream'),  # Important to use absolute path when doing chdir
    os.path.join(cwd, 'ss-pybullet'),
    # 'pddlstream/examples',
    # 'pddlstream/examples/pybullet/utils',
])

import argparse
import numpy as np
import time
import datetime
import math

np.set_printoptions(precision=3, threshold=3, edgeitems=1, suppress=True)  # , linewidth=1000)

import pddlstream.language.statistics
pddlstream.language.statistics.LOAD_STATISTICS = False
pddlstream.language.statistics.SAVE_STATISTICS = False

from plan_tools.visualization import draw_forward_reachability, draw_names
from plan_tools.planner import plan_actions, PlanningWorld
from plan_tools.ros_world import ROSWorld
from plan_tools.common import set_seed, VIDEOS_DIRECTORY

from control_tools.common import get_arm_prefix
from control_tools.execution import execute_plan
from learn_tools.collectors.collect_pour import POUR_COLLECTOR
from learn_tools.collectors.collect_scoop import SCOOP_COLLECTOR
from learn_tools.collectors.collect_stir import STIR_COLLECTOR
from learn_tools.collectors.collect_push import PUSH_COLLECTOR
from learn_tools.learner import RANDOM, TRAINING, get_trial_parameter_fn, DATA_DIRECTORY, SEPARATOR, REAL_PREFIX, \
    PARAMETER, SKILL, FEATURE, SCORE
from learn_tools.active_learner import ActiveLearner
from learn_tools.common import DATE_FORMAT

from pybullet_tools.utils import ClientSaver, HideOutput, elapsed_time, has_gui, user_input, ensure_dir, \
    create_attachment, wait_for_user, read_pickle, write_json, is_darwin, WorldSaver, VideoSaver
from pddlstream.utils import str_from_object, safe_rm_dir, get_python_version, implies, find_unique

from multiprocessing import cpu_count

# HideOutput.DEFAULT_ENABLE = False

HOURS_TO_SECS = 60. * 60.
TRIALS_DIRECTORY = os.path.join(DATA_DIRECTORY, 'trials/')
TEMP_DIRECTORY = 'temp_parallel/'
# MAX_CPU_FRACTION = 0.75
# MAX_SERIAL = 1
# RAM_PER_TRIAL = (2.215/100)*64 # GB # TODO: automatically obtain RAM

# TODO: score the skill names as an enum
SKILL_COLLECTORS = {
    'pour': POUR_COLLECTOR,
    'scoop': SCOOP_COLLECTOR,
    'stir': STIR_COLLECTOR,
    'push': PUSH_COLLECTOR,
}


##############################################################################################################

def skip_to_end(sim_world, planning_world, plan):
    if plan is None:
        return plan
    bead_attachments = [create_attachment(cup, -1, bead)
                        for bead, cup in sim_world.initial_beads.items()]
    action_index = -1
    action, args = plan[action_index]
    arm = args[0]
    control = args[-1]
    context = control['context']
    command = control['commands'][0]
    context.apply_mapping(planning_world.body_mapping)
    for arm, name in list(sim_world.controller.holding.items()):
        sim_world.controller.detach(arm, name)
    attachments = context.assign()  # TODO: need to move the beads as well
    next(command.iterate(sim_world, attachments))
    for attachment in list(attachments.values()) + bead_attachments:
        attachment.assign()
    for name in attachments:
        sim_world.controller.attach(get_arm_prefix(arm), name)
    return plan[action_index:]


def simulate_trial(sim_world, task, parameter_fns,
                   teleport=True, collisions=True, max_time=20, verbose=False, **kwargs):
    with ClientSaver(sim_world.client):
        viewer = has_gui()
    planning_world = PlanningWorld(task, visualize=False)
    planning_world.load(sim_world)
    print(planning_world)

    start_time = time.time()
    plan = plan_actions(planning_world, max_time=max_time, verbose=verbose,
                        collisions=collisions, teleport=teleport,
                        parameter_fns=parameter_fns) # **kwargs
    # plan = None
    plan_time = time.time() - start_time
    planning_world.stop()
    if plan is None:
        print('Failed to find a plan')
        # TODO: allow option of scoring these as well?
        return None, plan_time

    if teleport:
        # TODO: skipping for scooping sometimes places the spoon in the bowl
        plan = skip_to_end(sim_world, planning_world, plan)
    if viewer:
        # wait_for_user('Execute?')
        wait_for_user()

    sim_world.controller.set_gravity()
    videos_directory = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, VIDEOS_DIRECTORY))
    #skill_videos = [filename.startswith(task.skill) for filename in os.listdir(videos_directory)]
    #suffix = len(skill_videos)
    suffix = datetime.datetime.now().strftime(DATE_FORMAT)
    video_path = os.path.join(videos_directory, '{}_{}.mp4'.format(task.skill, suffix))
    video_path = None
    with VideoSaver(video_path): # requires ffmpeg
        # TODO: teleport=False
        execute_plan(sim_world, plan, default_sleep=0.)
        if viewer:
            wait_for_user('Finished!')
        return plan, plan_time


##############################################################################################################

def start_task():
    pid = os.getpid()
    current_wd = os.getcwd()
    trial_wd = os.path.join(current_wd, TEMP_DIRECTORY, '{}/'.format(pid))
    safe_rm_dir(trial_wd)
    ensure_dir(trial_wd)
    os.chdir(trial_wd)
    return current_wd, trial_wd


def sample_task(skill, **kwargs):
    # TODO: extract and just control robot gripper
    with HideOutput():
        sim_world = ROSWorld(sim_only=True, visualize=kwargs['visualize'])
    # Only used by pb_controller, for ros_controller it's assumed always True
    sim_world.controller.execute_motor_control = True

    collector = SKILL_COLLECTORS[skill]
    while True:  # TODO: timeout if too many failures in a row
        sim_world.controller.set_gravity()
        task, feature, evaluation_fn = collector.collect_fn(sim_world, **kwargs['collect_kwargs'])
        # evaluation_fn is the score function for the skill
        if task is not None:
            task.skill = skill
            feature['simulation'] = True
            break
        sim_world.reset(keep_robot=True)
    saver = WorldSaver()
    if kwargs['visualize'] and kwargs['visualize'] != 'No_name':
        draw_names(sim_world)
        draw_forward_reachability(sim_world, task.arms)
    return sim_world, collector, task, feature, evaluation_fn, saver


def get_parameter_fns(sim_world, collector, fn, feature, valid):
    if isinstance(fn, str) and os.path.isfile(fn):
        fn = read_pickle(fn)
    parameter = prediction = None
    if isinstance(fn, ActiveLearner):
        fn.reset_sample()  # Should be redundant
        print('Learner: {} | Query type: {}'.format(fn.name, fn.query_type))
        # parameter_fn = lambda world, feature: iter([fn.query(feature)]) # Need to convert to parameter
        parameter_fn = fn.parameter_generator
        parameter = next(parameter_fn(sim_world, feature, valid=valid), False)
        # prediction = fn.prediction(feature, parameter)
        if parameter is not False:
            parameter['policy'] = (fn.name, fn.query_type) # fn.algorithm?
    elif fn == TRAINING:
        # Commits to a single parameter
        parameter_fn = collector.parameter_fns[RANDOM]
        for parameter in parameter_fn(sim_world, feature): # TODO: timeout
            if not valid or collector.validity_test(sim_world, feature, parameter):
                break
        else:
            parameter = False
        if parameter is not False:
            parameter['policy'] = fn
    elif fn in collector.parameter_fns:
        # Does not commit to a single parameter
        parameter_fn = collector.parameter_fns[fn]
    else:
        raise ValueError(fn)
    # TODO: could prune failing parameters here
    print('Parameter:', parameter)
    print('Prediction:', prediction)
    if parameter is not None:
        parameter_fn = get_trial_parameter_fn(parameter)
    parameter_fns = {collector.gen_fn: parameter_fn}
    return parameter_fns, parameter, prediction


def test_validity(result, world, collector, feature, parameter, prediction):
    result[FEATURE] = feature
    result[SCORE] = None
    if parameter is not None:
        result[PARAMETER] = parameter
    if prediction is not None:
        result['prediction'] = prediction
    if parameter is False:
        return False
    if parameter is None:
        return True
    result['valid'] = collector.validity_test(world, feature, parameter)
    return result['valid']


def get_parameter_result(sim_world, task, parameter_fns, evaluation_fn, **kwargs):
    start_time = time.time()
    plan, plan_time = None, 0
    while (plan is None) and (elapsed_time(start_time) < kwargs['max_time']):
        plan, plan_time = simulate_trial(sim_world, task, parameter_fns, **kwargs)
    score = None
    if plan is None:
        print('No plan was found in {} seconds.'.format(kwargs['max_time']))
        control = {'feature': None, 'parameter': None}
    else:
        score = evaluation_fn(plan)
        # _, args = find_unique(lambda a: a[0] == 'pour', plan)
        # control = args[-1]
    elapsed = elapsed_time(start_time)
    result = {
        'success': (plan is not None),
        'score': score,
        'plan-time': plan_time,
        'elapsed-time': elapsed, # Includes execution time
        # 'feature': control['feature'],
        # 'parameter': control['parameter']
    }
    return result


def complete_task(sim_world, current_wd, trial_wd):
    # TODO: the pybullet windows aren't fully closing on OSX
    sim_world.stop()
    os.chdir(current_wd)
    safe_rm_dir(trial_wd)


def run_trial(args):
    skill, fn, trial, kwargs = args
    if isinstance(fn, str) and os.path.isfile(fn):
        fn = read_pickle(fn)
    if not kwargs['verbose']:
        sys.stdout = open(os.devnull, 'w')
    current_wd, trial_wd = start_task()

    # if not is_darwin():
    #    # When disabled, the pybullet output still occurs across threads
    #    HideOutput.DEFAULT_ENABLE = False
    seed = hash((kwargs.get('seed', time.time()), trial))
    set_seed(seed)
    # time.sleep(1.*random.random()) # Hack to stagger processes

    sim_world, collector, task, feature, evaluation_fn, _ = sample_task(skill, **kwargs)
    print('Feature:', feature)
    parameter_fns, parameter, prediction = get_parameter_fns(sim_world, collector, fn, feature, kwargs['valid'])
    # TODO: some of the parameters are None when run in parallel
    result = {
        SKILL: skill,
        'date': datetime.datetime.now().strftime(DATE_FORMAT),
        'seed': seed,
        'trial': trial,
        'simulated': True,
        FEATURE: feature,
    }
    if test_validity(result, sim_world, collector, feature, parameter, prediction):
        result.update(get_parameter_result(sim_world, task, parameter_fns, evaluation_fn, **kwargs))
    else:
        print('Invalid parameter! Skipping planning.')
    # result['seed'] = seed
    # except BaseException as e:
    #    traceback.print_exc() # e

    complete_task(sim_world, current_wd, trial_wd)
    if not kwargs['verbose']:
        sys.stdout.close()
    return result

##############################################################################################################

def write_results(filename, results):
    if not results or (filename is None):
        return False
    # path = '{}.pk{}'.format(filename, get_python_version())
    # write_pickle(path, results)
    path = '{}.json'.format(filename)
    try:
        write_json(path, results)
    except KeyboardInterrupt:
        write_json(path, results)
    finally:
        print('Saved {} trials to {}'.format(len(results), path))
    return True


# def mute():
#    sys.stdout = open(os.devnull, 'w')
#    # sys.stdout = open(str(os.getpid()) + ".out", "w") # TODO: log output
#    # TODO: maybe this isn't closed...

def run_trials(trials, data_path=None, num_cores=False, **kwargs):
    # https://stackoverflow.com/questions/15314189/python-multiprocessing-pool-hangs-at-join
    # https://stackoverflow.com/questions/39884898/large-amount-of-multiprocessing-process-causing-deadlock
    # TODO: multiprocessing still seems to hang on one thread before starting
    assert (get_python_version() == 3)
    results = []
    if not trials:
        return results
    start_time = time.time()
    serial = (num_cores is False)  # None is the max number
    failures = 0
    scored = 0
    try:
        for result in map_general(run_trial, trials, serial, num_cores=num_cores, **kwargs):
            num_trials = len(results) + failures
            print('{}\nTrials: {} | Successes: {} | Failures: {} | Scored: {} | Time: {:.3f}'.format(
                SEPARATOR, num_trials, len(results), failures, scored, elapsed_time(start_time)))
            print('Result:', str_from_object(result))
            if result is None:
                failures += 1
                print('Error! Trial resulted in an exception')
                continue
            scored += int(result.get('score', None) is not None)
            results.append(result)
            write_results(data_path, results)
    # except BaseException as e:
    #    traceback.print_exc() # e
    finally:
        print(SEPARATOR)
        safe_rm_dir(TEMP_DIRECTORY)
        write_results(data_path, results)
        print('Hours: {:.3f}'.format(elapsed_time(start_time) / HOURS_TO_SECS))
    # TODO: make a generator version of this
    return results


##############################################################################################################

def get_data_path(skill_name, trials=[], real=False):
    print('Skill:', skill_name)
    if is_darwin():
        return None
    # (system, node, release, version, machine, processor)
    # system_name = platform.uname()[0].lower()
    # computer_name = platform.uname()[1]
    # computer_name = socket.gethostname() # TODO: changes when connected to wifi
    # What is really want is the name in PS1
    # user_name = getpass.getuser()

    prefix = '{}_'.format(REAL_PREFIX) if real else ''
    date_name = datetime.datetime.now().strftime(DATE_FORMAT)
    directory = os.path.join(DATA_DIRECTORY, '{}{}_{}/'.format(
        prefix, skill_name, date_name))
    ensure_dir(directory)
    suffix = '' if real else '_n={}'.format(len(trials))
    data_path = os.path.join(directory, 'trials{}'.format(suffix))
    print('Data path:', data_path)
    # print('System:', system_name)
    # print('Username:', user_name)
    return data_path


def get_trials(problem, fn, num_trials=1, max_time=60, seed=None, valid=True, visualize=False, verbose=False, collect_kwargs={}):
    kwargs = {
        'teleport': True,  # TODO: maybe just return kwargs
        'max_time': max_time,
        'seed': seed,
        'valid': valid,
        'visualize': visualize,
        'verbose': verbose,
        'collect_kwargs': collect_kwargs,
    }
    trials = [(problem, fn, trial, kwargs) for trial in range(num_trials)]
    print('Trials:', len(trials))
    return trials


def get_num_cores(trials, serial=False):
    if not trials:
        return 0
    # max_ram = 16. if is_darwin() else 64. # GB
    total_time = sum(args[-1]['max_time'] for args in trials)
    average_time = float(total_time) / len(trials)
    # TODO: estimate this per skill
    # average_time = args.time
    num_cores = get_max_cores(serial=serial)
    max_parallel = math.ceil(float(len(trials)) / num_cores)
    print('Max Cores:', cpu_count())
    print('Serial:', serial)
    print('Using Cores:', num_cores)
    print('Rounds:', max_parallel)
    print('Max hours: {:.3f}'.format(max_parallel * average_time / HOURS_TO_SECS))
    if serial:
        num_cores = False
    return num_cores


##############################################################################################################

def main():
    # collect data in parallel, parameters are generated uniformly randomly in a range
    # data stored to pour_date/trails_n=10000.json
    # TODO: ulimit settings
    # https://ss64.com/bash/ulimit.html
    # https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
    # import psutil
    # TODO: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    assert (get_python_version() == 3)
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fn', default=TRAINING,
                        help='The parameter function to use.')
    parser.add_argument('-n', '--num', type=int, default=10000,
                        help='The number of samples to collect')
    parser.add_argument('-p', '--problem', required=True,
                        choices=sorted(SKILL_COLLECTORS.keys()),
                        help='The name of the skill to learn.')
    parser.add_argument('-t', '--time', type=int, default=1*60,
                        help='The max planning runtime for each trial.')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='When enabled, visualizes execution.')
    args = parser.parse_args()
    serial = is_darwin()
    assert implies(args.visualize, serial)

    trials = get_trials(args.problem, args.fn, args.num, max_time=args.time,
                        valid=True, visualize=args.visualize, verbose=serial)
    data_path = None if serial else get_data_path(args.problem, trials)
    num_cores = get_num_cores(trials, serial)
    user_input('Begin?')
    # TODO: store the generating distribution for samples and objects?

    print(SEPARATOR)
    results = run_trials(trials, data_path, num_cores=num_cores)

    # TODO: could try removing HideOutput
    # https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/cf45791c40bed67f7e1c0dedbe5d60cad91dbde2/plan_tools/run_train.py
    # https://github.com/caelan/ss-pybullet/tree/6a525987393a98c5717b3d3857322ec6f6f9327f
    # https://github.com/caelan/ss-pybullet/compare/6a525987393a98c5717b3d3857322ec6f6f9327f...master


# python2 -m learn_tools.collect_pr2 -p scoop -f data/scoop_19-04-24_18-18-35/gp_batch_mlp.pk2 -m red_men

# Planning failure
# Result: {date: 19-12-09_21-32-54, elapsed-time: 62.864604234695435, feature: {bowl_base_diameter: 0.09711557680076334,
# bowl_diameter: 0.1637091189622879, bowl_height: 0.037615127861499786, bowl_name: red_bowl#1, bowl_type: red_bowl,
# cup_base_diameter: 0.05838908583815237, cup_diameter: 0.06021374464035034, cup_height: 0.06468292325735092,
# cup_name: orange3D_cup#1, cup_type: orange3D_cup, simulation: True}, parameter: None, plan-time: 3.284528970718384,
# score: None, seed: -645259421257130633, skill: pour, success: False, trial: 18, valid: True}

# Kitchen3D Experiments
# https://docs.google.com/document/d/1sOzTR42YF0AMSz4_MPDuBALOPC9Oc4C31cvDijx3Qcg/edit

if __name__ == '__main__':
    main()
