import argparse
import os
from collections import namedtuple

import numpy as np

from learn_tools.active_gp import ActiveGP, POUR_MLP_HYPERPARAM_3000, SCOOP_MLP_HYPERPARAM_3000
from learn_tools.collect_simulation import start_task, complete_task, get_parameter_result
from learn_tools.learnable_skill import load_data
from learn_tools.learner import FAILURE, SCORE, get_trial_parameter_fn
from learn_tools.retired.run_sample import get_data_dir, sample_task_with_seed
from learn_tools.uncertain_learner import DIVERSE, DIVERSELK, SAMPLE_STRATEGIES, BESTPROB
from pddlstream.utils import get_python_version, mkdir
from perception_tools.common import create_name
from plan_tools.common import set_seed
from pybullet_tools.utils import write_pickle, read_pickle, create_cylinder, get_point, unit_quat, \
    get_aabb_extent, set_pose, INFO_FROM_BODY, ModelInfo, CLIENT, create_box

import pybullet as p
Result = namedtuple('Result', ['samples', 'plan_time',
                               'scores', 'plan_results', 'task_lengthscale', 'context', 'n_samples', 'beta',
                               'best_beta'])

SEED = 11


def evaluate_generator_for_task(sim_world, collector, feature, task, domain, generator, evalfunc, saver):
    parameter_fns = {collector.gen_fn: generator}
    saver.restore()
    # TODO(caelan): update given new function parameterization
    result = get_parameter_result(sim_world, task, parameter_fns, evalfunc, max_time=150)
    if result is None or result[SCORE] is None:
        return FAILURE, result
    score = domain.score_fn(feature, None, result[SCORE])
    return score, result


def get_sample_strategy_result(sample_strategy, learner, context, sim_world, collector, task, feature, evalfunc, saver):
    learner.sample_strategy = sample_strategy
    learner.reset_sample()
    learner.set_world_feature(sim_world, feature)
    domain = learner.func

    def learner_generator(world=None, feature=None):
        while True:
            sample = learner.sample(context)
            if sample is None:
                print('======================No sample generated====================')
                break
            parameter = domain.parameter_from_sample(sample[domain.param_idx])
            yield parameter

    if learner.sample_strategy == BESTPROB:
        learner_generator = get_trial_parameter_fn(
            domain.parameter_from_sample(learner.sample(context)[domain.param_idx]))
    scores, plan_results = evaluate_generator_for_task(sim_world, collector, feature, task, learner.func, learner_generator,
                                                       evalfunc, saver)

    plan_time = plan_results['plan-time']

    n_samples = len(learner.sampled_xx)
    return Result(learner.sampled_xx, plan_time, scores, plan_results, learner.task_lengthscale.copy(),
                  context, n_samples, learner.beta, learner.best_beta)


def save_experiments(experiments_dir, expid, experiments):
    if experiments_dir is None:
        return None
    mkdir(experiments_dir)
    data_path = os.path.join(experiments_dir, 'experiments_{}.pk{}'.format(expid, get_python_version()))
    write_pickle(data_path, experiments)
    print('Saved', data_path)
    return data_path


def get_seeds(n):
    seeds = []
    for i in range(n):
        seeds.append(i + np.random.randint(1 << 31))
    return seeds


def eval_task_with_seed(domain, seed, learner, sample_strategy=DIVERSELK, task_lengthscale=None, obstacle=True):
    if task_lengthscale is not None:
        learner.task_lengthscale = task_lengthscale

    print('sample a new task')
    current_wd, trial_wd = start_task()

    ### fill in additional object
    item_ranges = {}
    ###
    sim_world, collector, task, feature, evalfunc, saver = sample_task_with_seed(domain.skill, seed=seed,
                                                                                 visualize=False,
                                                                                 item_ranges=item_ranges)

    context = domain.context_from_feature(feature)
    print('context=', *context)
    # create coffee machine
    if obstacle:
        bowl_name = 'bowl'
        cup_name = 'cup'
        if domain.skill == 'scoop':
            cup_name = 'spoon'
        for key in sim_world.perception.sim_bodies:
            if bowl_name in key:
                bowl_name = key
            if cup_name in key:
                cup_name = key

        cylinder_size = (.01, .03)
        dim_bowl = get_aabb_extent(p.getAABB(sim_world.perception.sim_bodies[bowl_name]))
        dim_cup = get_aabb_extent(p.getAABB(sim_world.perception.sim_bodies[cup_name]))
        bowl_point = get_point(sim_world.perception.sim_bodies[bowl_name])
        bowl_point = np.array(bowl_point)
        cylinder_point = bowl_point.copy()
        cylinder_offset = 1.2
        if domain.skill == 'scoop':
            cylinder_offset = 1.6
        cylinder_point[2] += (dim_bowl[2] + dim_cup[2]) * (np.random.random_sample() * .4 + cylinder_offset) \
                             + cylinder_size[1] / 2.
        # TODO figure out the task space
        cylinder_name = create_name('cylinder', 1)
        obstacle = create_cylinder(*cylinder_size, color=(0, 0, 0, 1))
        INFO_FROM_BODY[(CLIENT, obstacle)] = ModelInfo(cylinder_name, cylinder_size, 1, 1)
        sim_world.perception.sim_bodies[cylinder_name] = obstacle
        sim_world.perception.sim_items[cylinder_name] = None
        set_pose(obstacle, (cylinder_point, unit_quat()))

        box_name = create_name('box', 1)
        box1_size = (max(.17, dim_bowl[0] / 2 + cylinder_size[0] * 2), 0.01, 0.01)
        if domain.skill == 'scoop':
            box1_size = (max(.23, dim_bowl[0] / 2 + cylinder_size[0] * 2 + 0.05), 0.01, 0.01)
        obstacle = create_box(*box1_size)
        INFO_FROM_BODY[(CLIENT, obstacle)] = ModelInfo(box_name, box1_size, 1, 1)
        sim_world.perception.sim_bodies[box_name] = obstacle
        sim_world.perception.sim_items[box_name] = None
        box1_point = cylinder_point.copy()
        box1_point[2] += cylinder_size[1] / 2 + box1_size[2] / 2
        box1_point[0] += box1_size[0] / 2 - cylinder_size[0] * 2
        set_pose(obstacle, (box1_point, unit_quat()))

        box_name = create_name('box', 2)
        box2_size = (0.01, .01, box1_point[2] - bowl_point[2] + box1_size[2] / 2)
        obstacle = create_box(*box2_size)
        INFO_FROM_BODY[(CLIENT, obstacle)] = ModelInfo(box_name, box2_size, 1, 1)
        sim_world.perception.sim_bodies[box_name] = obstacle
        sim_world.perception.sim_items[box_name] = None
        box2_point = bowl_point.copy()
        box2_point[2] += box2_size[2] / 2
        box2_point[0] = box1_point[0] + box1_size[0] / 2
        set_pose(obstacle, (box2_point, unit_quat()))

    result = get_sample_strategy_result(sample_strategy, learner, context, sim_world, collector, task,
                                        feature, evalfunc, saver)
    print('context=', *context)
    complete_task(sim_world, current_wd, trial_wd)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trainsize', default=2000, type=int, help='training set size')
    parser.add_argument('expid', default=1, type=int, help='experiment ID')
    parser.add_argument('beta_lambda', type=float, default=0.9,
                        help='lambda parameter for computing beta from best beta')
    parser.add_argument('sample_strategy_id', default=1, type=int)  # 1, 2, 3
    parser.add_argument('paths', default=[os.path.join(get_data_dir('pour'), 'trials_n=10000.json')], nargs='*',
                        help='Paths to the data.')
    parser.add_argument('-u', '--use_hyper', action='store_true',
                        help='When enabled, use existing hyper parameter.')
    parser.add_argument('-o', '--use_obstacle', action='store_true',
                        help='When enabled, no obstacle is used in the scene.')

    args = parser.parse_args()
    beta_lambda = args.beta_lambda
    sample_strategy = SAMPLE_STRATEGIES[args.sample_strategy_id]
    global SEED
    SEED = args.expid
    set_seed(SEED)
    n_train_tasks = 50
    n_test_tasks = 20

    train_tasks_seeds = get_seeds(n_train_tasks)
    test_tasks_seeds = get_seeds(n_test_tasks)
    print('loading data')
    domain = load_data(args.paths)
    data = domain.create_dataset(include_none=True, binary=False)
    data.shuffle()
    X, Y, W = data.get_data()
    print('finished obtaining x y data')

    n_train = args.trainsize
    X = X[:n_train]
    Y = Y[:n_train]

    print('initializing ActiveGP with #datapoints = {}'.format(len(X)))

    hype = None

    if 'pour' in args.paths[0] and args.use_hyper:
        hype = POUR_MLP_HYPERPARAM_3000
    elif 'scoop' in args.paths[0] and args.use_hyper:
        hype = SCOOP_MLP_HYPERPARAM_3000

    learner = ActiveGP(domain, initx=X, inity=Y, hyperparameters=hype, sample_time_limit=60, beta_lambda=beta_lambda)
    learner.retrain(num_restarts=10)

    exp_file = 'tasklengthscale_sampling_trainsize={}_beta_lambdda={}_strategy_{}_obs_{}_expid_{}.pk3'.format(len(X),
                                                                                                              beta_lambda,
                                                                                                              args.sample_strategy_id,
                                                                                                              int(
                                                                                                                  args.use_obstacle),
                                                                                                              args.expid)
    exp_dirname = os.path.dirname(args.paths[0])
    if args.use_hyper:
        exp_dirname = os.path.join(exp_dirname,'default_hyper/')
        mkdir(exp_dirname)

    exp_file = os.path.join(exp_dirname, exp_file)
    print('saving results to ', exp_file)

    results = []

    if sample_strategy != DIVERSELK:
        n_train_tasks = 0
        # no need to train

    prev_tasklengthscale = None
    for i in range(n_train_tasks + 1):
        test_results = []
        print('task_lengthscal = {}'.format(learner.task_lengthscale))
        print('================BEGIN TESTING==============')
        if prev_tasklengthscale is not None and (learner.task_lengthscale == prev_tasklengthscale).all():
            test_results = results[-1][1]
        else:
            for j in range(n_test_tasks):
                if sample_strategy == DIVERSELK:
                    test_sample_strategy = DIVERSE
                else:
                    test_sample_strategy = sample_strategy
                seed = test_tasks_seeds[j]
                test_result = eval_task_with_seed(domain, seed, learner, sample_strategy=test_sample_strategy,
                                                  obstacle=args.use_obstacle)
                test_results.append(test_result)

        prev_tasklengthscale = learner.task_lengthscale.copy()

        if i != n_train_tasks:
            seed = train_tasks_seeds[i]
            train_result = eval_task_with_seed(domain, seed, learner, sample_strategy=sample_strategy,
                                               obstacle=args.use_obstacle)
        else:
            train_result = None



        results.append((train_result, test_results))
        write_pickle(exp_file, (results, SEED, train_tasks_seeds, test_tasks_seeds))


# TODO fix this func
def gen_exp_script(trainsize, beta_lambda, nexp, nparallel=10, skill='pour'):
    skill_dir = get_data_dir(skill)
    shfile = 'run_tasklengthscale_{}_exp.sh'.format(skill)

    with open(shfile, 'w') as f:
        lines = []

        for sample_strategy_id in [1, 2, 4, 3]:
            for obstacle in [True, False]:
                exp_dir = 'tasklengthscale_sampling_trainsize={}_beta_lambdda={}_strategy_{}_obs_{}'.format(trainsize,
                                                                                                            beta_lambda,
                                                                                                            sample_strategy_id,
                                                                                                            int(obstacle))
                exp_dir = os.path.join(skill_dir, exp_dir)
                for i in range(nexp): # , nexp*2
                    expfile = exp_dir + '_expid_{}.pk3'.format(i)
                    if not os.path.exists(expfile):
                        line = 'python3 -m learn_tools.run_taskkernel {} {} {} {} {}trials_n=10000.json -u '.format(trainsize,
                                                                                                                 i,
                                                                                                                 beta_lambda,
                                                                                                                 sample_strategy_id,
                                                                                                                 skill_dir)
                        if obstacle:
                            line += '-o '
                        line += '> {}.out &\n'.format(expfile)
                        lines += [line]
        for i in range(nparallel - 1, len(lines), nparallel):
            lines[i] = lines[i].replace('&', '')
        f.writelines(lines)


def plot():
    # success rate
    skill = 'pour' # 'scoop'
    paths = [os.path.join(get_data_dir(skill), 'trials_n=10000.json')]
    beta_lambda = 0.99
    success_rate = {}
    plan_score = {}
    trainsize = 3000
    use_obstacle = 1
    strategy_ids = [1, 2, 4, 3]
    planing_time = {}
    exprange = range(5) # range(5, 10) #
    exp_dir = os.path.join(os.path.dirname(paths[0]), 'default_hyper')
    print('trainsize={}\t beta_lambda={}\t use_obstacle={}'.format(trainsize, beta_lambda, use_obstacle))
    for expid in exprange:
        context = None
        for sample_strategy_id in strategy_ids:  # range(1,5):
            exp_file = 'tasklengthscale_sampling_trainsize={}_beta_lambdda={}_strategy_{}_obs_{}_expid_{}.pk3'.format(
                trainsize,
                beta_lambda,
                sample_strategy_id,
                int(use_obstacle),
                expid)
            exp_file = os.path.join(exp_dir, exp_file)

            results = read_pickle(exp_file)

            assert (context is None or (context == results[0][0][1][0].context).all())

            context = results[0][0][1][0].context

            prev_results = results

    for sample_strategy_id in strategy_ids:
        sample_strategy = SAMPLE_STRATEGIES[sample_strategy_id]
        success_rate[sample_strategy] = []
        plan_score[sample_strategy] = []
        minlen = 1 << 31
        planing_time_res = []
        succ_rate_res = []
        none_rate_res = []
        fail_rate_res = []
        for expid in exprange:


            for exp_dir in [os.path.join(os.path.dirname(paths[0]), 'default_hyper'), os.path.join(os.path.dirname(paths[0]), 'shakey_default_hyper')]:
                exp_file = 'tasklengthscale_sampling_trainsize={}_beta_lambdda={}_strategy_{}_obs_{}_expid_{}.pk3'.format(
                    trainsize,
                    beta_lambda,
                    sample_strategy_id,
                    int(use_obstacle),
                    expid)

                exp_file = os.path.join(exp_dir, exp_file)

                results = read_pickle(exp_file)
                res = []
                score = []
                for train, test in results[0]:
                    res.append(sum([t.scores >= 0 for t in test]) / len(test))
                    try:
                        planing_time_res.append(np.mean([t.plan_results['plan-time'] for t in test]))
                    except:
                        print('no plan results')

                    succ_rate_res.extend([1 if t.scores > 0 and t.plan_results['plan-time'] < 120 else 0 for t in test])
                    #none_rate_res.extend([1 if t.plan_results['score'] is None else 0 for t in test])
                    #fail_rate_res.extend([1 if t.scores < 0 and t.plan_results['score'] is not None else 0 for t in test])
                    score.append([len(t.samples) for t in
                                  test])  # ([.6**len(t.samples) if t.scores >= 0 else 0 for t in test]) # ([len(t.samples) for t in test])#
                    # if train:
                    #    print(*train.task_lengthscale)
                success_rate[sample_strategy].append(res)
                plan_score[sample_strategy].append(score)

                minlen = min(minlen, len(res))

        tmp = []
        tmpp = []
        for i in range(len(exprange)):
            tmp.append(success_rate[sample_strategy][i][:minlen])
            tmpp.append(plan_score[sample_strategy][i][:minlen])
        tmp = np.array(tmp)
        tmpp = np.array(tmpp)
        success_rate[sample_strategy] = (np.mean(tmp, axis=0), np.std(tmp, axis=0))
        plan_score[sample_strategy] = (np.mean(tmpp), np.std(tmpp))
        print(sample_strategy)
        print('plan time', np.mean(planing_time_res), np.std(planing_time_res))
        print('success rate', np.mean(succ_rate_res), np.std(succ_rate_res))
        print('none rate', np.mean(none_rate_res), np.std(none_rate_res))
        print('fail rate', np.mean(fail_rate_res), np.std(fail_rate_res))
    print(success_rate)
    print(plan_score)
    print()


if __name__ == '__main__':
    plot()

    # gen_exp_script(3000, 0.99, 5, nparallel=10, skill='scoop')

    #
    # scp -r ziw@shakey.csail.mit.edu:ltamp-pr2/data/pour_19-06-13_00-59-21/sampling* data/pour_19-06-13_00-59-21/
    main()
    # TODO: set beta by validity test in addition to lambdabeta
    # TODO: set a fixed GP hyper parameter by model selection via a validation set
    # TODO: use full dataset with plan failures
    # TODO: make the coffee machine thinner
