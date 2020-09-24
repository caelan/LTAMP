from __future__ import print_function

import time
from itertools import product, islice
from random import choice

from images.common import BOWLS, CUPS
from learn_tools.active_gp import STRADDLE_ACTIVE_GP, VAR_ACTIVE_GP, BATCH_MAXVAR_GP, BATCH_STRADDLE_GP
from learn_tools.active_rf import BATCH_STRADDLE_RF, BATCH_MAXVAR_RF
from learn_tools.active_learner import x_from_context_sample, STRADDLE, VARIANCE
from learn_tools.uncertain_learner import DEFAULT_ALPHA
from learn_tools.collect_simulation import get_trials, run_trials
from learn_tools.learner import SEPARATOR, SCORE
from pddlstream.utils import str_from_object
from plan_tools.common import SPOONS
from plan_tools.ros_world import ROSWorld
from plan_tools.samplers.pour import get_pour_feature
from plan_tools.samplers.scoop import get_scoop_feature
from pybullet_tools.utils import INF, randomize, ClientSaver, elapsed_time, implies

BATCH_VAR_ACTIVE = (BATCH_MAXVAR_GP, BATCH_MAXVAR_RF)
BATCH_STRADDLE_ACTIVE = (BATCH_STRADDLE_GP, BATCH_STRADDLE_RF)

# TODO: epsilon greedy strategy that scales epsilon

def generate_candidates(skill, bowls=BOWLS, cups=CUPS, spoons=SPOONS):
    # Could also just return any str values
    if skill == 'pour':
        feature_fn = get_pour_feature
        pairs = product(bowls, cups)
        attributes = ['bowl_type', 'cup_type']
    elif skill == 'scoop':
        feature_fn = get_scoop_feature
        pairs = product(bowls, spoons)
        attributes = ['bowl_type', 'spoon_type']
    else:
        raise NotImplementedError(skill)
    return feature_fn, attributes, list(pairs)

def sample_feature(skill, **kwargs):
    # TODO: instead compute this from the available results
    _, attributes, pairs = generate_candidates(skill, **kwargs)
    pair = choice(pairs) # Use empirical distribution
    print('Feature Pairs: {} | Sampled pair: {}'.format(len(pairs), pair))
    return dict(zip(attributes, pair))

def optimize_feature(learner, max_time=INF, **kwargs):
    #features = read_json(get_feature_path(learner.func.skill))
    feature_fn, attributes, pairs = generate_candidates(learner.func.skill, **kwargs)
    start_time = time.time()
    world = ROSWorld(use_robot=False, sim_only=True)
    saver = ClientSaver(world.client)
    print('Optimizing over {} feature pairs'.format(len(pairs)))

    best_pair, best_score = None, -INF # maximize
    for pair in randomize(pairs): # islice
        if max_time <= elapsed_time(start_time):
            break
        world.reset(keep_robot=False)
        for name in pair:
            world.perception.add_item(name)
        feature = feature_fn(world, *pair)
        parameter = next(learner.parameter_generator(world, feature, min_score=best_score, valid=True, verbose=False), None)
        if parameter is None:
            continue
        context = learner.func.context_from_feature(feature)
        sample = learner.func.sample_from_parameter(parameter)
        x = x_from_context_sample(context, sample)
        #x = learner.func.x_from_feature_parameter(feature, parameter)
        score_fn = learner.get_score_f(context, noise=False, negate=False) # maximize
        score = float(score_fn(x)[0, 0])
        if best_score < score:
            best_pair, best_score = pair, score
    world.stop()
    saver.restore()
    print('Best pair: {} | Best score: {:.3f} | Pairs: {} | Time: {:.3f}'.format(
        best_pair, best_score, len(pairs), elapsed_time(start_time)))
    assert best_score is not None
    # TODO: ensure the same parameter is used
    return dict(zip(attributes, best_pair))

##################################################

def query_from_name(name):
    if name in STRADDLE_ACTIVE_GP:
        return STRADDLE
    if name in VAR_ACTIVE_GP:
        return VARIANCE
    raise ValueError(name)

def dump_result(result):
    for field in sorted(result):
        print('{}: {}'.format(field, str_from_object(result[field])))

def active_learning(learner, num, discrete_feature=False, random_feature=False, include_none=True, visualize=False):
    online = []
    if not learner.trained:
        learner.retrain()
    if not num:
        return online
    original_query_type = learner.query_type
    attempts = 0
    # TODO: heuristic that we keep going into we have X scored successes
    while len(online) < num:
        print(SEPARATOR)
        print('{}/{} active samples after {} attempts'.format(len(online), num, attempts))
        attempts += 1
        #learner.query_type = STRADDLE if np.random.random() < 0.5 else BEST # epsilon greedy
        learner.query_type = query_from_name(learner.algorithm.name)
        print('Query type:', learner.query_type)
        #seed = ('active', learner.nx) # attempts
        seed = time.time()
        collect_kwargs = {}
        if discrete_feature:
            collect_kwargs = {'randomize': False}
            if random_feature:
                collect_kwargs.update(sample_feature(learner.func.skill))
            else:
                collect_kwargs.update(optimize_feature(learner))

        trials = get_trials(problem=learner.func.skill, fn=learner, num_trials=1,
                            seed=seed, visualize=visualize, verbose=True,
                            collect_kwargs=collect_kwargs)
        [result] = run_trials(trials, num_cores=False)
        if result is None or not implies(result.get(SCORE, None) is None, include_none):
            continue

        print('\nNum: {}'.format(len(online)))
        dump_result(result)
        print()
        online.append(result)
        x, y, w = learner.func.example_from_result(result, binary=False)
        learner.retrain(newx=[x], newy=[y], new_w=[w])
        learner.results.append(result)
        #new_parameter = next(learner.parameter_generator(None, feature))
        #print(new_parameter)
    learner.query_type = original_query_type
    return online

##################################################

def optimize_index(learner, select_data, indices, noise=False):
    X_select, _, _ = select_data.get_data()
    X_select = X_select[indices]
    alg_name = learner.algorithm.name
    if alg_name in BATCH_STRADDLE_ACTIVE:
        idx = learner.get_highest_lse_idx(X_select, alpha=DEFAULT_ALPHA, noise=noise)  # DEFAULT_ALPHA
    elif alg_name in BATCH_VAR_ACTIVE:
        idx = learner.get_highest_var_idx(X_select, noise=noise)  # Noise shouldn't affect
    else:
        raise NotImplementedError(alg_name)
    return indices[idx]

def optimize_class(learner, select_data, indices, noise=False):
    # TODO: average score across the class and select
    # TODO: randomly sample from the selected class
    #feature_fn, attributes, pairs = generate_candidates(learner.func.skill, **kwargs)
    raise NotImplementedError()

def active_learning_discrete(learner, select_data, num=1, random_feature=False):
    online = []
    if not learner.trained:
        learner.retrain() #num_restarts=10, num_processes=1)
    if not num:
        return online
    if learner.algorithm.label is not None:
        random_feature = learner.algorithm.label.endswith('2')
    # TODO: subsample the discrete context
    alg_name = learner.algorithm.name
    while len(online) < num:
        print(SEPARATOR)
        # new example
        # No need to to change learner.query_type
        if random_feature:
            # TODO: this could sample a removed bowl
            value_from_attr = sample_feature(learner.func.skill)
            indices = [i for i, result in enumerate(select_data.results)
                       if all(result['feature'][attr] == value for attr, value in value_from_attr.items())]
            if not indices:
                continue
        else:
            indices = list(range(len(select_data)))
        idx = optimize_index(learner, select_data, indices)
        print('Acquisition: {} | {}/{} active samples | {} candidates | selected index: {}'.format(
            alg_name, len(online), num, len(indices), idx)) # TODO: print the score
        #idx = learner.select_index(X_select)
        result = select_data.results.pop(idx) # TODO: GP should automatically avoid old samples
        print('Feature:', result['feature'])
        online.append(result)
        x, y, w = learner.func.example_from_result(result, binary=False)
        learner.retrain(newx=[x], newy=[y], new_w=[w])
        learner.results.append(result)
    return online