from __future__ import print_function

import argparse
import datetime
import os
import sys
import time
import random
import math
import numpy as np

sys.path.extend([
    os.path.join(os.getcwd(), 'pddlstream'), # Important to use absolute path when doing chdir
    os.path.join(os.getcwd(), 'ss-pybullet'),
])

from collections import Counter
from itertools import product

from pybullet_tools.utils import is_darwin, user_input, elapsed_time, write_pickle, read_pickle, randomize
from pddlstream.utils import get_python_version, mkdir, implies

from learn_tools.statistics import get_scored_results
from learn_tools.select_active import active_learning, active_learning_discrete
from learn_tools.analyze_experiment import get_label, plot_experiments, Algorithm, \
    Experiment, compute_metrics
from learn_tools.learnable_skill import load_data, Dataset
from learn_tools.learner import score_accuracy, DATA_DIRECTORY, SEPARATOR, inclusive_range
from learn_tools.collect_simulation import run_trials, get_trials, get_num_cores, HOURS_TO_SECS
from learn_tools.common import get_max_cores, DATE_FORMAT
from learn_tools.learner import FEATURE, PARAMETER, SCORE  # LEARNED
from learn_tools.active_learner import BEST
from learn_tools.active_nn import ActiveNN, NN_MODELS, RF_CLASSIFIER, get_sklearn_model, RF_REGRESSOR
from learn_tools.active_gp import ActiveGP, HYPERPARAM_FROM_KERNEL, BATCH_GP, GP_MODELS, \
    BATCH_ACTIVE_GP, CONTINUOUS_ACTIVE_GP, BATCH_STRADDLE_GP, BATCH_MAXVAR_GP, VAR_ACTIVE_GP, \
    STRADDLE_GP, MAXVAR_GP
from learn_tools.active_rf import ActiveRF, RF_MODELS, BATCH_STRADDLE_RF, BATCH_ACTIVE_RF, BATCH_MAXVAR_RF, BATCH_RF

BATCH_ACTIVE = BATCH_ACTIVE_GP + BATCH_ACTIVE_RF

#np.show_config()
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
# TODO: multiprocessing predictions don't seem to work at all on OS X

BALANCED = 'balanced'
UNIFORM = 'uniform'
ALL = 'all'
SPLIT_TYPES = (BALANCED, UNIFORM, ALL)

SEC_PER_EXPERIMENT = 60.0 # 30.9035954114
# (2.695*60*60) / (10000/45)

#UNCERTAIN_LEARNERS = RF_MODELS + GP_MODELS

##################################################

def split_data(all_data, num_train, min_classes=2):
    while True:
        train_data, test_data = all_data.partition(num_train)
        X, Y, W = train_data.get_data()
        if min_classes <= len(Counter(Y.tolist())):
            return train_data, test_data
        all_data.shuffle()

##################################################

def create_learner(domain, train_data, algorithm, **kwargs):
    # TODO: recover domain from train_data
    X_train, Y_train, W_train = train_data.get_data()

    #X_train, Y_train, _, _ = split_data(X, Y, split, num_train)
    print('Training examples: {} | Average Score: {:.3f}'.format(len(Y_train), np.average(Y_train)))
    #labels = threshold_scores(Y_train)
    #labels = Y_train
    #frequencies = Counter(labels)
    #for c in sorted(frequencies):
    #    print('Label={}: {}/{}={:.3f}'.format(c, frequencies[c], len(labels), float(frequencies[c]) / len(labels)))
    #frequencies = Counter(W_train)
    #for c in sorted(frequencies):
    #    print('Weight={}: {}/{}={:.3f}'.format(c, frequencies[c], len(W_train), float(frequencies[c]) / len(W_train)))

    #start_time = time.time()
    if algorithm.name in GP_MODELS:
        hyperparameters = HYPERPARAM_FROM_KERNEL[domain.skill, algorithm.kernel] \
            if algorithm.hyperparameters else None
        #hyperparameters = None
        #kern_var_range = (math.pow(10, -algorithm.transfer_weight),
        #                  math.pow(10, +algorithm.transfer_weight))
        learner = ActiveGP(domain, initx=X_train, inity=Y_train, initw=W_train, kernel_name=algorithm.kernel,
                           hyperparameters=hyperparameters, use_var=algorithm.variance,
                           #kern_var_range=kern_var_range,
                           transfer_weight=algorithm.transfer_weight,
                           **kwargs)
    elif algorithm.name in RF_MODELS:
        learner = ActiveRF(domain, initx=X_train, inity=Y_train, initw=W_train,
                           model_type=algorithm.name, use_var=algorithm.variance,
                           transfer_weight=algorithm.transfer_weight, **kwargs)
    elif algorithm.name in NN_MODELS:
        learner = ActiveNN(domain, initx=X_train, inity=Y_train, initw=W_train,
                           model_type=algorithm.name, transfer_weight=algorithm.transfer_weight, **kwargs)
    else:
        raise ValueError(algorithm.name)
    #learner.name = '{}_{}_n{}_a{}'.format(learner.name, algorithm.split, num_train, num_active)
    learner.algorithm = algorithm
    learner.results.extend(train_data.results)
    learner.num_batch += len(train_data)

    #num_transfer = 0
    #if num_transfer != 0:
    #    X_transfer, Y_transfer = X[-num_transfer:], Y[-num_transfer:]
    #    learner.add_data(X_transfer, Y_transfer, weights=0.1*np.ones(Y_transfer.shape))

    # run_ActiveLearner(learner)
    # train learner with training data
    #learner.retrain()
    #save_learner(learner)
    #print('Training examples: {} | Training time: {:.3f}'.format(Y_train.shape[0], time.time() - start_time))
    #confusion = evaluate_confusions(X_train, Y_train, algorithm, learner, num_train, alphas, header='Train metrics:')
    confusion = None
    return learner, confusion

##################################################

def evaluate_confusions(test_data, learner, alphas=None, serial=False, header=None, verbose=True):
    if len(test_data) == 0:
        return None
    if alphas is None:
        #alphas = [1.0] # Uses beta (previously was None)
        alphas = np.linspace(0.0, 0.9, num=5, endpoint=True) # linspace | logspace
        #alphas = [0.0, 0.8, 0.9, 0.95, 0.99]
    #X, Y, _ = test_data.get_data() # TODO: normal for RMSE and binary for confusion
    X, Y, _ = test_data.func.examples_from_results(test_data.results, binary=True) # TODO: remove None

    # TODO: kernel density estimation
    max_cores = get_max_cores(serial=serial)
    algorithm = learner.algorithm
    if algorithm.name in NN_MODELS:
        alphas = [0.0]
    if verbose and (header is not None):
        print(header)
    confusions = {}
    for alpha in alphas:
        if verbose:
            print('Alpha={:.3f}'.format(alpha), end=', ')
        Y_pred = learner.score_x(X, alpha=alpha, max_cores=max_cores) # max_cores is not used
        confusions[alpha] = compute_metrics(Y, Y_pred, verbose=verbose)
    return confusions[alphas[0]]
    # TODO: store predictions rather than confusions
    # alphas = list(np.linspace(0.0, 1.0, num=25, endpoint=True)) # linspace | logspace
    # predictions = learner.predict_x(X)
    # best_alpha, best_recall = optimize_recall(Y, predictions, min_precision=0.9, alphas=alphas)
    # print('Best alpha: {:.3f} | best recall: {:.3f}'.format(best_alpha, best_recall))
    # Y_pred = np.array([score_prediction(prediction, alpha=best_alpha) for prediction in predictions])
    # return compute_metrics(Y, Y_pred)

def evaluate_scores(domain, seed, algorithm, learner, num_trials,
                    serial=False, visualize=False):
    results = []
    if num_trials == 0:
        return results
    start_time = time.time()
    # trials selected by learner, num_trials is number of contexts that the learner evaluated upon
    trials = get_trials(problem=domain.skill, fn=learner, num_trials=num_trials,
                        seed=seed, visualize=visualize, verbose=serial)
    num_cores = get_num_cores(trials, serial=serial)
    results = run_trials(trials, num_cores=num_cores)
    scored = get_scored_results(results)
    scores = [domain.score_fn(result[FEATURE], result[PARAMETER], result[SCORE])
              for result in scored]
    # TODO: record best predictions as well as average prediction quality
    print('Seed: {} | {} | Results: {} | Scored: {} | Time: {:.3f}'.format(
        seed, algorithm, len(results), len(scored), elapsed_time(start_time)))
    if scores:
        print('Average: {:.3f} | Success: {:.1f}%'.format(
            np.average(scores), 100 * score_accuracy(scores)))
    return results

def evaluate_learner(domain, seed, train_confusion, test_data, algorithm, learner,
                     train_size, num_trials, serial=False, visualize=False):
    test_confusion = evaluate_confusions(test_data, learner, serial=serial, header='Test metrics:')
    #algorithm = learner.algorithm
    confusion = {
        'train': train_confusion,
        'test': test_confusion,
    }
    results = evaluate_scores(domain, seed, algorithm, learner, num_trials, serial=serial, visualize=visualize)
    return Experiment(algorithm, train_size, confusion, results)
    # tuple saved to experiments.pk3

def save_experiments(data_path, experiments):
    if data_path is None:
        return None
    write_pickle(data_path, experiments)
    print('Saved experiments:', data_path)
    return data_path

def create_validity_classifier(transfer_domain):
    validity_data = transfer_domain.create_dataset(include_invalid=False, include_none=True, validity=True)
    # validity_data, test_data = validity_data.partition(9000)
    # X_train, Y_train, _ = validity_data.get_data()
    # validity_model = get_sklearn_model(RF_CLASSIFIER, n_estimators=500)
    # validity_model.fit(X_train, Y_train)
    # X_test, Y_test, _ = test_data.get_data()
    # print('Train: {:.3f} | OOB: {:.3f} | Test: {:.3f}'.format(
    #     validity_model.score(X_train, Y_train), validity_model.oob_score_, validity_model.score(X_test, Y_test)))
    # return validity_model
    validity_algorithm = Algorithm(RF_CLASSIFIER, variance=False)
    validity_learner, _ = create_learner(transfer_domain, validity_data, validity_algorithm, verbose=True)
    validity_learner.retrain()
    return validity_learner

##################################################

OLD_DATASETS = [
    'data/pr2_pour_19-05-24_12-40-11/trials.json', # yellow_bowl, blue_bowl
    'data/pr2_pour_19-05-31_11-10-47/trials.json', # whitebowl
    'data/pr2_pour_19-06-01_12-14-55/trials.json', # yellow_bowl, green_bowl, red_bowl
    'data/pr2_pour_19-06-01_12-58-24/trials.json', # red_bowl, yellow_bowl
    'data/pr2_pour_19-06-03_17-24-58/trials.json', # blue_white_bowl, blue_bowl, tan_bowl
    'data/pr2_pour_19-06-04_17-11-18/trials.json', # blue_bowl, tan_bowl
    'data/pr2_pour_19-06-10_15-23-57/trials.json', # blue_white_bowl, yellow_bowl, red_speckled_bowl
    'data/pr2_pour_19-07-19_18-40-29/trials.json', # red_speckled_bowl, purple_cup
    # 'data/pr2_pour_all/trials.json',
]

TRAIN_DATASETS = [
    # Caelan
    'data/pr2_pour_19-11-28_13-49-49/trials.json',
    'data/pr2_pour_19-11-28_14-35-27/trials.json',
    'data/pr2_pour_19-11-28_15-37-42/trials.json',
    'data/pr2_pour_19-11-28_16-23-58/trials.json',
    # Zi
    'data/pr2_pour_19-11-29_09-45-29/trials.json',
    'data/pr2_pour_19-11-29_11-19-25/trials.json',
    #'test.json',
]
# data/pr2_pour_19-11-28_13-49-49/trials.json data/pr2_pour_19-11-28_14-35-27/trials.json data/pr2_pour_19-11-28_15-37-42/trials.json data/pr2_pour_19-11-28_16-23-58/trials.json

ACTIVE_DATASETS = [
    # 'data/pr2_pour_19-12-09_18-41-12/trials.json',
    # 'data/pr2_pour_19-12-16_15-48-13/trials.json',
    # 'data/pr2_pour_19-12-17_16-46-15/trials.json',
    # 'data/pr2_pour_19-12-17_16-57-57/trials.json',
    # 'data/pr2_pour_19-12-18_13-17-30/trials.json',
    # 'data/pr2_pour_19-12-18_13-43-36/trials.json',
    # 'data/pr2_pour_19-12-19_13-25-42/trials.json',
    # 'data/pr2_pour_19-12-19_14-05-43/trials.json',
    # 'data/pr2_pour_19-12-19_14-45-38/trials.json',
    # 'data/pr2_pour_19-12-19_16-05-45/trials.json',
    # 'data/pr2_pour_19-12-20_14-14-45/trials.json',
    # 'data/pr2_pour_20-01-06_14-24-42/trials.json',
    # 'data/pr2_pour_20-01-06_17-25-02/trials.json',
    # 'data/pr2_pour_20-01-07_12-19-01/trials.json',
    # 'data/pr2_pour_20-01-07_13-39-06/trials.json',
    # 'data/pr2_pour_20-01-09_14-44-17/trials.json',
    'data/pr2_pour_20-01-09_16-26-01/trials.json',
    # 'data/pr2_pour_20-01-10_12-35-38/trials.json',
    # 'data/pr2_pour_20-01-10_13-20-09/trials.json',
    # 'data/pr2_pour_20-01-10_14-43-48/trials.json',
    # 'data/pr2_pour_20-01-10_16-17-12/trials.json',
]
#TRAIN_DATASETS += ACTIVE_DATASETS

TEST_DATASETS = [
    # Caelan
    'data/pr2_pour_19-11-27_12-49-19/trials.json',
    'data/pr2_pour_19-11-27_16-25-15/trials.json',
    # Zi
    'data/pr2_pour_19-11-30_12-21-35/trials.json',
    'data/pr2_pour_19-11-30_13-32-25/trials.json',
]
#TRAIN_DATASETS += TEST_DATASETS

BEST_DATASETS = [
    'data/pr2_pour_20-01-13_16-24-39/trials.json',
    'data/pr2_pour_20-01-13_17-23-53/trials.json',
    'data/pr2_pour_20-01-13_18-28-16/trials.json',
]

TRANSFER_DATASETS = [
    #'data/pour_19-11-22_13-08-55/trials_n=10000.json', # 6228/10000 scored
    'data/pour_19-12-09_22-16-07/trials_n=10000.json', # 10000 examples, 7050 valid, 6196 scored
]

##################################################

# TODO: separate file for datasets

SCOOP_TRAIN_DATASETS = [
    'data/pr2_scoop_20-01-22_15-16-47/trials.json',
    'data/pr2_scoop_20-01-22_16-12-44/trials.json',
    'data/pr2_scoop_20-01-23_13-45-28/trials.json',
    'data/pr2_scoop_20-01-23_14-07-14/trials.json',
    'data/pr2_scoop_20-01-23_14-14-53/trials.json',
    'data/pr2_scoop_20-01-23_14-27-17/trials.json',
    'data/pr2_scoop_20-01-23_15-25-13/trials.json',
    'data/pr2_scoop_20-01-23_15-32-04/trials.json',
]

SCOOP_TEST_DATASETS = [
    #'data/pr2_scoop_20-01-14_15-58-50/trials.json', # orange_spoon
    #'data/pr2_scoop_20-01-14_16-59-16/trials.json', # grey_spoon
    'data/pr2_scoop_20-01-14_17-29-54/trials.json', # green_spoon
    'data/pr2_scoop_20-01-15_15-23-44/trials.json',
    'data/pr2_scoop_20-01-21_10-44-55/trials.json',
    'data/pr2_scoop_20-01-21_14-20-47/trials.json',
    'data/pr2_scoop_20-01-21_15-05-29/trials.json',
    'data/pr2_scoop_20-01-22_14-05-18/trials.json',
]
#SCOOP_TRAIN_DATASETS += SCOOP_TEST_DATASETS

##################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*', help='Paths to the data.')
    #parser.add_argument('-a', '--active', type=int, default=0, # None
    #                    help='The number of active samples to collect')
    parser.add_argument('-d', '--deterministic', action='store_true',
                        help='Whether to deterministically create training splits')
    parser.add_argument('-n', '--num_trials', type=int, default=-1,
                        help='The number of samples to collect')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Whether to save the learners')
    parser.add_argument('-r', '--num_rounds', type=int, default=1,
                        help='The number of rounds to collect')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Whether to save the data')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='When enabled, visualizes execution.')
    args = parser.parse_args()

    # TODO: be careful that paging isn't altering the data
    # TODO: use a different set of randomized parameters for train and test

    serial = is_darwin()
    visualize = serial and args.visualize
    assert implies(visualize, serial)
    num_trials = get_max_cores(serial) if args.num_trials < 0 else args.num_trials

    ##################################################

    #train_sizes = inclusive_range(50, 200, 10) # Best
    #train_sizes = inclusive_range(50, 400, 10) # F1
    #train_sizes = inclusive_range(25, 400, 25)
    #train_sizes = inclusive_range(50, 100, 5) # Real
    #train_sizes = inclusive_range(100, 200, 5)
    #train_sizes = inclusive_range(10, 250, 5)
    #train_sizes = inclusive_range(35, 70, 5)
    #train_sizes = inclusive_range(5, 50, 5)
    #train_sizes = inclusive_range(40, 80, 5)
    #train_sizes = inclusive_range(100, 1000, 100)
    #train_sizes = [50]
    #train_sizes = [250]
    train_sizes = [1000]
    #train_sizes = [327] # train + test
    #train_sizes = inclusive_range(5, 150, 25)
    #train_sizes = [100]

    #kernels = ['RBF', 'Matern52', 'MLP']
    kernels = ['MLP']

    hyperparams = [None]
    #hyperparams = [True]
    #hyperparams = [None, True]

    query_type = BEST # BEST | CONFIDENT | REJECTION | ACTIVE # type of query used to evaluate the learner

    include_none = False
    binary = False

    # 0 => no transfer
    # 1 => mean transfer
    # 2 => kernel transfer
    # 3 => both transfer
    transfer_weights = [None]
    #transfer_weights = list(range(4))
    #transfer_weights = [0, 1]
    #transfer_weights = [3]
    #transfer_weights = np.around(np.linspace(0.0, 1.0, num=1+5, endpoint=True), decimals=3) # max 10 colors
    #transfer_weights = list(range(1, 1+3))

    #split = UNIFORM # BALANCED
    #print('Split:', split)
    #parameters = {
    #    'include None': include_none,
    #    'binary': binary,
    #    'split': split,
    #}

    # Omitting failed labels is okay because they will never be executed
    algorithms = []
    #algorithms += [(Algorithm(nn_model, label='NN'), [num])
    #              for nn_model, num in product(NN_MODELS, train_sizes)]
    #algorithms += [(Algorithm(RANDOM), None), (Algorithm(DESIGNED), None)]

    #algorithms += [(Algorithm(RF_CLASSIFIER, variance=False, transfer_weight=tw, label='RF'), [num])
    #                for num, tw in product(train_sizes, [None])] # transfer_weights
    #algorithms += [(Algorithm(RF_REGRESSOR, variance=False, transfer_weight=tw, label='RF'), [num])
    #                for num, tw in product(train_sizes, [None])] # transfer_weights
    #algorithms += [(Algorithm(BATCH_RF, variance=True, transfer_weight=tw, label='RF'), [num])
    #                for num, tw in product(train_sizes, [None])] # transfer_weights
    #algorithms += [(Algorithm(BATCH_MAXVAR_RF, variance=True, transfer_weight=tw), train_sizes)
    #                for tw in product(use_vars, [None])] # transfer_weights
    #algorithms += [(Algorithm(BATCH_STRADDLE_RF, variance=True, transfer_weight=tw), train_sizes)
    #                for tw, in product([None])] # transfer_weights

    use_vars = [True]
    # STRADDLE is better than MAXVAR when the learner has a good estimate of uncertainty
    algorithms += [(Algorithm(BATCH_GP, kernel, hype, use_var, tw, label='GP'), [num]) # label='GP-{}'.format(kernel)
                   for num, kernel, hype, use_var, tw in product(train_sizes, kernels, hyperparams, use_vars, transfer_weights)]
    #algorithms += [(Algorithm(BATCH_MAXVAR_GP, kernel, hype, True, tw, label='GP-Var'), train_sizes)
    #                for kernel, hype, tw in product(kernels, hyperparams, transfer_weights)]
    #algorithms += [(Algorithm(BATCH_STRADDLE_GP, kernel, hype, True, tw, label='GP-LSE'), train_sizes)
    #                for kernel, hype, tw in product(kernels, hyperparams, transfer_weights)] # default active
    #algorithms += [(Algorithm(BATCH_STRADDLE_GP, kernel, hype, True, tw, label='GP-LSE2'), train_sizes)
    #                for kernel, hype, tw in product(kernels, hyperparams, transfer_weights)] # active control only

    # algorithms += [(Algorithm(MAXVAR_GP, kernel, hype, use_var), train_sizes)
    #                for kernel, hype, use_var in product(kernels, hyperparams, use_vars)]
    #algorithms += [(Algorithm(STRADDLE_GP, kernel, hype, use_var, tw), train_sizes)
    #                for kernel, hype, use_var, tw in product(kernels, hyperparams, use_vars, transfer_weights)]

    #batch_sizes = inclusive_range(train_sizes[0], 90, 10)
    #step_size = 10 # TODO: extract from train_sizes
    #final_size = train_sizes[-1]
    # Previously didn't have use_var=True
    # algorithms += [(Algorithm(BATCH_STRADDLE_GP, kernel, hyperparameters=batch_size, variance=True, transfer_weight=tw),
    #                 inclusive_range(batch_size, final_size, step_size))
    #                for kernel, tw, batch_size in product(kernels, transfer_weights, batch_sizes)]
    # algorithms += [(Algorithm(BATCH_STRADDLE_RF, hyperparameters=batch_size, variance=True, transfer_weight=tw),
    #                 inclusive_range(batch_size, final_size, step_size))
    #                 for tw, batch_size in product(transfer_weights, batch_sizes)]

    print('Algorithms:', algorithms)

    ##################################################

    real_world = not args.paths
    transfer_domain = load_data(TRANSFER_DATASETS, verbose=False)
    transfer_algorithm = None
    if real_world and transfer_weights != [None]:
        #assert transfer_weights[0] is not None
        transfer_data = transfer_domain.create_dataset(include_none=include_none, binary=binary)
        transfer_algorithm = Algorithm(BATCH_GP, kernel=kernels[0], variance=use_vars[0])

    validity_learner = None
    #validity_learner = create_validity_classifier(transfer_domain)

    ##################################################

    train_paths = args.paths
    if real_world:
        train_paths = SCOOP_TRAIN_DATASETS # TRAIN_DATASETS
        #train_paths = TRANSFER_DATASETS
        #train_paths = TRAIN_DATASETS + TRANSFER_DATASETS # Train before transfer
    #scale_paths = TRAIN_DATASETS + TEST_DATASETS
    scale_paths = None
    print(SEPARATOR)
    print('Train paths:', train_paths)
    domain = load_data(train_paths)
    print()
    print(domain)
    all_data = domain.create_dataset(include_none=include_none, binary=binary, scale_paths=scale_paths)
    #all_data.results = all_data.results[:1000]

    num_failed = 0
    #num_failed = 100
    failed_domain = transfer_domain if real_world else domain
    failed_results = randomize(result for result in failed_domain.results
                               if not result.get('success', False))[:num_failed]
    #failed_data = Dataset(domain, failed_results, **all_data.kwargs)

    test_paths = SCOOP_TEST_DATASETS # TEST_DATASETS | SCOOP_TEST_DATASETS
    #test_paths = None
    if real_world and not (set(train_paths) & set(test_paths)):
        #assert not set(train_paths) & set(test_paths)
        #max_test = 0
        test_data = load_data(test_paths).create_dataset(include_none=False, binary=binary, scale_paths=scale_paths)
    else:
        #assert scale_paths is None # TODO: max_train will be too small otherwise
        test_paths = test_data = None
    print(SEPARATOR)
    print('Test paths:', test_paths)

    all_active_data = None
    #if real_world:
    #    all_active_data = load_data(ACTIVE_DATASETS).create_dataset(include_none=True, binary=binary, scale_paths=scale_paths)

    # TODO: could include OS and username if desired
    date_name = datetime.datetime.now().strftime(DATE_FORMAT)
    size_str = '[{},{}]'.format(train_sizes[0], train_sizes[-1])
    #size_str = '-'.join(map(str, train_sizes))
    experiments_name = '{}_r={}_t={}_n={}'.format(date_name, args.num_rounds, size_str, num_trials)

    trials_per_round = sum(1 if train_sizes is None else
                           (train_sizes[-1] - train_sizes[0] + len(train_sizes))
                           for _, train_sizes in algorithms)
    num_experiments = args.num_rounds*trials_per_round
    max_train = min(max([0] + [active_sizes[0] for _, active_sizes in algorithms
                               if active_sizes is not None]), len(all_data))
    max_test = min(len(all_data) - max_train, 1000)

    ##################################################

    # #features = ['bowl_height']
    # features = ['spoon_height']
    # #features = ['bowl_height', 'spoon_height']
    # X, Y, _ = all_data.get_data()
    # #indices = [domain.inputs.index(feature) for feature in features]
    # #X = X[:,indices]
    # X = [[result[FEATURE][name] for name in features] for result in all_data.results]
    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression(fit_intercept=True, normalize=False)
    # model.fit(X, Y)
    # #print(model.get_params())
    # print(model.coef_.tolist(), model.intercept_)
    # print(model.score(X, Y))

    #data_dir = os.path.join(DATA_DIRECTORY, domain.name) # EXPERIMENT_DIRECTORY
    data_dir = os.path.abspath(os.path.join(domain.name, os.path.pardir))
    experiments_dir, data_path = None, None
    if not args.test or not serial:
        experiments_dir = os.path.join(data_dir, experiments_name)
        data_path = os.path.join(experiments_dir, 'experiments.pk{}'.format(get_python_version()))

    ##################################################

    print(SEPARATOR)
    print('Name:', experiments_name)
    print('Experiments:', num_experiments)
    print('Experiment dir:', experiments_dir)
    print('Data path:', data_path)
    print('Examples:', len(all_data))
    print('Valid:', sum(result.get('valid', True) for result in all_data.results))
    print('Success:', sum(result.get('success', False) for result in all_data.results))
    print('Scored:', sum(result.get('score', None) is not None for result in all_data.results))
    print('Max train:', max_train)
    print('Max test:', max_test)
    print('Include None:', include_none)
    print('Examples: n={}, d={}'.format(len(all_data), domain.dx))
    print('Binary:', binary)
    print('Serial:', serial)
    print('Estimated hours: {:.3f}'.format(num_experiments * SEC_PER_EXPERIMENT / HOURS_TO_SECS))
    user_input('Begin?')

    ##################################################

    experiments = []
    if experiments_dir is not None:
        mkdir(experiments_dir)
        # if os.path.exists(data_path):
        #     experiments.extend(read_pickle(data_path))

    # TODO: embed in a KeyboardInterrupt to allow early termination
    start_time = time.time()
    for round_idx in range(args.num_rounds):
        seed = round_idx if args.deterministic else hash(time.time()) # vs just time.time()?
        random.seed(seed)
        all_data.shuffle()
        if test_paths is None: # cannot use test_data
            #test_data, train_data = split_data(all_data, max_test)
            train_data = test_data = all_data # Training performance
        else:
            train_data = all_data

        transfer_learner = None
        if transfer_algorithm is not None:
            round_data, _ = transfer_data.partition(index=1000)
            transfer_learner, _ = create_learner(transfer_domain, round_data, transfer_algorithm, verbose=True)
            transfer_learner.retrain()

        print(SEPARATOR)
        print('Round {} | Train examples: {} | Test examples: {}'.format(round_idx, len(train_data), len(test_data)))
        for algorithm, active_sizes in algorithms:
            # active_sizes = [first #trainingdata selected from X_train, #active exploration + #trainingdata]
            print(SEPARATOR)
            print('Round: {} | {} | Seed: {} | Sizes: {}'.format(round_idx, algorithm, seed, active_sizes))
            # TODO: allow keyboard interrupt
            if active_sizes is None:
                learner = algorithm.name
                active_size = train_confusion = None
                experiments.append(evaluate_learner(domain, seed, train_confusion, test_data, algorithm, learner,
                                                    active_size, num_trials, serial, args.visualize))
                continue
            # [10 20 25] take first 10 samples from X_train to train the model, 10 samples chosen actively
            # sequentially + evaluate model, 5 samples chosen actively sequentially + evaluate model
            # Could always keep around all the examples and retrain
            # TODO: segfaults when this runs in parallel
            # TODO: may be able to retrain in parallel if I set OPENBLAS_NUM_THREADS
            num_batch = active_sizes[0]
            batch_data, active_data = train_data.partition(num_batch)
            if all_active_data is not None:
                active_data = all_active_data.clone()

            #batch_data.results.extend(failed_results)
            learner, train_confusion = create_learner(domain, batch_data, algorithm, # alphas,
                                                      query_type=query_type, verbose=True)
            learner.validity_learner = validity_learner
            if transfer_learner is not None:
                learner.sim_model = transfer_learner.model
            learner.retrain()
            for active_size in active_sizes:
                num_active = active_size - (learner.nx - len(failed_results))
                print('\nRound: {} | {} | Seed: {} | Size: {} | Active: {}'.format(
                    round_idx, algorithm, seed, active_size, num_active))
                if algorithm.name in CONTINUOUS_ACTIVE_GP:
                    active_learning(learner, num_active, visualize=visualize)
                    #active_learning(learner, num_active, discrete_feature=True, random_feature=False)
                    #active_learning_discrete(learner, active_data, num_active, random_feature=False)
                elif algorithm.name in BATCH_ACTIVE:
                    active_learning_discrete(learner, active_data, num_active)
                    #active_learning(learner, num_active, discrete_feature=True, random_feature=True)
                    #active_learning_discrete(learner, active_data, num_active, random_feature=True)
                #if round_dir is not None:
                #    save_learner(round_dir, learner)
                if args.save:
                    learner.save(data_dir)
                experiments.append(evaluate_learner(domain, seed, train_confusion, test_data,
                                                    algorithm, learner, active_size, num_trials,
                                                    serial, args.visualize))
                save_experiments(data_path, experiments)

    print(SEPARATOR)
    if experiments:
        save_experiments(data_path, experiments)
        plot_experiments(domain, experiments_name, experiments_dir, experiments, include_none=False)
        print('Experiments: {}'.format(experiments_dir))
    print('Total experiments: {}'.format(len(experiments)))
    print('Total hours: {:.3f}'.format(elapsed_time(start_time) / HOURS_TO_SECS))

##################################################

# TODO: Multiprocessing with numpy makes Python quit unexpectedly on OSX
# https://stackoverflow.com/questions/19705200/multiprocessing-with-numpy-makes-python-quit-unexpectedly-on-osx

# TODO: Unable to quickly train models in parallel due to scipy
# https://stackoverflow.com/questions/10025866/parallel-linear-algebra-for-multicore-system
# https://github.com/SheffieldML/GPy/issues/330

# ModuleNotFoundError: No module named 'numpy.core._multiarray_umath'
# https://github.com/alpacahq/pylivetrader/issues/73
# https://stackoverflow.com/questions/54200850/attributeerror-tuple-object-has-no-attribute-type-upon-importing-tensorflow
# https://github.com/numpy/numpy/issues/12977

# Final datasets
# python3 -m learn_tools.run_active data/pour_19-12-09_22-16-07/trials_n\=10000.json -r 1 -n 0 2>&1 | tee log.txt
# python3 -m learn_tools.run_active data/scoop_20-02-26_11-38-37/trials_n\=10000.json -r 1 -n 0 2>&1 | tee log.txt

if __name__ == '__main__':
    main()
