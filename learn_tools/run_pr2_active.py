from __future__ import print_function

import argparse
import datetime
import math
import os
import sys
import time

import numpy as np

sys.path.extend([
    os.path.join(os.getcwd(), 'pddlstream'), # Important to use absolute path when doing chdir
    os.path.join(os.getcwd(), 'ss-pybullet'),
])

from collections import Counter
from itertools import product

from pybullet_tools.utils import is_darwin, user_input, elapsed_time, write_pickle, read_pickle
from pddlstream.utils import get_python_version, mkdir, implies, str_from_object

from learn_tools.analyze_experiment import Confusion, get_label, plot_experiments, Algorithm, \
    Experiment
from learn_tools.learnable_skill import load_data
from learn_tools.learner import balanced_split, score_accuracy, DATA_DIRECTORY, SEPARATOR, threshold_scores, inclusive_range
from learn_tools.collect_simulation import run_trials, get_trials, get_num_cores, HOURS_TO_SECS
from learn_tools.common import get_max_cores, DATE_FORMAT
from learn_tools.learner import FEATURE, PARAMETER, SCORE, DESIGNED, RANDOM # LEARNED
from learn_tools.active_learner import BEST
from learn_tools.active_nn import ActiveNN, NN_MODELS, RF_MODELS, RF_CLASSIFIER, RF_REGRESSOR, NN_CLASSIFIER, NN_REGRESSOR
from learn_tools.active_gp import ActiveGP, HYPERPARAM_FROM_KERNEL, STRADDLE_GP, GP_MODELS, BATCH_GP
from learn_tools.active_rf import ActiveRF, RF_MODELS

from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

#np.show_config()
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
# TODO: multiprocessing predictions don't seem to work at all on OS X

BALANCED = 'balanced'
UNIFORM = 'uniform'
ALL = 'all'

SEC_PER_EXPERIMENT = 60.0 # 30.9035954114
# (2.695*60*60) / (10000/45)

##################################################

def active_learning_discrete(learner, num, X_select, Y_select):

    if not learner.trained:
        learner.retrain() #num_restarts=10, num_processes=1)
    if not num:
        return X_select, Y_select
    for t in range(num):
        print(SEPARATOR)
        print('{}/{} active samples'.format(t, num))
        # new example
        idx = learner.get_highest_lse_idx(X_select)
        learner.retrain(newx=X_select[idx], newy=Y_select[idx])
        #new_parameter = next(learner.parameter_generator(None, feature))
        #print(new_parameter)
        X_select = np.delete(X_select, idx, 0)
        Y_select = np.delete(Y_select, idx, 0)

    return X_select, Y_select

def compute_metrics(Y, Y_pred):
    rmse = math.sqrt(mean_squared_error(Y, Y_pred)) # mean_absolute_error | median_absolute_error
    labels = threshold_scores(Y)
    label_pred = threshold_scores(Y_pred)
    accuracy = accuracy_score(labels, label_pred)
    categories = sorted(set(np.concatenate([labels, label_pred])))
    #print('Size:', len(Y))
    #print('RMSE:', rmse)
    print('Accuracy:', accuracy)
    confusion_counts = confusion_matrix(labels, label_pred, labels=categories)
    confusion_freqs = confusion_counts / len(labels)
    for r, truth in enumerate(categories):
        # TODO: metrics don't print correctly in python2
        print('Truth={:2}: {}'.format(truth, confusion_freqs[r,:].round(3).tolist()))

    confusion_dict = {}
    for r, truth in enumerate(categories):
        for c, pred in enumerate(categories):
            confusion_dict[truth, pred] = confusion_counts[r, c]

    #[[tn  fp]
    # [fn  tp]]
    # Rows are ground truth, columns are predictions
    #tn, fp, fn, tp = matrix
    #print('TN: {:.3f} | FP: {:.3f} | FN: {:.3f} | TP: {:.3f}'.format(tn, fp, fn, tp))
    # Could adjust threshold and compute Precision/Recall as well as average precision

    # TODO: other classification metrics
    # accuracy = (tn + tp) / (tn + fp + fn + tp)
    # balanced accuracy = 0.5*(tp / (tp + fn) + tn / (tn + fp)) = 0.5*(pos-recall + neg-recall)
    # pos-support = tp + fn
    # pos-recall * pos-support = tp
    # neg-support = tn + fp
    # neg-recall * neg-support = tn
    # macro average (averaging the unweighted mean per label)
    # weighted average (averaging the support-weighted mean per label)
    # micro average (averaging the total true positives, false negatives and false positives)
    # it is only shown for multi-label or multi-class with a subset of classes because it is accuracy otherwise
    #print(confusion)
    #print('+1 Precision:', precision_score(labels, label_pred, pos_label=SUCCESS)) # pos-precision = tp / (tp + fp)
    #print('+1 Recall:', recall_score(labels, label_pred, pos_label=SUCCESS)) # pos-recall = tp / (tp + fn) = tp / (pos-support)
    #print('Balanced:', balanced_accuracy_score(labels, label_pred, # average of recall obtained on each class
    #                              adjusted=False)) # the result is adjusted for chance, so that random performance would score 0
    #print('F1:', f1_score(labels, label_pred, pos_label=SUCCESS)) # 2 * (precision * recall) / (precision + recall)
    #print('FBeta:', fbeta_score(labels, label_pred, beta=1.0, pos_label=SUCCESS)) # weighted harmonic mean of precision and recall
    # F-measures do not take the true negatives into account
    #print('mAP:', average_precision_score(labels, label_pred, average=None))
    #print('ROC AUC:', roc_auc_score(labels, label_pred, average=None)) # Equal to balanced?
    # precision_recall_curve, precision_recall_fscore_support
    #report = classification_report(labels, label_pred, labels=categories, #target_names=['failure', 'success'],
    #                               digits=5, output_dict=True)
    #print(report)

    #confusion = Confusion(rmse, accuracy)
    confusion = Confusion(rmse, confusion_dict)

    return confusion

def save_learner(data_dir, learner):
    if data_dir is None:
        return False
    #domain = learner.func
    #data_dir = os.path.join(MODEL_DIRECTORY, domain.name)
    #name = learner.name
    name = get_label(learner.algorithm)
    mkdir(data_dir)
    learner_path = os.path.join(data_dir, '{}.pk{}'.format(name, get_python_version()))
    print('Saved', learner_path)
    write_pickle(learner_path, learner)
    return True

def split_data_helper(X, Y, split_type, num_train, shuffle=True):
    if len(Y) <= num_train:
        return X, Y, np.array([]), np.array([])
    #Y = Y[:, None]
    if split_type == BALANCED:
        X_train, X_test, Y_train, Y_test = balanced_split(
            X, Y, train_size=num_train, shuffle=shuffle)
    elif split_type == UNIFORM:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, train_size=min(len(X), num_train), shuffle=shuffle) # stratify=Y,
        #sss = StratifiedShuffleSplit(n_splits=1, train_size=num_train, random_state=0)
        #train_index, test_index = next(sss.split(X, Y))
        #X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
    elif split_type == ALL:
        X_train, X_test, Y_train, Y_test = X, None, Y, None
    else:
        raise ValueError(split_type)
    return X_train, Y_train, X_test, Y_test

def split_data(X, Y, split_type, num_train, min_classes=2, **kwargs):
    while True:
        X_train, Y_train, X_test, Y_test = split_data_helper(X, Y, split_type, num_train, **kwargs)
        Y_labels = threshold_scores(Y_train)
        if min_classes <= len(Counter(Y_labels.tolist())):
            return X_train, Y_train, X_test, Y_test

def create_learner(domain, X, Y, split, algorithm, num_train, **kwargs):
    X_train, Y_train = X[:num_train], Y[:num_train]
    #X_train, Y_train, _, _ = split_data(X, Y, split, num_train)
    print('Training examples:', len(Y_train))
    print('Average score:', np.average(Y_train))
    labels = threshold_scores(Y_train)
    frequencies = Counter(labels)
    for c in sorted(frequencies):
        print('Label={}: {:.3f}'.format(c, float(frequencies[c]) / len(labels)))

    start_time = time.time()
    if algorithm.name in GP_MODELS:
        hyperparameters = HYPERPARAM_FROM_KERNEL[domain.skill, algorithm.kernel] if algorithm.hyperparameters else None
        learner = ActiveGP(domain, initx=X_train, inity=Y_train, kernel_name=algorithm.kernel,
                                   hyperparameters=hyperparameters, use_var=algorithm.use_var, **kwargs)
    elif algorithm.name in RF_MODELS:
        learner = ActiveRF(domain, initx=X_train, inity=Y_train,
                           model_type=algorithm.name, use_var=algorithm.use_var, **kwargs)
    elif algorithm.name in NN_MODELS:
        learner = ActiveNN(domain, initx=X_train, inity=Y_train,
                           model_type=algorithm.name, **kwargs)
    else:
        raise ValueError(algorithm.name)
    #learner.name = '{}_{}_n{}_a{}'.format(learner.name, algorithm.split, num_train, num_active)
    learner.algorithm = algorithm
    # run_ActiveLearner(learner)
    # train learner with training data
    learner.retrain()  # num_restarts=10, num_processes=1)
    #save_learner(learner)
    print('Train time:', time.time() - start_time)
    print('Train examples:', Y_train.shape[0])
    confusion = compute_metrics(Y_train, learner.score_x(X_train))
    return learner, confusion

##################################################

def evaluate_learner(domain, seed, train_confusion,
                     X_test, Y_test, algorithm, learner,
                     train_size, num_trials, alphas,
                     serial=False, confusion_only=True):
    # learner select train_size number of trials?
    start_time = time.time()
    max_cores = get_max_cores(serial=serial)

    test_confusion = None
    if (train_size is not None) and (len(Y_test) != 0):
        # TODO: kernel density estimation
        print('Test examples:', Y_test.shape[0])
        if algorithm.name in NN_MODELS:
            alphas = alphas[:1]
        confusions = []
        for alpha in alphas:
            print('Alpha={}'.format(alpha) if alpha is None else 'Alpha={:.3f}'.format(alpha))
            confusions.append(compute_metrics(Y_test, learner.score_x(
                X_test, alpha=alpha, max_cores=max_cores)))
        test_confusion = confusions[0]

    results = []
    if not confusion_only:
        # trials selected by learner, num_trials is number of contexts that the learner evaluated upon
        trials = get_trials(problem=domain.skill, fn=learner, num_trials=num_trials,
                            seed=seed, verbose=serial)
        num_cores = get_num_cores(trials, serial=serial)
        results = run_trials(trials, num_cores=num_cores)
        scores = [domain.score_fn(result[FEATURE], result[PARAMETER], result[SCORE])
                  for result in results]
        # TODO: record best predictions as well as average prediction quality
        print('Seed: {} | {} | Results: {} | Time: {:.3f}'.format(
            seed, algorithm, len(results), elapsed_time(start_time)))
        if scores:
            print('Average: {:.3f} | Success: {:.1f}%'.format(
                np.average(scores), 100 * score_accuracy(scores)))
    confusion = {
        'train': train_confusion,
        'test': test_confusion,
    }
    return Experiment(algorithm, train_size, confusion, results)
    # tuple saved to experiments.pk3

def save_experiments(experiments_dir, experiments):
    if experiments_dir is None:
        return None
    data_path = os.path.join(experiments_dir, 'experiments.pk{}'.format(get_python_version()))
    write_pickle(data_path, experiments)
    print('Saved', data_path)
    return data_path

##################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*', help='Paths to the data.')
    #parser.add_argument('-a', '--active', type=int, default=0, # None
    #                    help='The number of active samples to collect')
    parser.add_argument('-l', '--learner', default=None,
                        help='Path to the learner that should be used')
    parser.add_argument('-n', '--num_trials', type=int, default=100,
                        help='The number of samples to collect')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Whether to save the learners')
    parser.add_argument('-r', '--num_rounds', type=int, default=1,
                        help='The number of rounds to collect')
    #parser.add_argument('-t', '--num_train', type=int, default=None,
    #                    help='The size of the training set')
    args = parser.parse_args()

    # TODO: be careful that paging isn't altering the data
    # TODO: don't penalize if the learner identifies that it can't make a good prediction
    # TODO: use a different set of randomized parameters for train and test

    include_none = False
    serial = is_darwin()


    #training_sizes = inclusive_range(50, 500, 25)
    #training_sizes = inclusive_range(25, 100, 5)
    #training_sizes = inclusive_range(25, 100, 5)
    training_sizes = inclusive_range(10, 50, 5)
    #training_sizes = inclusive_range(100, 1000, 100)
    #training_sizes = [20]
    #training_sizes = [1500]

    #kernels = ['RBF', 'Matern52', 'MLP']
    kernels = ['MLP']

    #hyperparameters = [None]
    #hyperparameters = [True]
    hyperparameters = [True, None] # None,

    query_type = BEST # BEST | CONFIDENT | REJECTION | ACTIVE # type of query used to evaluate the learner
    is_adaptive = False
    max_test = 50 #
    #alphas = np.linspace(0.0, 0.9, num=5, endpoint=True)
    alphas = [0.0, .8, .9, .99]
    #alphas = [None]  # Use the default (i.e. GP parameters)

    use_vars = [True]
    binary = False
    split = UNIFORM # BALANCED

    # Omitting failed labels is okay because they will never be executed
    algorithms = []
    #algorithms += [(Algorithm(BATCH_GP, kernel=kernel, hyperparameters=hype, use_var=use_var), [num_train])
    #               for num_train, kernel, hype, use_var in product(training_sizes, kernels, hyperparameters, use_vars)]
    algorithms += [(Algorithm(STRADDLE_GP, kernel, hype, use_var), training_sizes)
                   for kernel, hype, use_var in product(kernels, hyperparameters, use_vars)]
    #algorithms += [(Algorithm(rf_model, p_explore=None, use_var=use_var), [num_train])
    #               for rf_model, num_train, use_var in product(RF_MODELS, training_sizes, use_vars)]
    #algorithms += [(Algorithm(nn_model, p_explore=None), [num_train])
    #               for nn_model, num_train in product(NN_MODELS, training_sizes)]
    #algorithms += [(Algorithm(RANDOM), None), (Algorithm(DESIGNED), None)]
    print('Algorithms:', algorithms)
    print('Split:', split)

    trials_per_round = sum(1 if train_sizes is None else
                           (train_sizes[-1] - train_sizes[0] + len(train_sizes))
                           for _, train_sizes in algorithms)
    num_experiments = args.num_rounds*trials_per_round

    date_name = datetime.datetime.now().strftime(DATE_FORMAT)
    size_str = '[{},{}]'.format(training_sizes[0], training_sizes[-1])
    #size_str = '-'.join(map(str, training_sizes))
    experiments_name = '{}_r={}_t={}_n={}'.format(date_name, args.num_rounds, size_str, args.num_trials) #'19-08-09_21-44-58_r=5_t=[10,150]_n=1'#
    #experiments_name = 't={}'.format(args.num_rounds)
    # TODO: could include OS and username if desired

    domain = load_data(args.paths)
    print()
    print(domain)
    X, Y, W = domain.get_data(include_none=include_none)
    print('Total number of examples:', len(X))
    if binary:
        # NN can fit perfectly when binary
        # Binary seems to be outperforming w/o
        Y = threshold_scores(Y)

    max_train = len(X) - max_test #min(max([0] + [active_sizes[0] for _, active_sizes in algorithms
                     #          if active_sizes is not None]), len(X))

    #parameters = {
    #    'include None': include_none,
    #    'binary': binary,
    #    'split': split,
    #}

    print('Name:', experiments_name)
    print('Experiments:', num_experiments)
    print('Max train:', max_train)
    print('Include None:', include_none)
    print('Examples: n={}, d={}'.format(*X.shape))
    print('Binary:', binary)
    print('Estimated hours:', num_experiments * SEC_PER_EXPERIMENT / HOURS_TO_SECS)
    user_input('Begin?')
    # TODO: residual learning for sim to real transfer
    # TODO: can always be conservative and add sim negative examples

    # TODO: combine all data to write in one folder
    data_dir = os.path.join(DATA_DIRECTORY, domain.name) # EXPERIMENT_DIRECTORY
    experiments_dir = os.path.join(data_dir, experiments_name)
    mkdir(experiments_dir)
    start_time = time.time()
    experiments = []
    for round_idx in range(args.num_rounds):
        round_dir = os.path.join(data_dir, experiments_name, str(round_idx))
        mkdir(round_dir)
        seed = hash(time.time())
        train_test_file = os.path.join(round_dir, 'data.pk3')
        if not os.path.exists(train_test_file):
            X_train, Y_train, X_test, Y_test = split_data(X, Y, split, max_train)
            X_test, Y_test = X_test[:max_test], Y_test[:max_test]
            write_pickle(train_test_file, (X_train, Y_train, X_test, Y_test))
        else:
            X_train, Y_train, X_test, Y_test = read_pickle(train_test_file)

        print('Train examples:', X_train.shape)
        print('Test examples:', X_test.shape)
        # TODO: need to be super careful when running with multiple contexts

        for algorithm, active_sizes in algorithms:
            # active_sizes = [first #trainingdata selected from X_train, #active exploration + #trainingdata]
            print(SEPARATOR)
            print('Round: {} | {} | Seed: {} | Sizes: {}'.format(round_idx, algorithm, seed, active_sizes))
            # TODO: allow keyboard interrupt
            if active_sizes is None:
                learner = algorithm.name
                active_size = None
                train_confusion = None
                experiments.append(evaluate_learner(domain, seed, train_confusion, X_test, Y_test, algorithm, learner,
                                                    active_size, args.num_trials, alphas,
                                                    serial))
            else:
                # [10 20 25] take first 10 samples from X_train to train the model, 10 samples chosen actively
                # sequentially + evaluate model, 5 samples chosen actively sequentially + evaluate model
                # Could always keep around all the examples and retrain
                # TODO: segfaults when this runs in parallel
                # TODO: may be able to retrain in parallel if I set OPENBLAS_NUM_THREADS
                learner_prior_nx = 0
                '''
                if algorithm.hyperparameters:
                    if domain.skill == 'pour':
                        learner_file = '/Users/ziw/ltamp_pr2/data/pour_19-06-13_00-59-21/19-08-09_19-30-01_r=10_t=[50,400]_n=1/{}/gp_active_mlp_true_true.pk3'.format(
                            round_idx)
                    elif domain.skill == 'scoop':
                        learner_file = '/Users/ziw/ltamp_pr2/data/scoop_19-06-10_20-16-59_top-diameter/19-08-09_19-34-56_r=10_t=[50,400]_n=1/{}/gp_active_mlp_true_true.pk3'.format(
                            round_idx)
                    learner = read_pickle(learner_file)
                    learner_prior_nx = learner.nx
                    learner.retrain(newx=X_train[:active_sizes[0]], newy=Y_train[:active_sizes[0], None])
                else:
                '''
                learner, train_confusion = create_learner(domain, X_train, Y_train, split, algorithm,
                                                          num_train=active_sizes[0], query_type=query_type,
                                                          is_adaptive=is_adaptive)

                if algorithm.name == STRADDLE_GP:
                    X_select, Y_select = X_train[active_sizes[0]:], Y_train[active_sizes[0]:]

                for active_size in active_sizes:
                    num_active = active_size - learner.nx + learner_prior_nx# learner.nx is len(learner.xx)
                    print('\nRound: {} | {} | Seed: {} | Size: {} | Active: {}'.format(
                        round_idx, algorithm, seed, active_size, num_active))
                    if algorithm.name == STRADDLE_GP:
                        X_select, Y_select = active_learning_discrete(learner, num_active, X_select, Y_select)
                    #if args.save:
                    save_learner(round_dir, learner)
                    experiments.append(evaluate_learner(domain, seed, None, X_test, Y_test,
                                                        algorithm, learner,
                                                        active_size, args.num_trials, alphas,
                                                        serial))
                    save_experiments(experiments_dir, experiments)

    print(SEPARATOR)
    if experiments:
        save_experiments(experiments_dir, experiments)
        plot_experiments(domain, experiments_name, experiments_dir, experiments,
                         include_none=False)
                         #include_none=include_none)
        print('Experiments:', experiments_dir)
    print('Total experiments:', len(experiments))
    print('Total hours:', elapsed_time(start_time) / HOURS_TO_SECS)

if __name__ == '__main__':
    main()
