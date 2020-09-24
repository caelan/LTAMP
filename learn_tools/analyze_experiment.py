from __future__ import print_function, division

import argparse
import math
import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score

sys.path.extend([
    os.path.join(os.getcwd(), 'pddlstream'), # Important to use absolute path when doing chdir
    os.path.join(os.getcwd(), 'ss-pybullet'),
])

from collections import namedtuple, OrderedDict, defaultdict
from pybullet_tools.utils import is_darwin
from scipy.stats import rankdata

#has_gui = sys.stdin.isatty()
#has_gui = 'DISPLAY' in os.environ
has_gui = is_darwin()
#print('GUI:', has_gui)
#if not has_gui:
#    import matplotlib
#    matplotlib.use('Agg')

from learn_tools.active_nn import REGRESSOR, CLASSIFIER, NN
from learn_tools.active_learner import score_prediction
from learn_tools.learnable_skill import LearnableSkill, read_data, is_real, get_skill
from learn_tools.statistics import compare_distributions
from learn_tools.learner import plot_learning_curve, LATENT, SCORE, FEATURE, PARAMETER, threshold_score, THRESHOLD, \
    estimate_gaussian, SEPARATOR, SUCCESS, FAILURE, DYNAMICS, threshold_scores, SKILL
from pddlstream.utils import mkdir, find_unique
from pybullet_tools.utils import is_remote, safe_zip, SEPARATOR

Algorithm = namedtuple('Algorithm', ['name', 'kernel', 'hyperparameters', 'variance', 'transfer_weight', 'label'])
Algorithm.__new__.__defaults__ = (None,) * len(Algorithm._fields)
Experiment = namedtuple('Experiment', ['algorithm', 'num_train', 'confusion', 'results'])
#Experiment = namedtuple('Experiment', ['algorithm', 'num_train', 'results'])

Confusion = namedtuple('Confusion', ['rmse', 'accuracy']) # 'fn', 'fp', 'tn', 'tp'

##################################################

def get_label(algorithm):
    label = algorithm.label
    if algorithm.name.startswith(NN):
        label = NN.upper()
    #label = 'GP-{}'.format(algorithm.kernel)
    if label is not None:
        suffix = ''
        if algorithm.name.endswith(CLASSIFIER):
            suffix = 'c'
        elif algorithm.name.endswith(REGRESSOR):
            suffix = 'r'
        return label + suffix
    return '_'.join(str(v) if i == 0 else '{}={}'.format(f[0], v) for i, (f, v) in enumerate(
        zip(algorithm._fields, algorithm)) if v is not None).lower()

def num_truth_from_confusion(confusion, label=SUCCESS):
    return sum(n for (t, _), n in confusion.items() if t == label)

def num_pred_from_confusion(confusion, label=SUCCESS):
    return sum(n for (_, p), n in confusion.items() if p == label)

def precision_from_confusion(confusion, label=SUCCESS):
    num_pred = num_pred_from_confusion(confusion, label=label)
    if num_pred == 0:
        return 0
    return confusion.get((label, label), 0) / num_pred

def recall_from_confusion(confusion, label=SUCCESS):
    num_truth = num_truth_from_confusion(confusion, label=label)
    if num_truth == 0:
        return 0
    return confusion.get((label, label), 0) / num_truth

def f1_from_confusion(confusion, **kwargs):
    precision = precision_from_confusion(confusion, **kwargs)
    recall = recall_from_confusion(confusion, **kwargs)
    if precision + recall == 0:
        return 0
    return 2*(precision * recall) / (precision + recall)

def accuracy_from_confusion(confusion):
    return sum(n for (t, p), n in confusion.items() if t == p) / sum(confusion.values())

##################################################

def compute_metrics(Y, Y_pred, verbose=True):
    rmse = math.sqrt(mean_squared_error(Y, Y_pred)) # mean_absolute_error | median_absolute_error

    indices = [i for i, y in enumerate(Y) if y in [SUCCESS, FAILURE]]
    labels = threshold_scores(Y[indices])
    label_pred = threshold_scores(Y_pred[indices])
    categories = sorted(set(np.concatenate([labels, label_pred])))
    confusion_counts = confusion_matrix(labels, label_pred, labels=categories)
    confusion_dict = {}
    for r, truth in enumerate(categories):
        for c, pred in enumerate(categories):
            confusion_dict[truth, pred] = confusion_counts[r, c]

    accuracy = accuracy_from_confusion(confusion_dict)
    f1 = f1_from_confusion(confusion_dict)
    prec = precision_from_confusion(confusion_dict)
    recall = recall_from_confusion(confusion_dict)
    #fp = confusion_dict[-1, +1]/len(labels)
    #fn = confusion_dict[+1, -1]/len(labels)
    if verbose:
        print('Num={}, Valid={}, RMSE={:.3f}, Accuracy={:.3f}, Prec={:.3f}, Recall={:.3f}, F1={:.3f}'.format(
            len(Y), len(indices), rmse, accuracy, prec, recall, f1))
    #confusion_freqs = confusion_counts / len(labels)
    #for r, truth in enumerate(categories):
    #    # TODO: metrics don't print correctly in python2
    #    print('Truth={:2}: {}'.format(truth, confusion_freqs[r,:].round(3).tolist()))
    return Confusion(rmse, confusion_dict)

def optimize_recall(Y, predictions, min_precision=0.0, alphas=[0.0]):
    # TODO: optimize via binary search
    # https://docs.python.org/2/library/bisect.html
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.bisect.html#scipy.optimize.bisect
    assert alphas
    labels = threshold_scores(Y)
    best_alpha, best_recall = 1.0, 0.0
    for alpha in alphas:
        Y_pred = np.array([score_prediction(prediction, alpha=alpha) for prediction in predictions])
        label_pred = threshold_scores(Y_pred)
        #print(Counter(label_pred))
        num_positive = sum(label == SUCCESS for label in label_pred) # Weakly decreasing
        # Precision might not be decreasing though
        if num_positive:
            precision = precision_score(labels, label_pred, pos_label=SUCCESS) # pos-precision = tp / (tp + fp)
        else:
            precision = 1.0
        recall = recall_score(labels, label_pred, pos_label=SUCCESS) # pos-recall = tp / (tp + fn) = tp / (pos-support)
        print('Alpha: {:.3f} | Positive: {:.3f} | Precision: {:.3f} | Recall: {:.3f}'.format(
              alpha, (num_positive / len(Y)), precision, recall))
        if precision < min_precision:
            continue
        if best_recall < recall:
            best_alpha, best_recall = alpha, recall
    return best_alpha, best_recall

##################################################

def plot_learning_curves(name, figure_dir, learning_curves, y_label,
                         y_limits=(0, +1), **kwargs):
    import matplotlib.pyplot as plt
    plt.figure()
    for label, sizes, scores in learning_curves:
        plot_learning_curve(sizes, scores, name=label, **kwargs)

    if name is not None:
        plt.title(name) # Could extract from figure_dir
    if y_limits is not None:
        plt.ylim(*y_limits)
    plt.xlabel('# Training Examples')
    plt.ylabel(y_label)
    plt.grid()
    plt.legend(loc='best')
    plt.tight_layout()

    if figure_dir is not None:
        mkdir(figure_dir)
        figure_path = os.path.join(figure_dir, '{}.png'.format(y_label)) # pdf
        plt.savefig(figure_path, transparent=False, bbox_inches='tight') # dpi=1000,
        print('Saved', figure_path)
    if not is_remote():
        plt.show()
    return plt


def plot_experiments(domain, experiments_name, experiments_dir, experiments,
                     evaluation_metric='success', prediction_metric='f1', # accuracy | recall | precision | f1
                     include_latent=False, include_none=False, scale=0.25, verbose=True, dump=False): # TODO: kwargs
    # TODO: deprecate experiments_name
    all_train_sizes = sorted({exp.num_train for exp in experiments
                              if exp.num_train is not None})
    #limits = (0, 1)
    #limits = (0.8, 1)
    limits = None
    # TODO: record experiment information somewhere

    dataset = 'test'  # train | test
    dataset_metric = '{} {}'.format(dataset, prediction_metric)

    results_from_algorithm = OrderedDict()
    confusion_from_algorithm = defaultdict(list)
    for experiment in experiments:
        #if experiment.algorithm.label != 'GP':
        #    continue
        #if (experiment.algorithm.label) == 'GP' and (experiment.algorithm.kernel != 'MLP'):
        #    continue
        if experiment.num_train is None:
            print(experiment.algorithm)
        train_sizes = all_train_sizes if experiment.num_train is None else [experiment.num_train]
        for train_size in train_sizes:
            results_from_algorithm.setdefault(experiment.algorithm, {})\
                .setdefault(train_size, []).extend(experiment.results)
            if experiment.confusion is not None:
                confusion_from_algorithm[experiment.algorithm, train_size].append(experiment.confusion)

    # TODO: skip empty results
    accuracy_curves = []
    success_curves = []
    for algorithm in results_from_algorithm:
        label = get_label(algorithm)
        sizes = sorted(results_from_algorithm[algorithm].keys())
        num_simulations = [len(results_from_algorithm[algorithm][size]) for size in sizes]
        num_predictions = [len(confusion_from_algorithm[algorithm, size]) for size in sizes]
        if verbose:
            print(SEPARATOR)
            print('Algorithm:', label)
            print('Sizes:', sizes)
            print('Simulations:', num_simulations)
            print('Predictions:', num_predictions)
        simulation_label = '{} ({})'.format(label, min(num_simulations))
        prediction_label = '{} ({})'.format(label, min(num_predictions))

        scores = []
        accuracies = []
        for size in sizes:
            confusions = confusion_from_algorithm[algorithm, size]
            results = results_from_algorithm[algorithm][size]
            for result in results:
                if include_latent and (result[SCORE] is not None):
                    result[LATENT] = {'{}:{}'.format(body, prop): value
                                      for body, parameters in result[SCORE].get(DYNAMICS, {}).items()
                                      for prop, value in parameters.items() if isinstance(value, float)} # TODO: np arrays

            # TODO: extend past just scoop
            #results = [r for r in results if (r[FEATURE]['spoon_type'] in SCOOP_SPOONS) and
            #           (r[FEATURE]['bowl_type'] in SCOOP_BOWLS)]
            if confusions:
                # TODO: analyze distribution of scores
                if isinstance(confusions[0], dict): # TODO: assert all
                    confusions = [c[dataset] for c in confusions]

            if confusions and not all(c is None for c in confusions):
                size_rmses = [confusion.rmse for confusion in confusions]
                size_accuracies = [confusion.accuracy for confusion in confusions]
                #print(size_accuracies[0])
                if isinstance(size_accuracies[0], dict):
                    if prediction_metric == 'rmse':
                        size_accuracies = size_rmses
                    elif prediction_metric == 'accuracy':
                        size_accuracies = list(map(accuracy_from_confusion, size_accuracies))
                    elif prediction_metric == 'f1':
                        size_accuracies = list(map(f1_from_confusion, size_accuracies))
                    elif prediction_metric == 'precision':
                        size_accuracies = list(map(precision_from_confusion, size_accuracies))
                    elif prediction_metric == 'recall':
                        size_accuracies = list(map(recall_from_confusion, size_accuracies))
                    else:
                        raise NotImplementedError(prediction_metric)
                else:
                    assert prediction_metric == 'accuracy'

                accuracies.append(size_accuracies)
                num_none = sum(result[SCORE] is None for result in results)
                if verbose:
                    print('Size: {:4} | Num: {} | None: {} | RMSE: {:.3f} | {}: {:.3f}'.format(
                        size, len(results), num_none, np.mean(size_rmses),
                        prediction_metric, np.mean(size_accuracies)))

            if dump and (size == sizes[-1]):
                #decompose_discrete(results)
                #analyze_data(None, results)
                #print(SEPARATOR)
                # TODO: could compare other discrete attributes
                compare_distributions(results)

            #predictions = defaultdict()
            size_scores = []
            for result in results:
                if not include_none and (result[SCORE] is None):
                    continue
                success = domain.score_fn(result[FEATURE], result[PARAMETER], result[SCORE])
                #if limits is not None:
                success = threshold_score(success, THRESHOLD, below=0, above=1)
                size_scores.append(success)
                #for name, value in result.get('prediction', {}).items():
                #    predictions.setdefault((success, name), []).append(value)
            scores.append(size_scores)

            # Statistics about how confident the model was
            #if not predictions:
            #    continue
            #print('Size:', size)
            #for success, name in sorted(predictions):
            #    # How many predictions are above zero?
            #    values = predictions[success, name]
            #    print('Success: {} | Name: {} | Num: {} | Mean: {:.3f} | Deviation {:.3f}'.format(
            #        success, name, len(values), np.mean(values), np.std(values)))

        if verbose:
            print('{} size: {}'.format(evaluation_metric, np.array(scores).shape))
            score_mean, score_std = estimate_gaussian(scores)
            print('{} means: {}'.format(evaluation_metric, score_mean.round(3).tolist()))
            print('{} deviations: {}'.format(evaluation_metric, score_std.round(3).tolist()))

            print('{} size: {}'.format(dataset_metric, np.array(scores).shape))
            accuracy_mean, accuracy_std = estimate_gaussian(accuracies)
            print('{} means: {}'.format(dataset_metric, accuracy_mean.round(3).tolist()))
            print('{} deviations: {}'.format(dataset_metric, accuracy_std.round(3).tolist()))

        if any(len(s) != 0 for s in scores):
            success_curves.append((simulation_label, sizes, scores))
        if accuracies:
            accuracy_curves.append((prediction_label, sizes, accuracies)) # tp/fp

    real_world = is_real(domain.name)
    skill = get_skill(domain.name)
    if verbose:
        print(SEPARATOR)
        print('Dataset:', domain.name)
        #print('Experiment:', experiments_name)
        print('Real:', real_world)
        print('Skill:', skill)
        print('Include None:', include_none)
        print('Sizes:', all_train_sizes)

    #combined_name = '{}\n{}'.format(domain.name, experiments_name)
    #combined_name = experiments_name
    #combined_name = None
    combined_name = '{} {}'.format('real world' if real_world else 'kitchen 3d', skill).title()
    if success_curves:
        plot_learning_curves(combined_name, experiments_dir, success_curves,
                             y_label=evaluation_metric.title(), y_limits=limits, scale=scale)
    if accuracy_curves:
        plot_learning_curves(combined_name, experiments_dir, accuracy_curves,
                             y_label=dataset_metric.title(), y_limits=None, scale=scale)


def analyze_experiment(experiment_path, **kwargs):
    experiment_dir = os.path.abspath(os.path.join(experiment_path, os.pardir))
    experiment_name = os.path.basename(experiment_dir)
    domain_name = os.path.basename(os.path.abspath(os.path.join(experiment_dir, os.pardir)))
    skill = domain_name.split('_')[0]
    print(domain_name, skill)

    domain = LearnableSkill(name=domain_name, skill=skill, parameter_ranges={None: (-np.inf, np.inf)})
    experiments = read_data(experiment_path)
    return plot_experiments(domain, experiment_name, experiment_dir, experiments, **kwargs)

##################################################

def test_parameter_variance(domain, rounds=10, digits=3):
    # TODO: we don't choose the ordering of domains to try
    # Each algorithm operates on a different domain
    # Active learning (via LSE) might only help for capturing the full super-level set

    algorithm = Algorithm(BATCH_GP, UNIFORM)
    #train_sizes = list(range(10, 500+10, 100))
    train_sizes = [100, 500, 1000]
    data = [[] for _ in range(len(train_sizes))]
    for t, train_size in enumerate(train_sizes):
        for r in range(rounds):
            print('Round: {} | Size: {}'.format(r, train_size))
            learner = train_learner(domain, algorithm, num_train=train_size)
            parameters = learner.get_parameters()
            print(parameters)
            for param, value in parameters.items():
                if len(value) != 1:
                    parameters[param] = list(rankdata(value))
            print(parameters)
            data[t].append(parameters)

    for train_size, data_at_size in safe_zip(train_sizes, data):
        data_per_parameter = {}
        for d in data_at_size:
            for param, value in d.items():
                data_per_parameter.setdefault(param, []).append(value)
        print()
        print('Size:', train_size)
        print('Examples:', len(data_at_size))
        for param in sorted(data_per_parameter):
            print('{}) | mu: {} | std: {}'.format(
                param, np.mean(data_per_parameter[param], axis=0).round(digits).tolist(),
                np.std(data_per_parameter[param], axis=0).round(digits).tolist()))
    # TODO: plot variance with respect to training and test statistics
    # TODO: graph these statistics over time

##################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment',
                        help='Path to the experiment that should be used')
    parser.add_argument('-d', '--dump', action='store_true',
                        help='When enabled, dumps trial statistics.')
    parser.add_argument('-n', '--none', action='store_true',
                        help='When enabled, includes trials that could not be executed.')
    args = parser.parse_args()
    analyze_experiment(args.experiment, dump=args.dump, include_none=args.none)

if __name__ == '__main__':
    main()

# python3 -m learn_tools.analyze_experiment data/pour_19-06-13_00-59-21/19-10-30_17-25-58_r=5_t=[25,100]_n=45/experiments.pk3
