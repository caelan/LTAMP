from __future__ import print_function

import os
import random
import numpy as np
import math

from itertools import count
from collections import namedtuple

from pybullet_tools.utils import read_pickle, INF

Collector = namedtuple('Collector', ['collect_fn', 'gen_fn', 'parameter_fns',
                                     'validity_test', 'features', 'score_fn'])

THRESHOLD = 0
FAILURE = -1
SUCCESS = +1
DEFAULT_INTERVAL = (FAILURE, SUCCESS)

# Set None examples as value 0 between -1 and +1
#PLANNING_FAILURE = THRESHOLD
PLANNING_FAILURE = -1e-1 # Larger (even positive) performs better
#PLANNING_FAILURE = -0.5
#PLANNING_FAILURE = FAILURE

NORMAL_WEIGHT = 0.0
FAILURE_WEIGHT = math.pow(0.5, 2) # variance (instead of std)
TRANSFER_WEIGHT = 0.5
# TODO: hard failures vs soft planning failures

file_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIRECTORY = os.path.abspath(os.path.join(file_dir, os.pardir, 'data/'))
LEARNER_DIRECTORY = os.path.abspath(os.path.join(file_dir, os.pardir, 'learners/'))
#EXPERIMENT_DIRECTORY = os.path.join(DATA_DIRECTORY, 'experiments/')

DESIGNED = 'designed'
CONSTANT = 'constant'
RANDOM = 'random'
TRAINING = 'training'
LEARNED = 'learned'
POLICIES = (DESIGNED, CONSTANT, RANDOM, TRAINING, LEARNED)

SKILL = 'skill'
LATENT = 'latent'
FEATURE = 'feature'
PARAMETER = 'parameter'
SCORE = 'score'
DYNAMICS = 'dynamics'
RANGES = 'ranges'
CATEGORIES = (DYNAMICS, LATENT, FEATURE, PARAMETER, SCORE)

REAL_PREFIX = 'pr2'
BAD = 'bad'

SEPARATOR = '\n' + 50*'-' + '\n'

def rescale(value, interval, new_interval=DEFAULT_INTERVAL):
    lower, upper = interval
    f = float(value - lower) / (upper - lower)
    new_lower, new_upper = new_interval
    return new_lower + f*(new_upper - new_lower)

def threshold_score(score, threshold=THRESHOLD, below=FAILURE, above=SUCCESS):
    if score is PLANNING_FAILURE: # numpy.int64 vs int
        return threshold
    #if score < threshold: # Regressors will sometimes predict 0
    if score <= threshold:
        return below
    if threshold < score:
        return above
    return threshold

def threshold_scores(Y, **kwargs):
    return np.array([threshold_score(y, **kwargs) for y in Y])
    #return 2 * binarize(Y, threshold=THRESHOLD) - np.ones(Y.shape)

def inclusive_range(start, stop, step=1):
    return list(range(start, stop + step, step))

def score_accuracy(Y, threshold=THRESHOLD):
    Y = np.array(Y)
    if len(Y) == 0:
        return 0.0
    return float(len(Y[threshold < Y])) / len(Y)

def get_trial_parameter_fn(parameter):
    def trial_parameter(world=None, feature=None):
        if parameter is False:
            return
        yield parameter
    return trial_parameter

def sample_parameter(parameter_ranges):
    parameter = {name: random.uniform(*interval) for name, interval in sorted(parameter_ranges.items())}
    parameter[RANGES] = parameter_ranges
    return parameter

def get_explore_parameter_fn(parameter_ranges, max_parameters=INF):
    def parameter_fn(world, feature):
        num_parameters = count()
        while next(num_parameters) < max_parameters:
            yield sample_parameter(parameter_ranges)
    return parameter_fn

def x_from_context_sample(context, sample):
    # Zi uses the convention where the parameter comes before the feature
    return np.concatenate([sample, context])

##################################################

def balanced_split(X, Y, threshold=0, train_size=INF, shuffle=True):
    negative_indices = np.argwhere(Y.flatten() <= threshold).flatten()
    positive_indices = np.argwhere(threshold < Y.flatten()).flatten()
    num_train = int(min(len(positive_indices), len(negative_indices), train_size / 2))
    print('Positive: {} | Negative: {} | Total: {}'.format(len(positive_indices), len(negative_indices), len(Y)))
    if shuffle:
        np.random.shuffle(negative_indices)
        np.random.shuffle(positive_indices)

    train_indices = np.concatenate([negative_indices[:num_train], positive_indices[:num_train]])
    test_indices = np.concatenate([negative_indices[num_train:], positive_indices[num_train:]])
    if shuffle:
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
    return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]

##################################################

class Sampler(object):
    def predict(self, feature):
        raise NotImplementedError()

class SKLearnSampler(Sampler):
    def __init__(self, model, features, parameters, parameter_ranges, num_samples=1000):
        # TODO: include information about the data files used
        self.model = model
        self.features = tuple(features)
        self.parameters = tuple(parameters)
        self.inputs = self.features + self.parameters
        self.parameter_ranges = parameter_ranges
        assert set(parameters) == set(parameter_ranges.keys())
        self.num_samples = num_samples
        # score threshold?
    def combine_inputs(self, feature, parameter):
        return x_from_context_sample([feature[name] for name in self.features],
                                     [parameter[name] for name in self.parameters])
    def sample_parameter(self):
        return {name: np.random.uniform(*interval)
                for name, interval in self.parameter_ranges.items()}
    def predict(self, feature):
        parameters = [self.sample_parameter() for _ in range(self.num_samples)]
        inputs = [self.combine_inputs(feature, parameter) for parameter in parameters]
        #scores = self.model.predict(inputs)
        scores = [d[1] for d in self.model.predict_proba(inputs)]
        #print(sorted(scores))
        best_score = np.amax(scores)
        best_indices = np.argwhere(scores == best_score).flatten()
        print('Parameters: {} | Best score: {:.3f} | Num indices: {}'.format(
            len(parameters), best_score, len(best_indices)))
        best_indices = sorted(best_indices, key=lambda i: scores[i], reverse=True)
        #random.shuffle(best_indices)
        #index = random.choice(best_indices)
        #return parameters[index]
        for index in best_indices:
            yield parameters[index]
    @staticmethod
    def load(rel_path):
        return read_pickle(os.path.join(LEARNER_DIRECTORY, rel_path))


def estimate_gaussian(test_scores): # When scores is not of uniform size
    mean = np.array([np.mean(scores) if scores else np.nan
                     for scores in test_scores])
    std = np.array([np.std(scores) if scores else np.nan
                    for scores in test_scores])
    return mean, std


def plot_learning_curve(train_sizes, test_scores, scale=1.0, name=None):
    import matplotlib.pyplot as plt
    test_scores_mean, test_scores_std = estimate_gaussian(test_scores)
    width = scale * test_scores_std # standard deviation
    # TODO: standard error (confidence interval)
    # from learn_tools.active_learner import tail_confidence
    # alpha = 0.95
    # scale = tail_confidence(alpha)
    # width = scale * test_scores_std / np.sqrt(train_sizes)
    plt.fill_between(train_sizes, test_scores_mean - width, test_scores_mean + width, alpha=0.1)
    plt.plot(train_sizes, test_scores_mean, 'o-', label=name)
    return plt
