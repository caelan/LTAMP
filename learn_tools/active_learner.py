from __future__ import print_function, division

import numpy as np
import scipy
import time
import os
import datetime

from learn_tools import helper
from operator import itemgetter

from learn_tools.learner import FAILURE, THRESHOLD, SUCCESS, score_accuracy
#from pybullet_tools.utils import interval_generator
from pddlstream.utils import mkdir, get_python_version, write_pickle, elapsed_time

try:
    import cPickle as pickle
except ImportError:
    import pickle

BEST = 'best_prob'
CONFIDENT = 'sample'
REJECTION = 'rejection'
TEST_TYPES = (BEST, CONFIDENT, REJECTION)

# TODO: unify with active_gp
STRADDLE = 'straddle'
VARIANCE = 'variance'
LSE_TYPES = (STRADDLE, VARIANCE)
QUERY_TYPES = TEST_TYPES + LSE_TYPES

##################################################

def x_from_context_sample(context, sample):
    return np.hstack([sample, context])  # TODO: use generic concatenation

def format_x(sample, context):
    parameters = sample[None, :] if sample.ndim == 1 else sample
    return np.hstack((parameters, np.tile(context, (parameters.shape[0], 1))))

def tail_confidence(alpha):
    lower, upper = scipy.stats.norm.interval(alpha=alpha)
    return upper

def score_prediction(prediction, alpha=0.0):
    if not prediction.get('robust', True):
        return FAILURE
    if alpha == 1.0:
        return FAILURE
    mu = prediction.get('mean')
    std = np.sqrt(prediction.get('variance', 0.0))
    scale = tail_confidence(alpha)
    return mu - scale * std
    # mu - scale * std > 0
    # mu / std > scale
    # Equivalent to selecting threshold for mu / std

##################################################

class ActiveLearner(object):
    # TODO ZI: why do we need activelearner, uncertainlearner, active_gp again? to support RF?
    # but it seems like many of the sampling/query functions can't be shared with RF?
    # TODO ZI: move _gen_adaptive
    DEFAULT_WEIGHT = 1.
    def __init__(self, func, initx=np.array([[]]), inity=np.array([]), initw=None,
                 query_type=BEST, transfer_weight=None):
        self.func = func
        self.query_type = query_type
        #self.max_samples = np.inf
        self.trained = False
        self.results = [] # TODO: use this instead
        self.num_batch = 0
        self.xx = np.empty([0, self.dx])
        self.yy = np.empty(0)
        self.weights = np.empty(0)
        self.add_data(initx, inity, initw)
        self.transfer_weight = transfer_weight
        self.algorithm = None
        self.date = datetime.datetime.now()
        self.validity_learner = None
        self.sim_model = None # Transfer model
        #self.sample_generator = interval_generator(*self.func.x_range, use_halton=False)

    @property
    def nx(self):
        return len(self.xx)

    @property
    def dx(self):
        return self.func.dx

    @property
    def dp(self):
        return len(self.func.param_idx)

    def save(self, data_dir):
        from learn_tools.analyze_experiment import get_label
        if data_dir is None:
            return False
        # domain = learner.func
        # data_dir = os.path.join(MODEL_DIRECTORY, domain.name)
        # name = learner.name
        name = get_label(self.algorithm)
        mkdir(data_dir)
        learner_path = os.path.join(data_dir, '{}.pk{}'.format(name, get_python_version()))
        print('Saved learner:', learner_path)
        write_pickle(learner_path, self)
        return True

    def add_data(self, xx, yy, weights=None):
        if (xx is None) and (yy is None) and (weights is None):
            return
        xx, yy, = np.array(xx), np.array(yy)
        assert (xx.ndim == 2) and (yy.ndim == 1)
        if weights is None:
            weights = self.DEFAULT_WEIGHT*np.ones(yy.shape)
        self.xx = np.vstack((self.xx, xx))
        self.yy = np.hstack((self.yy, yy))
        self.weights = np.hstack((self.weights, weights))
        self.trained = False

    #########################

    #def prediction(self, feature, parameter):
    #    return None

    def query_best_prob(self, context):
        raise NotImplementedError()

    def query_lse(self, context):
        raise NotImplementedError()

    def sample(self, context, **kwargs):
        raise NotImplementedError()

    def query(self, context):
        """
        Select the next input to query.
        """
        #self.reset_sample()
        if self.query_type in LSE_TYPES:
            return self.query_lse(context)
        if self.query_type == BEST:
            return self.query_best_prob(context)
        if self.query_type == CONFIDENT:
            return self.sample(context)
        if self.query_type == REJECTION:
            raise NotImplementedError()
        raise ValueError(self.query_type)

    def retrain(self, newx=None, newy=None, new_w=None):
        raise NotImplementedError()

    def reset_sample(self):
        pass

    def score_x(self, X, **kwargs):
        raise NotImplementedError()

    def score(self, feature, parameter, **kwargs):
        x = self.func.x_from_feature_parameter(feature, parameter)
        y = self.score_x(x[None, :], **kwargs)
        return float(y[0,0])

    def get_score_f(self, context, **kwargs):
        raise NotImplementedError()

    #########################

    # def dataset_generator(self, world, feature):
    #     print(feature)
    #     context = self.func.context_from_feature(feature)
    #     print(context.tolist())
    #     assert len(self.xx) == len(self.yy)
    #     for x, y in zip(self.xx, self.yy):
    #         if THRESHOLD < y[0]:
    #             print(x)
    #             sample = x[self.func.param_idx]
    #             parameter = self.func.parameter_from_sample(sample)
    #             mean, var = self.score(feature, parameter)
    #             print(y[0], mean, var)
    #             yield parameter

    def sample_parameter(self):
        # TODO: option to randomly select the feature as well?
        # This includes an incorrect context
        # TypeError: can't pickle generator objects
        x = np.random.uniform(*self.func.x_range)
        #x = next(self.sample_generator)
        return self.func.parameter_from_sample(x[self.func.param_idx])

    def sampling_optimization(self, world, feature, min_score=-np.inf, num_candidates=1000, valid=True, verbose=True):
        '''

        :param world: planning world
        :param feature: context of the skill
        :param min_score: lower bound on the score
        :param num_candidates: number of uniform random candidates
        :param kwargs:
        :return: the valid skill parameter that has the highest score among all the uniform random samples
        The score is evaluated by score_x
        '''
        # TODO: Why use score_x to sort the parameters?
        # TODO: take in an arbitrary score function
        start_time = time.time()
        parameters = [self.sample_parameter() for _ in range(num_candidates)]
        #scores = [self.score(feature, p, **kwargs) for p in parameters]

        context = self.func.context_from_feature(feature)
        X = np.array([self.func.sample_from_parameter(p) for p in parameters])
        if self.validity_learner is not None:
            prob_fn = self.validity_learner.get_score_f(context, negate=False)
            validities = prob_fn(X)[:, 0] # TODO: confidence interval
            print('{:.2f}% of {} predicted to be valid'.format(100*score_accuracy(validities), len(validities)))
        else:
            validities = SUCCESS*np.ones(len(X))
        score_fn = self.get_score_f(context, negate=False)
        scores = score_fn(X)[:, 0]

        #X = np.array([self.func.x_from_feature_parameter(feature, p) for p in parameters])
        #scores = self.score_x(X)[:, 0]

        scored_parameters = list(zip(scores, validities, parameters))
        if verbose:
            print('Scored {} parameters in {:.3f} sec'.format(len(scored_parameters), elapsed_time(start_time)))
        scored_parameters = [(s, v, p) for s, v, p in scored_parameters if min_score <= s]
        ordered_parameters = sorted(scored_parameters, key=itemgetter(0), reverse=True)
        # TODO: could also filter before scoring
        for i, (score, validity, parameter) in enumerate(ordered_parameters):
            if not valid or ((THRESHOLD < validity) and self.func.collector.validity_test(world, feature, parameter)):
                if verbose:
                    print('Valid ({:.3f}) parameter with score {:.3f} after {}/{} iterations and {:.3f} sec'.format(
                        validity, score, i+1, len(ordered_parameters), elapsed_time(start_time)))
                return parameter
        if verbose:
            print('No valid after {} iterations and {:.3f} sec'.format(
                len(ordered_parameters), elapsed_time(start_time)))
        return None

    def sampling_generator(self, world, feature, max_attempts=1, **kwargs):
        while True:
            for _ in range(max_attempts):
                parameter = self.sampling_optimization(world, feature, **kwargs)
                if parameter is not None:
                    yield parameter
                    break
            else:
                #return
                yield None

    # def query_generator(self, world, feature, max_attempts=1):
    #     self.reset_sample()
    #     context = self.func.context_from_feature(feature)
    #     while True:
    #         # TODO: even when querying best it seems to return random things
    #         for _ in range(max_attempts):
    #             x = self.query(context)
    #             if x is not None:
    #                 parameter = self.func.parameter_from_sample(x[self.func.param_idx])
    #                 yield parameter
    #                 break
    #         else:
    #             return

    def parameter_generator(self, world, feature, **kwargs):
        #self.reset_sample()
        #self.dataset_generator(world, feature)
        if self.query_type in (CONFIDENT, REJECTION):
            return self.sampling_generator(world, feature, max_attempts=1000, num_candidates=1, min_score=THRESHOLD, **kwargs)
        if self.query_type in (BEST,) + LSE_TYPES:
            return self.sampling_generator(world, feature, max_attempts=1, num_candidates=1000, **kwargs)
            #return self.query_generator(world, feature, **kwargs)
        raise ValueError(self.query_type)
        # TODO ZI: the query generator can use the random sampling generator to select the sample achieving the best acquisition function value?
        # TODO ZI: we can sample adaptively around the best sample every iteration to refine? Maybe use CEM/evolutionary alg to do the optimization with the validity filter

    #########################

    def _update_samples(self, context, new_x_samples, new_prob):
        raise NotImplementedError()

    def _gen_uniform(self, context, n_proposals):
        xmin, xmax = self.func.x_range[:, self.func.param_idx]
        prob_unif_unit = np.prod(xmax - xmin)
        s = time.time()
        x_samples_unif = np.random.uniform(xmin, xmax, (n_proposals, self.dp))
        prob_unif = np.ones(n_proposals) * prob_unif_unit # TODO: divide by prob_unif_unit instead?
        print('gen uniform sample time ', time.time() - s)
        self._update_samples(context, x_samples_unif, prob_unif) # Could include importance weight
        #print('unif good={}'.format(len(x_samples_unif)))

    def _gen_adaptive(self, context, n_proposals, x_samples, scale):
        if len(x_samples) == 0:
            return
        # _gen_adaptive_samples in undertainty learner uses this function..
        # TODO(ziw): sample x_samples_unif xor x_samples_gmm
        s = time.time()
        xmin, xmax = self.func.x_range[:, self.func.param_idx]
        x_samples_gmm, prob_gmm = helper.sample_tgmm(
            x_samples, scale, n_proposals, xmin, xmax)
        print('#centers = {}, gen adaptive sample tgmm time = {}'.format(len(x_samples), time.time() - s))
        x_samples_gmm = self._update_samples(context, x_samples_gmm, prob_gmm)
        filteredlen = len(x_samples_gmm)
        # TODO: divide by zero error

        if filteredlen < (n_proposals / 10):
            scale *= 0.5
            print('tune down scale')
        elif (n_proposals / 2.) < filteredlen:
            scale *= 2
            print('tune up scale')
        #print('gmm good={}'.format(len(x_samples_gmm)))
        return scale

    def _sample_from_good(self, n_samples):
        '''
        if good_samples is generated, return good_samples if not enough samples
        otherwise return n_samples of good_samples according to the weight specified in good_prob
        :param n_samples: minimum number of samples to be generated.
        :return: at most n_samples samples
        '''
        if len(self.good_samples) <= n_samples:
            return self.good_samples
        p = self.good_prob / np.sum(self.good_prob)
        x_samples_inds = np.random.choice(np.arange(len(self.good_samples)),
                                          size=n_samples, replace=False, p=p)
        return self.good_samples[x_samples_inds]

##################################################

class RandomSampler(ActiveLearner):
    def __init__(self, func):
        super(RandomSampler, self).__init__(func)
        self.name = 'random'

    def query(self, context):
        xmin = self.func.x_range[0, self.func.param_idx]
        xmax = self.func.x_range[1, self.func.param_idx]
        x_star = np.random.uniform(xmin, xmax)
        return np.hstack((x_star, context))

    def sample(self, context, **kwargs):
        return self.query(context, **kwargs)
