from __future__ import print_function, division

import numpy as np
import time

from collections import Counter

from pybullet_tools.utils import elapsed_time
from learn_tools.active_learner import ActiveLearner, format_x, THRESHOLD, tail_confidence
from learn_tools.learner import rescale, DEFAULT_INTERVAL, TRANSFER_WEIGHT, NORMAL_WEIGHT
from learn_tools import helper

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier # TODO: python2.7 naming issue

NEARBY_COEFFICIENT = 0.5
BETA = 0.5

CLASSIFIER = '_classifier'
REGRESSOR = '_regressor'
NN = 'nn'
RF = 'rf'

NN_CLASSIFIER = NN + CLASSIFIER
RF_CLASSIFIER = RF + CLASSIFIER
CLASSIFIERS = (NN_CLASSIFIER, RF_CLASSIFIER,)

NN_REGRESSOR = NN + REGRESSOR
RF_REGRESSOR = RF + REGRESSOR
REGRESSORS = (NN_REGRESSOR, RF_REGRESSOR,)

# TODO: dummy learners (e.g. returns the mean)
NN_MODELS = (NN_CLASSIFIER, NN_REGRESSOR,)
RF_MODELS = (RF_CLASSIFIER, RF_REGRESSOR,)

CLASSIFIER_INTERVAL = (0, +1)
CLASSIFIER_THRES = np.mean(CLASSIFIER_INTERVAL)

############################################################

def get_sklearn_model(model_type, hidden_layers=[50, 50], n_estimators=500, **kwargs):
    if model_type == NN_CLASSIFIER:
        return MLPClassifier(hidden_layer_sizes=hidden_layers,
                             max_iter=200, shuffle=True, early_stopping=True,
                             validation_fraction=0.1, verbose=False, **kwargs)
    elif model_type == NN_REGRESSOR:
        return MLPRegressor(hidden_layer_sizes=hidden_layers,
                            max_iter=200, shuffle=True, early_stopping=True,
                            validation_fraction=0.1, verbose=False, **kwargs)
    elif model_type == RF_CLASSIFIER:
        return ExtraTreesClassifier(n_estimators=n_estimators, bootstrap=True,
                                    oob_score=True, warm_start=False, **kwargs)
    elif model_type.startswith('rf_'):
        return ExtraTreesRegressor(n_estimators=n_estimators, criterion='mse', bootstrap=True, # mse | mae
                                   oob_score=True, warm_start=False, **kwargs)
    raise ValueError(model_type)

############################################################

# TODO: rethink the inheritance

def _get_sample_weight(learner):
    #from sklearn.utils import compute_sample_weight
    n_tot = learner.xx.shape[0]
    if not learner.use_sample_weights:
        sample_weight = np.ones(n_tot)
        return sample_weight

    pos_idx = (THRESHOLD < learner.yy)
    n_pos = np.sum(pos_idx)
    print('number of positive datapoints = {}'.format(n_pos))
    sample_weight = np.ones(n_tot)
    sample_weight[pos_idx] = float(n_tot - n_pos) / n_pos
    return sample_weight

def retrain_sklearer(learner, newx=None, newy=None, new_w=None):
    start_time = time.time()
    if (newx is not None) and (newy is not None):
        learner.xx = np.vstack((learner.xx, newx))
        if learner.model_type in CLASSIFIERS:
            newy = (THRESHOLD < newy)
        learner.yy = np.hstack((learner.yy, newy))
        if new_w is not None:
            learner.weights = np.hstack((learner.weights, new_w))

    # TODO: what was this supposed to do?
    #self.xx, self.yy, sample_weight = shuffle(self.xx, self.yy, _get_sample_weight(self))
    if learner.model.__class__ in [MLPClassifier, MLPRegressor]:
        learner.model.fit(learner.xx, learner.yy)
    else:
        # TODO: preprocess to change the mean_function
        # https://scikit-learn.org/stable/modules/preprocessing.html
        xx, yy = learner.xx, learner.yy
        weights = None
        if (learner.transfer_weight is not None) and (2 <= len(Counter(learner.weights))):
            assert 0. <= learner.transfer_weight <= 1.
            #weights = learner.weights
            weights = np.ones(yy.shape)
            total_weight = sum(weights)
            normal_indices = np.argwhere(learner.weights==NORMAL_WEIGHT)
            weights[normal_indices] = total_weight*(1 - learner.transfer_weight) / len(normal_indices)
            transfer_indices = np.argwhere(learner.weights==TRANSFER_WEIGHT)
            weights[transfer_indices] = total_weight*learner.transfer_weight/len(transfer_indices)

            print('Transfer weight: {:.3f} | Normal: {} | Transfer: {} | Other: {}'.format(
                learner.transfer_weight, len(normal_indices), len(transfer_indices),
                len(weights) - len(normal_indices) - len(transfer_indices)))
        #weights = 1./len(yy) * np.ones(len(yy))
        #num_repeat = 0
        # if num_repeat != 0:
        #     xx = np.vstack([xx, xx[:num_repeat]])
        #     yy = np.vstack([yy, yy[:num_repeat]])
        #     weights = np.concatenate([
        #         1./len(yy) * np.ones(len(yy)),
        #         1./num_repeat * np.ones(num_repeat),
        #     ])
        learner.model.fit(xx, yy, sample_weight=weights)
    print('Trained in {:.3f} seconds'.format(elapsed_time(start_time)))
    learner.metric()
    learner.trained = True

############################################################

def rf_predict(model, X):
    predictions = [model.predict(X) for model in model.estimators_]
    #if type(model) not in [ExtraTreesRegressor, ExtraTreesClassifier]: TODO: python2 naming issue
    #    raise ValueError(model)
    #if isinstance(model, ExtraTreesClassifier):
    if 'classifier' in model.__class__.__name__.lower():
        lower, upper = DEFAULT_INTERVAL
        predictions = [lower + (upper - lower) * pred for pred in predictions]
    mu = np.mean(predictions, axis=0)
    var = np.var(predictions, axis=0)
    return mu, var

############################################################

# TODO: the keras models don't seem to work well when using multiple processors

class ActiveNN(ActiveLearner):
    def __init__(self, func, initx=np.array([[]]), inity=np.array([]),
                 model_type=NN_CLASSIFIER, epochs=1000,
                 validation_split=0.1, batch_size=100,
                 use_sample_weights=False, verbose=True, **kwargs):
        print('{} using {}'.format(self.__class__.__name__, model_type))
        if model_type in CLASSIFIERS:
            inity = (THRESHOLD < inity).astype(float)
        super(ActiveNN, self).__init__(func, initx, inity, **kwargs)
        self.model_type = model_type
        self.model = get_sklearn_model(model_type)
        self.use_sample_weights = use_sample_weights
        self.verbose = verbose
        self.name = 'nn_{}'.format(model_type)
        self.epochs = epochs
        self.validation_split = validation_split
        self.batch_size = batch_size

    ##############################

    def score_x(self, X, alpha=0.0, **kwargs):
        mu, var = self.predict(X)
        #if (alpha is not None) and (self.model_type in CLASSIFIERS):
        #    scale = tail_confidence(alpha)
        #    return mu - scale*std
        return mu.reshape(-1, 1)

    def predict_x(self, X):
        mu, _ = self.predict(X)
        return [{'mean': float(mu[i])} for i in range(X.shape[0])]

    # def prediction(self, feature, parameter):
    #     context = self.func.context_from_feature(feature)
    #     sample = self.func.sample_from_parameter(parameter)
    #     x = format_x(sample, context)
    #     mu, var = self.predict(x)
    #     #self.model.oob_score_
    #     #self.model.oob_decision_function_
    #     #self.feature_importances_
    #     #p = self.model.predict_proba(x)[:,-1]
    #     # estimators_
    #     #self.model.apply(x)
    #     return {
    #         'mean': float(mu[0]),
    #         # TODO: compute statistics for a nearby neighborhood
    #     }

    def metric(self):
        mu, var = self.predict(self.xx)
        pred = (THRESHOLD < mu)
        label = (THRESHOLD < self.yy)
        #try:
        tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
        acc = float(tn + tp) / len(label)
        print('accuracy = {}, fpr = {}, fnr = {}'.format(
            acc, float(fp)/(tn + fp), float(fn)/(tp + fn)))
        return acc
        #except:
        #    print('error in metric')
        #    return None

    ##############################

    def predict(self, X):
        if type(self.model) in [ExtraTreesRegressor, ExtraTreesClassifier]:
            mu, _ = rf_predict(self.model, X)
        elif self.model_type in CLASSIFIERS:
            mu = self.model.predict_proba(X)[:, -1]
            mu = np.array([rescale(v, interval=CLASSIFIER_INTERVAL) for v in mu])
            mu = mu.reshape([-1, 1])
        elif self.model_type in REGRESSORS:
            mu = self.model.predict(X)
            mu = mu.reshape([-1, 1])
        else:
            raise NotImplementedError(self.model_type)
        var = None
        return mu, var

    def get_prob_f(self, context, negate=True):
        def ac_f(x):
            x = format_x(x, context)
            mu, _ = self.predict(x)
            if negate:
                mu = -mu
            return mu
        return ac_f

    get_score_f = get_prob_f

    def query_best_prob(self, context, guess_per_axis=5, n_samples=10000):
        x0, _ = helper.find_closest_positive_context_param(
            context, self.xx, self.yy, self.func.param_idx, self.func.context_idx)
        ac_f = self.get_prob_f(context, negate=True) # Maximize (minimize negated)

        x_range = self.func.x_range
        guesses = helper.grid_around_point(
            x0, NEARBY_COEFFICIENT * (x_range[1] - x_range[0]),
            n=guess_per_axis, x_range=x_range)
        x_star, y_star = helper.global_minimize(
            ac_f, None, x_range[:, self.func.param_idx],
            n=n_samples, guesses=guesses)
        print('x_star={}, y_star={}'.format(x_star.tolist(), y_star.tolist()))
        return np.hstack((x_star, context))

    ##############################

    def _update_samples(self, context, new_x_samples, new_prob):
        def ac_f(param):
            x = np.hstack((param, np.tile(context, (param.shape[0], 1))))
            mu, _ = self.predict(x)
            #if self.model_type in REGRESSORS: # TODO: why was this here?
            #    mu = helper.sigmoid(mu)
            return np.squeeze(mu)

        new_good_inds = (BETA < ac_f(new_x_samples))
        self.good_samples = np.vstack((new_x_samples[new_good_inds], self.good_samples))
        self.good_prob = np.hstack((new_prob[new_good_inds], self.good_prob))
        print('{}/{} are good'.format(len(new_x_samples[new_good_inds]), len(new_x_samples)))

    def gen_adaptive_samples(self, context, n_proposals=10000, n_samples=10):
        t_start = time.time()
        self.good_samples = np.empty(shape=(0, self.dp))
        self.good_prob = np.empty(shape=0)

        x_samples = self.xx[THRESHOLD < np.squeeze(self.yy)][:,self.func.param_idx]
        # TODO: sample is undefined
        #x_samples = np.vstack((x_samples, self.sample(context)[:, self.func.param_idx]))
        # TODO: query the best prob

        #good_inds = (BETA < ac_f(x_samples))
        #if (len(x_samples) == 1) and (len(good_inds) == 0):
        #    raise ValueError('no good samples to start with')
        self._update_samples(context, x_samples, np.ones(len(x_samples)))

        lengthscale = 1. * np.array(self.func.lengthscale_bound[1][self.func.param_idx])
        first_iter = True
        while first_iter or (len(self.good_samples) <= n_samples):
            first_iter = False # make sure it sample at least once
            if 60 < (time.time() - t_start):
                break
            self._gen_uniform(context, n_proposals)
            if self.is_adaptive:
                self._gen_adaptive(context, n_proposals, x_samples, lengthscale)
            x_samples = self._sample_from_good(n_samples)
        print('good samples len = {}'.format(len(self.good_samples)))
        return x_samples

    def sample(self, context, **kwargs):
        xx = self.gen_adaptive_samples(context, **kwargs)
        return np.hstack((xx, np.tile(context, (xx.shape[0], 1))))[-1]

    ##############################

    def retrain(self, **kwargs):
        return retrain_sklearer(self, **kwargs)
