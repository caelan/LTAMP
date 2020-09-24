from __future__ import print_function, division

import random
import time
import numpy as np
import scipy
from scipy.stats import norm

from learn_tools import helper
from learn_tools.active_learner import ActiveLearner, format_x, THRESHOLD, tail_confidence, \
    BEST, CONFIDENT, STRADDLE, VARIANCE, REJECTION

from learn_tools.common import map_general, get_max_cores

ROBUST_COEFFICIENT = 0.01
KERNEL_EPSILON = 0.3
DEFAULT_ALPHA = 0.95

UNIFORM = 'Uniform'
ADAPTIVE = 'Adaptive'
DIVERSE = 'Diverse'
DIVERSELK = 'Diverse-LK'
BESTPROB = 'Best-prob'
DIVERSE_STATEGIES = [DIVERSE, DIVERSELK]
SAMPLE_STRATEGIES = [UNIFORM, ADAPTIVE] + DIVERSE_STATEGIES + [BESTPROB]

# def negate(fn):
#    return lambda *args, **kwargs: -fn(*args, **kwargs)

class UncertainLearner(ActiveLearner):
    """
    Active learner with a GP backend.
    """

    USE_GRADIENTS = True

    def __init__(self, func, use_var=True, beta_lambda=0.98, sample_time_limit=60,
                 check_robust=False, robust_delta=None, sample_strategy=UNIFORM,
                 verbose=True, filter_invalid_samples=True, **kwargs):
        #if inity.ndim == 1:
        #    inity = inity[:, None]
        super(UncertainLearner, self).__init__(func, **kwargs)

        self.task_lengthscale = func.task_lengthscale[func.param_idx]
        self.best_beta = None
        self.beta_lambda = beta_lambda
        self.sample_time_limit = sample_time_limit
        self.check_robust = check_robust
        self.sample_strategy = sample_strategy
        assert sample_strategy in SAMPLE_STRATEGIES, 'Invalid sample strategy.'
        if robust_delta is None:
            self.robust_delta = (func.x_range[1] - func.x_range[0]) * ROBUST_COEFFICIENT
            self.robust_delta = self.robust_delta[self.func.param_idx]
        self.min_samples = 10
        self.verbose = verbose
        self.use_var = use_var
        self.filter_invalid_samples = filter_invalid_samples
        self.reset_sample()
        self.normalize_task_kernel = False
        # TODO: clone samplers so they have their own history

    ##############################

    def clone(self):
        # TODO: clone samplers so they have their own history
        raise NotImplementedError()

    def reset_sample(self):
        """
        Clear the list of samples.
        """
        # Note: this wasn't being used as much as it should have been
        self.best_beta = None
        self.sampled_xx = np.empty(shape=(0, len(self.func.param_idx) + len(self.func.context_idx)))
        self.good_samples = np.empty(shape=(0, self.dp))
        self.good_prob = np.empty(shape=0)
        self.unif_samples = np.empty(shape=(0, self.dp))
        print('Please reset world and feature.')

    def predict(self, X, **kwargs):
        raise NotImplementedError()

    def predictive_gradients(self, x, **kwargs):
        raise NotImplementedError()

    @property
    def beta(self):
        if self.best_beta is None:
            return self.best_beta
        # Units in mu / std
        return norm.ppf(self.beta_lambda * norm.cdf(self.best_beta))  # Percent point function

    @property
    def lengthscale(self):
        # TODO: option to project away dimensions that are effectively irrelevant
        raise NotImplementedError()

    ##############################

    def score_x(self, X, alpha=0.0, max_cores=None, **kwargs):
        # TODO ZI I don't know what this score is for. Guarantee safety from the uncertainty predictions?
        # This is for setting the decision boundary when computing F1 score
        mu, var = self.predict(X, **kwargs)
        std = np.sqrt(var)

        if (alpha is None) or (alpha == 1):
            # print('Cores:', max_cores)
            t0 = time.time()
            indices = list(range(X.shape[0]))
            contexts = [X[i, self.func.context_idx] for i in indices]
            #betas = list(map(self.compute_beta, contexts))
            # Parallelization doesn't seem to help here
            max_cores = 1
            #max_cores = get_max_cores()
            betas = list(map_general(lambda c: self.compute_beta(c, n_samples=10000, iterations=None),
                                     contexts, serial=(max_cores==1), num_cores=max_cores))
            scores = np.array([(mu[i, 0] / std[i, 0]) - betas[i] for i in indices])
            # TODO ZI the scores will represent different probability of being in the super level set for each different context
            # TODO ZI There is no need to compute beta for the same contexts.
            self.best_beta = None
            print('Scored {} contexts in {:.3f} seconds'.format(len(contexts), time.time() - t0))
        else:
            scale = tail_confidence(alpha)
            scores = mu - scale * std
        return scores

    def predict_x(self, X, **kwargs):
        mu, var = self.predict(X, **kwargs)
        if not self.use_var:
            var = np.zeros(var.shape)
        return [{'mean': float(mu[i, 0]), 'variance': float(var[i, 0]), 'beta': self.beta}
                for i in range(X.shape[0])]

    # def prediction(self, feature, parameter):
    #     context = self.func.context_from_feature(feature)
    #     sample = self.func.sample_from_parameter(parameter)
    #     mu, var = self.predict(format_x(sample, context))
    #     return {
    #         'mean': float(mu[0, 0]),
    #         'variance': float(var[0, 0]),
    #         #'robust': self._is_robust(sample, context),
    #         'beta': self.beta,
    #         # TODO: compute statistics for a nearby neighborhood
    #     }

    ##############################

    def get_score_f(self, context, alpha=DEFAULT_ALPHA, **kwargs):
        # negate=True => minimize
        # negate=False => maximize
        if self.query_type in [BEST, REJECTION]:
            return self.get_prob_f(context, **kwargs)
        if self.query_type == STRADDLE:
            return self.get_straddle_f(context, alpha=alpha, **kwargs)
        if self.query_type == VARIANCE:
            return self.get_var_f(context, **kwargs)
        if self.query_type == CONFIDENT:
            #scale = tail_confidence(alpha)
            if self.beta is None:
                self.compute_beta(context)
            scale = self.beta
            return self.get_lcb_f(context, scale=scale, **kwargs)
        raise ValueError(self.query_type)

    def get_lcb_f(self, context, scale, negate=False):
        def lcb_f(x):
            x = format_x(x, context)
            mu, var = self.predict(x)
            lcb = mu - scale * np.sqrt(var)
            if negate:
                lcb = -lcb # minimize
            return lcb # maximize

        return lcb_f

    def get_lcb_fg(self, context, scale, negate=False):
        if not self.USE_GRADIENTS:
            return None

        def lcb_fg(x):
            x = format_x(x, context)
            mu, var = self.predict(x)
            f = mu - scale * np.sqrt(var)
            dmdx, dvdx = self.predictive_gradients(x)
            g = dmdx - 0.5 * dvdx / np.sqrt(var)
            if negate:
                f, g = -f, -g # minimize
            return f[0, 0], g[0, self.func.param_idx] # maximize

        # lcb_f = lambda parameter: lcb_fg(parameter)[0]
        return lcb_fg

    ##############################

    def get_prob_f(self, context, negate=True):
        def ac_f(x):
            # x = helper.grid_around_point(x, self.robust_delta, n=3, x_range=self.func.x_range)
            x = format_x(x, context)
            mu, var = self.predict(x)
            if self.use_var:
                f = mu / np.sqrt(var)
            else:
                f = mu
            if negate:
                f = -f # minimize
            return f # maximize
            # Be careful, this is typically applied to a list of points
            # return np.average(f, axis=0)[0]

        return ac_f

    def get_prob_fg(self, context, negate=True):
        if not self.USE_GRADIENTS:
            return None

        def ac_fg(x):
            # TODO: could instead sample a fixed number of points
            # x = helper.grid_around_point(x, self.robust_delta, n=3, x_range=self.func.x_range)
            x = format_x(x, context)
            mu, var = self.predict(x)
            dmdx, dvdx = self.predictive_gradients(x)
            if self.use_var:
                f = mu / np.sqrt(var)
                g = (np.sqrt(var) * dmdx - 0.5 * mu * dvdx / np.sqrt(var)) / var
                # return np.average(f, axis=0)[0], np.average(g, axis=0)[self.func.param_idx]
                g = g[0, :]
                # return f[0, 0], g[0, self.func.param_idx]
            else:
                f, g = mu, dmdx
            if negate:
                f, g = -f, -g # minimize
            return f[0, 0], g[self.func.param_idx] # maximize

        # ac_f = lambda parameter: ac_fg(parameter)[0]
        return ac_fg

    ##############################

    def _is_robust(self, x_init, context, n=200):
        """
        Tests whether the neighborhood around x_init is robust
        :param x_init: initial sample
        :param delta: the neighborhood around x_init
        :param context:
        :param n: number of nearby nearby samples to consider
        :return: True iff the neighborhood around x_init satisfies [0 < mu_x - beta*std_x]
        """

        assert self.beta is not None
        lcb_f = self.get_lcb_f(context, scale=self.beta, negate=False)  # Minimize
        lcb_fg = self.get_lcb_fg(context, scale=self.beta, negate=False)
        ep_range = np.array([x_init - self.robust_delta, x_init + self.robust_delta])
        # import pdb; pdb.set_trace()
        tx = np.random.uniform(ep_range[0], ep_range[1], (n, len(x_init)))
        ty = lcb_f(tx)[0]
        min_idx = ty.argmin()  # Find worst in neighborhood of x_init
        if ty[min_idx] <= THRESHOLD:
            return False
        # import pdb; pdb.set_trace()
        x0 = tx[min_idx]
        # TODO: use global minimize instead
        if lcb_fg is None:
            lcb_fg = lambda *args, **kw: lcb_f(*args, **kw)[0, 0]
            x_ep, y_ep, _ = scipy.optimize.fmin_l_bfgs_b(
                lcb_fg, x0=x0, approx_grad=True, bounds=ep_range.T, callback=None)  # Find worst nearby x0
        else:
            x_ep, y_ep, _ = scipy.optimize.fmin_l_bfgs_b(
                lcb_fg, x0=x0, bounds=ep_range.T, maxiter=20, callback=None)  # Find worst nearby x0
        return bool(THRESHOLD < y_ep)

    def _extract_robust_samples(self, samples, prob, context):
        rt_samples = np.zeros((0, self.dp))
        rt_prob = np.zeros(0)
        for i in range(len(samples)):
            if self._is_robust(samples[i], context):
                rt_samples = np.vstack((samples[i], rt_samples))
                rt_prob = np.hstack((prob[i], rt_prob))
        return rt_samples, rt_prob

    ##############################

    def set_world_feature(self, world, feature):
        self.world = world
        self.feature = feature

    def _filter_samples(self, new_x_samples, new_prob):
        # return new_x_samples, new_prob
        if not self.filter_invalid_samples:
            return new_x_samples, new_prob
        new_good_inds = []
        start_time = time.time()
        for i in range(len(new_x_samples)):
            parameter = self.func.parameter_from_sample(new_x_samples[i])
            if self.func.collector.validity_test(self.world, self.feature, parameter):
                new_good_inds.append(i)
            print('{} valid parameter after {}/{} iterations and {:.3f} sec'.format(
                len(new_good_inds), i + 1, len(new_x_samples), time.time() - start_time))
        return new_x_samples[new_good_inds], new_prob[new_good_inds]


    def _update_samples(self, context, new_x_samples, new_prob):
        '''

        :param context: context under which samples need to be generated
        :param new_x_samples: new samples of parameters
        :param new_prob: probabilities (weights) for the new samples
        add new samples as long as they are in high probability super level set
        '''

        def ac_f(x):
            # TODO: acquisition predicate or unify with lcb_f
            x = format_x(x, context)
            mu, var = self.predict(x)
            ret = mu / np.sqrt(var)
            # self.beta < mu / np.sqrt(var)
            # self.beta*np.sqrt(var) < mu
            # 0 < self.beta*np.sqrt(var) - mu
            return ret.T[0]
            # s = 0.0
            # lcb = (mu - s*np.sqrt(var)).flatten()
            # return lcb

        dx = len(new_x_samples)
        scores = ac_f(new_x_samples)
        assert self.beta is not None
        new_good_inds = (self.beta <= scores)
        # new_good_inds = (THRESHOLD < ac_f(new_x_samples))
        new_x_samples, new_prob = self._filter_samples(new_x_samples[new_good_inds], new_prob[new_good_inds])
        if self.check_robust:
            new_x_samples, new_prob = self._extract_robust_samples(new_x_samples, new_prob, context)
        self.good_samples = np.vstack((new_x_samples, self.good_samples))
        self.good_prob = np.hstack((new_prob, self.good_prob))
        print('beta={:.3f} | mean={:.3f} | max={:.3f} | {}/{} are filtered | {}/{} have good score'.format(
            self.beta, np.mean(scores), np.max(scores), len(new_x_samples), sum(new_good_inds), sum(new_good_inds), dx))
        return new_x_samples

    def _gen_adaptive_samples(self, context, n_proposals=10000, n_samples=100):
        """
        Generate adaptive samples with rejection sampling, where the proposal
        distribution is uniform and truncated Gaussian mixtures.
        Args:
            context: the context the generator is conditioned upon.
            n_proposals: number of proposals per iteration.
            n_samples: minimum number of samples to be generated.
        """
        # TODO ZI: try to adapt it to RF, NN
        # self.good_samples = np.empty(shape=(0, self.dp))
        # self.good_prob = np.empty(shape=0)

        x_samples = self.xx[THRESHOLD < np.squeeze(self.yy)][:, self.func.param_idx]
        if len(self.sampled_xx) == 0:
            x_samples = np.vstack((x_samples, self.query_best_prob(context)[self.func.param_idx]))
        else:
            x_samples = np.vstack((x_samples, self.sampled_xx[:, self.func.param_idx]))
        assert len(x_samples) != 0
        x_samples = self._update_samples(context, x_samples, np.ones(len(x_samples)))

        tgmm_scale = np.array([1.] * len(self.func.param_idx))  # 1. * np.array(self.lengthscale[self.func.param_idx])
        sampled_cnt = 0
        t_start = time.time()
        while not sampled_cnt or (len(self.good_samples) <= n_samples):
            if self.sample_time_limit < (time.time() - t_start):
                print('Elapsed sampling time = {}, sampling iterations = {}, not enough good samples.'.format(
                    time.time() - t_start, sampled_cnt))
                break
            sampled_cnt += 1
            print('Iteration: {} | Samples: {} | Good Samples: {} | Time: {:.3f}'.format(
                sampled_cnt, len(x_samples), len(self.good_samples), time.time() - t_start))

            if self.sample_strategy != UNIFORM:
                s = time.time()
                self._gen_uniform(context, n_proposals)
                print('gen uniform time:', time.time() - s)
                if len(x_samples) != 0 and len(self.good_samples) < n_samples:
                    s = time.time()
                    tgmm_scale = self._gen_adaptive(context, n_proposals, x_samples, tgmm_scale)
                    print('gen adaptive time: ', time.time() - s)
                else:
                    self._gen_uniform(context, n_proposals)
            else:
                self._gen_uniform(context, n_proposals * 2)

            x_samples = self._sample_from_good(n_samples)
            # TODO: rather than subsample here, do it at test time

        print('{} samples are generated with the {} sampler in {} seconds.'.format(len(self.good_samples),
                                                                                   self.sample_strategy,
                                                                                   time.time() - t_start))
        return x_samples

    def _sample_adaptive(self, context, ordered=False, **kwargs):
        """
        Returns one sample from the high probability super level set for a given context,
        using the adaptive sampler.
        """
        if len(self.unif_samples) < self.min_samples:
            xx = self._gen_adaptive_samples(context, **kwargs)
            # TODO: why is tiling needed here?
            self.unif_samples = np.hstack((xx, np.tile(context, (xx.shape[0], 1))))
        if len(self.unif_samples) == 0:
            return None
        # np.random.randint and random.randint have different upper behaviors
        index = 0 if ordered else random.randint(0, len(self.unif_samples) - 1)
        self.sampled_xx = np.vstack((self.sampled_xx, self.unif_samples[index]))
        self.unif_samples = np.delete(self.unif_samples, index, axis=0)
        return self.sampled_xx[-1]

    ##############################

    def _update_learned_kernel(self):
        # Learning task-level kernel lengthscale
        if (self.sample_strategy == DIVERSELK) and (2 <= len(self.sampled_xx)):
            sum_task_lengthscal = sum(self.task_lengthscale)
            d = helper.important_d(self.sampled_xx[-1, self.func.param_idx],
                                   self.sampled_xx[:-1, self.func.param_idx],
                                   self.task_lengthscale)
            self.task_lengthscale[d] *= (1. - KERNEL_EPSILON)
            if self.normalize_task_kernel:
                self.task_lengthscale = self.task_lengthscale / sum(self.task_lengthscale) * sum_task_lengthscal
            # TODO: maybe normalize the lengthscal to sum to x's dimension
        # End of learning task-level kernel lengthscale

    def _sample_diverse(self, context, **kwargs):
        self._update_learned_kernel()
        if len(self.good_samples) < self.min_samples:
            _ = self._gen_adaptive_samples(context, **kwargs)
        '''
        # TODO: compare against using the GP lengthscale
        if len(self.sampled_xx) == 0:
            self.sampled_xx = np.array([self.query_best_prob(context)])
        else:
            self._update_learned_kernel()
            if len(self.good_samples) < self.min_samples:
                _ = self._gen_adaptive_samples(context, **kwargs)

            if len(self.good_samples) == 0:
                return None
            # Maximize diversity
            index = helper.argmax_condvar(self.good_samples, self.sampled_xx[:, self.func.param_idx], l=None)
            # TODO change l in argmax_condvar to l=self.task_lengthscale; current task_lengthscale does not reflect the right one
            self.sampled_xx = np.vstack((self.sampled_xx, np.hstack((self.good_samples[index], context))))
            self.good_samples = np.delete(self.good_samples, index, axis=0)
            self.good_prob = np.delete(self.good_prob, index)
        
        '''
        if len(self.good_samples) == 0:
            return None
        if len(self.sampled_xx) == 0:
            prob_func = self.get_prob_f(context, negate=True)
            index = prob_func(self.good_samples).argmin()
        else:
            # Maximize diversity
            index = helper.argmax_condvar(self.good_samples, self.sampled_xx[:, self.func.param_idx], l=None)
            # TODO change l in argmax_condvar to l=self.task_lengthscale; current task_lengthscale does not reflect the right one
        self.sampled_xx = np.vstack((self.sampled_xx, np.hstack((self.good_samples[index], context))))
        self.good_samples = np.delete(self.good_samples, index, axis=0)
        self.good_prob = np.delete(self.good_prob, index)
        return self.sampled_xx[-1]

    def sample(self, context, **kwargs):
        """
        Returns one sample from the high probability super level set for a given context.
        """
        if self.sample_strategy == BESTPROB:
            self.sampled_xx = np.vstack((self.sampled_xx, self.query_best_prob(context)))
            return self.sampled_xx[-1]
        if self.sample_strategy in DIVERSE_STATEGIES:
            return self._sample_diverse(context, **kwargs)
        return self._sample_adaptive(context, **kwargs)

    ##############################

    def query_best_prob(self, context, n_samples=10000, **kwargs):
        """
        Returns the input that has the highest probability to be in the super
        level set for a given context.
        """
        x0, _ = helper.find_closest_positive_context_param(
            context, self.xx, self.yy, self.func.param_idx, self.func.context_idx,
            weights=np.reciprocal(self.lengthscale))

        ac_f = self.get_prob_f(context, negate=True)  # Maximize (minimize negated)
        ac_fg = self.get_prob_fg(context, negate=True)
        # scale = tail_confidence(CONFIDENCE_ALPHA)
        # ac_f = self.get_lcb_f(context, scale=scale, negate=True)
        # ac_fg = self.get_lcb_fg(context, scale=scale, negate=True)
        x0 = np.vstack((x0, self.xx[THRESHOLD < np.squeeze(self.yy)][:, self.func.param_idx]))
        x_star, y_star = helper.global_minimize(
            ac_f, ac_fg, self.func.x_range[:, self.func.param_idx], n=n_samples, guesses=x0, **kwargs)
        # TODO: could initialize randomly
        # TODO: many of the failure cases seem like times in which best_beta < 0
        # TODO: apply self.func.collector.validity_test(world, feature, parameter)

        self.best_beta = -y_star
        if self.best_beta < 0:
            print('Warning! Cannot find any parameter to be super level set '
                  'with more than 0.5 probability!')
        print('best beta={:.3f} | beta={:.3f}'.format(self.best_beta, self.beta))
        # TODO: var might be too small
        #print('mu={:.5f} | var={:.5f}'.format(*map(float, self.predict(format_x(x_star, context)))))
        if self.best_beta < self.beta:
            raise ValueError('Beta cannot be larger than best beta.')
        return np.hstack((x_star, context))

    def compute_beta(self, context, **kwargs):
        # TODO: test validity?
        self.query_best_prob(context, **kwargs)
        return self.beta

    ##############################

    def get_highest_var_idx(self, X, **kwargs):
        _, var = self.predict(X, **kwargs)
        return var.argmax()

    def get_highest_lse_idx(self, X, alpha=DEFAULT_ALPHA, **kwargs):
        # https://las.inf.ethz.ch/files/gotovos13active.pdf
        scale = tail_confidence(alpha)
        mu, var = self.predict(X, **kwargs)
        lse = - np.abs(mu) + scale * np.sqrt(var)
        return lse.argmax()

    # def select_index(self, *args, **kwargs):
    #     if self.query_type == VARIANCE:
    #         return self.get_highest_var_idx(*args, **kwargs)
    #     if self.query_type == STRADDLE:
    #         return self.get_highest_lse_idx(*args, **kwargs)
    #     raise ValueError(self.query_type)

    ##############################

    # TODO: other acquisition functions
    # https://las.inf.ethz.ch/files/gotovos13active.pdf

    def get_var_f(self, context, negate=True, **kwargs):
        # TODO: unify with get_highest_var_idx

        def ac_f(x):
            x = format_x(x, context)
            _, var = self.predict(x, **kwargs)
            if negate:
                var = -var # minimize
            return var # maximize

        return ac_f

    def get_straddle_f(self, context, alpha, negate=True, noise=False, **kwargs):
        # TODO: unify with get_highest_lse_idx
        scale = tail_confidence(alpha)

        def ac_f(x):
            x = format_x(x, context)
            mu, var = self.predict(x, noise=noise, **kwargs)  # self.model.predict_quantiles(x, quantiles=(2.5, 97.5))
            # TODO: obtain gradient wrt query point for optimization
            # if x.shape[0] == 1:
            #    #print(self.model.predict_jacobian(Xnew=x))
            #    dmu_dX, dv_dX = self.model.predictive_gradients(Xnew=x)
            #    dmu_dX = dmu_dX[:, :, 0]
            #    ds_dX = dv_dX / (2 * np.sqrt(var))
            #    #print(dmu_dX, ds_dX)
            #    #print(dmu_dX.shape, dv_dX.shape)
            lse = - np.abs(mu) + scale * np.sqrt(var)
            if negate:
                lse = -lse # minimize
            return lse # maximize

        return ac_f

    ##############################

    def query_lse(self, context, alpha=DEFAULT_ALPHA):
        """
        Returns the next active query on the function in a particular context
        using level set estimation.
        We here implement the straddle algorithm from
        B. Bryan, R. C. Nichol, C. R. Genovese, J. Schneider, C. J. Miller, and L. Wasserman,
        "Active learning for identifying function threshold boundaries," in NIPS, 2006.
        """
        x0, _ = helper.find_closest_positive_context_param(
            context, self.xx, self.yy, self.func.param_idx, self.func.context_idx, weights=None)
        # TODO: could use the lengthscale from the model
        # TODO: add constraints to the optimizer

        ac_f = self.get_score_f(context, alpha=alpha, negate=True) # Maximize (minimize negated)
        # https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/optimization/acquisition_optimizer.py
        # https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/acquisitions/base.py#L42
        # https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/acquisitions/LCB.py
        # https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/models/gpmodel.py#L129
        x_star, _ = helper.global_minimize(
            f=ac_f, fg=None, x_range=self.func.x_range[:, self.func.param_idx], n=10000, guesses=x0)
        return np.hstack((x_star, context))
