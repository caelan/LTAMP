from __future__ import print_function, division

import GPy as gpy
import numpy as np
import scipy
import math

from learn_tools.uncertain_learner import UncertainLearner
#from learn_tools.common import map_general
from learn_tools.learner import SUCCESS, FAILURE, NORMAL_WEIGHT, TRANSFER_WEIGHT

# These are all training parameters (not model parameters)

BATCH_GP = 'gp_batch'

STRADDLE_GP = 'gp_straddle'
MAXVAR_GP = 'gp_maxvar'
CONTINUOUS_ACTIVE_GP = (MAXVAR_GP, STRADDLE_GP)

BATCH_MAXVAR_GP = 'gp_batch_maxvar'
BATCH_STRADDLE_GP = 'gp_batch_straddle'
BATCH_ACTIVE_GP = (BATCH_MAXVAR_GP, BATCH_STRADDLE_GP)

STRADDLE_ACTIVE_GP = (STRADDLE_GP, BATCH_STRADDLE_GP)
VAR_ACTIVE_GP = (MAXVAR_GP, BATCH_MAXVAR_GP)
# TODO: factor into learner, acquisition, and selection

ACTIVE_LEARNERS_GP = CONTINUOUS_ACTIVE_GP + BATCH_ACTIVE_GP # TODO: move to uncertain learner
GP_MODELS = (BATCH_GP,) + ACTIVE_LEARNERS_GP

#DEFAULT_INTERVAL = (1e-1, 1e+1)
DEFAULT_INTERVAL = (1e-2, 1e+2)
#DEFAULT_INTERVAL = (1e-3, 1e+3) # Limit before performance degrades


HYPERPARAM_FROM_KERNEL = {
    # TODO: deprecate
}

##############################

# exponential, Matern32, Matern52, Brownian, linear, bias, rbfcos, periodic_matern32,  TruncLinear
# TODO: IndependentOutputs kernel for different data sources?
# TODO: make a new instance per usage in a conditional generator

def constrain_interval(parameter, interval, **kwargs):
    if interval is None:
        return
    if not (isinstance(interval, tuple) or isinstance(interval, list)):
        interval = [interval]
    if len(interval) == 1:
        interval = (interval, interval)
    lower, upper = interval
    if lower == upper:
        parameter.constrain_fixed(lower, **kwargs)
    else:
        parameter.constrain_bounded(lower, upper, **kwargs)

def get_parameters(model):
    #print(self.model.name)
    #print(self.model.parameters)
    #print(self.model.grep_param_names(regexp=''))
    parameters = {n: list(model[n]) for n in model.parameter_names()}
    #print(self.model.parameter_names_flat(include_fixed=True).tolist())
    #print({param.name: list(param) for param in self.model.flattened_parameters})
    return parameters

##############################

class FixedMapping(gpy.core.Mapping):
    def update_gradients(self, *args, **kwargs):
        return None

class TransferMapping(FixedMapping):
    def __init__(self, model, **kwargs):
        self.model = model
        super(TransferMapping, self).__init__(input_dim=model.input_dim, output_dim=1, **kwargs)
    def f(self, X):
        mu, var = self.model.predict(X, include_likelihood=True)
        return mu

##############################

class ActiveGP(UncertainLearner):
    DEFAULT_WEIGHT = 0.
    def __init__(self, func,
                 kernel_name='MLP', hyperparameters=None,  # RBF | Matern52 | MLP
                 kern_var_range=DEFAULT_INTERVAL, gauss_var_range=(1e-6, 1e+2), **kwargs):
        """
        func: scoring function
        initx: initial inputs
        inity: initial outputs
        query_type: type of query ('lse' or 'best_prob')
        flag_lk: False if using diverse sampler with a fixed kernel;
                 True if using diverse sampler with a kernel learned online.
        is_adaptive: True if using the adaptive sampler; False if using the
        diverse sampler; None if using rejection sampler with uniform proposal
        distribution.
        task_lengthscale: the inverse length scale of the kernel for diverse sampling.
        betalambda: a hyper parameter
        sample_time_limit: time limit (seconds) for generating samples with (adaptive)
        rejection sampling.
        """
        super(ActiveGP, self).__init__(func, **kwargs)
        self.kernel_name = kernel_name
        self.name = 'gp-{}-{}'.format(kernel_name, self.query_type).lower()
        self.hyperparameters = hyperparameters
        self.kern_var_range = kern_var_range # Kernel variance range
        self.gauss_var_range = gauss_var_range # Gaussian noise variance range
        self.model = None
        self.mean_fn = None
        # TODO: save_model, load_model

    ##############################

    def get_var_range(self):
        if hasattr(self, 'mat_var_range'):
            return self.mat_var_range
        return self.kern_var_range

    @property
    def kernel(self):
        if isinstance(self.model.kern, gpy.kern.Add):
            return self.model.kern.parts[0]
        return self.model.kern

    @property
    def ard_parameters(self):
        if isinstance(self.kernel, gpy.kern.MLP):
            return self.kernel.weight_variance
        return self.kernel.lengthscale

    @property
    def lengthscale(self):
        if isinstance(self.model.kern, gpy.kern.MLP):
            return np.reciprocal(self.ard_parameters)
        return self.ard_parameters

    def metric(self):
        return self.model.log_likelihood()
        # return self.model.objective_function()

    def predict(self, X, noise=True):
        if not self.trained:
            self.retrain()
        #return self.model.predict_noiseless(X)
        # include_likelihood=True by default
        # https://github.com/zi-w/Kitchen2D/blob/ac7f38129668e38d22f3caf6d44da74c6dba80e1/active_learners/active_gp.py#L64
        # https://gpy.readthedocs.io/en/deploy/GPy.core.html
        mu, var = self.model.predict(X, include_likelihood=noise) # Adds Gaussian_noise.variance
        #if self.transfer_weight in [1, 3]:
        #    sim_mu, _ = self.sim_model.predict(X, include_likelihood=True)
        #    mu += sim_mu
        return mu, var

    def predictive_gradients(self, x):
        dmdx, dvdx = self.model.predictive_gradients(x)
        dmdx = dmdx[0, :, 0]
        dvdx = dvdx[0, :]
        return dmdx, dvdx

    ##############################

    def train_model(self, xx, yy, weights=None, mean_fn=None, hyperparameters=None,
                    iterations=1000, num_restarts=3, num_processes=1):
        assert xx.shape[0] == yy.shape[0]

        input_dim = self.func.x_range.shape[1]
        # Residual mean function
        # TODO: could also manually adjust the Y examples based on the mean_function
        # https://gpy.readthedocs.io/en/deploy/_modules/GPy/examples/regression.html#simple_mean_function
        if mean_fn is None:
            mean_function = None
        else:
            #def update_gradients(a, b):
            #    return None
            mean_function = TransferMapping(self.sim_model)
            #mean_function = FixedMapping(input_dim, 1)
            #mean_function.f = mean_fn
            #mean_function.update_gradients = update_gradients
            #mean_function.update_gradients = lambda a, b: None  # Don't learn weights
            # mean_function = gpy.mappings.Linear(1, 1) # Learn weights

        if self.kernel_name == 'RBF':
            # TODO: use active_dims to prune less important parameters
            kernel = gpy.kern.RBF(input_dim, ARD=True, active_dims=None)
        elif self.kernel_name == 'Matern52':
            kernel = gpy.kern.Matern52(input_dim, ARD=True)
        elif self.kernel_name == 'MLP':
            # https://gpy.readthedocs.io/en/deploy/GPy.kern.src.html#module-GPy.kern.src.mlp
            # Bias term is already included
            kernel = gpy.kern.MLP(input_dim, ARD=True)
            # TODO: combine MLP kernel with others kernels
        else:
            raise ValueError(self.kernel_name)

        # gpy.kern.Precomputed
        # use_weights &= np.count_nonzero(weights)
        if weights is not None:
            # noise_kernel = gpy.kern.White(input_dim, variance=1.)
            # Prediction excludes any noise learnt by this Kernel, so be careful using this kernel.
            noise_kernel = gpy.kern.WhiteHeteroscedastic(input_dim, num_data=xx.shape[0])
            kernel = gpy.kern.Add([kernel, noise_kernel])

        # If normalizer is False, no normalization will be done.
        # If it is None, we use GaussianNorm(alization).
        # If normalizer is True, we will normalize using Standardize.
        # https://gpy.readthedocs.io/en/deploy/GPy.core.html
        model = gpy.models.GPRegression(xx, yy, kernel=kernel, normalizer=None,
                                        mean_function=mean_function)  # None performs the best
        # GPHeteroscedasticRegression does not make inference on the noise outside the training set
        # model = gpy.models.GPHeteroscedasticRegression(xx, yy, kernel=kernel, Y_metadata=None)
        for param in model.parameter_names():
            constrain_interval(model[param], self.get_var_range(), warning=False)

        # if isinstance(model.kern, gpy.kern.Matern52):
        #     if self.func.lengthscale_bound is not None:
        #         for i in range(self.func.dx):
        #             interval = (self.func.lengthscale_bound[0][i], self.func.lengthscale_bound[1][i])
        #             constrain_interval(model.kern.lengthscale[i:i + 1], interval, warning=False)
        #     constrain_interval(model['Mat52.variance'], self.kern_var_range, warning=False)

        # https://github.com/SheffieldML/GPy/issues/506
        # model['.*variance'].constrain_bounded(*self.kern_var_range, warning=False)
        constrain_interval(model['Gaussian_noise.variance'], self.gauss_var_range, warning=False)
        # http://www.nathan-rice.net/gp-python/
        # prior = GPy.priors.Gamma.from_EV(5, 1)
        # data_labels, Y_metadata

        if weights is not None:  # and (self.transfer_weight is not None):
            # TODO: sum kernels defined on different datasets
            # variances = weights
            # variances = np.zeros(weights.shape)
            variances = np.array(weights)
            if self.transfer_weight is not None:
                normal_indices = np.argwhere(weights == NORMAL_WEIGHT)
                max_variance = np.var([SUCCESS, FAILURE])  # Maximum entropy
                transfer_indices = np.argwhere(weights == TRANSFER_WEIGHT)
                variances[transfer_indices] = max_variance * (1 - self.transfer_weight)
                print('Weight: {:.3f} | Variance: {:.3f} | Normal: {} | Transfer: {} | Other: {}'.format(
                        self.transfer_weight, max_variance, len(normal_indices), len(transfer_indices),
                        len(variances) - len(normal_indices) - len(transfer_indices)))
            # model['sum.white.variance'].constrain_fixed(0.5) # Fixed supersedes interval constraints
            model['sum.white_hetero.variance'].constrain_fixed(variances)

        if hyperparameters is not None:
            for name, value in list(hyperparameters.items()):  # [:-1]:
                #if name == 'Gaussian_noise.variance': # Leave noise parameter free
                #    continue
                # TODO: ensure that these are the same dimension
                # if name not in model:
                #    continue
                # model[name] = value # Only sets the initial value
                model[name].constrain_fixed(value)
                # Gaussian, Uniform, LogGaussian, Gamma, InverseGamma, HalfT, Exponential, StudentT
                # https://gpy.readthedocs.io/en/deploy/_modules/GPy/core/parameterization/priors.html
                # if len(value) == 1:
                #    # Domain of prior and constraint have to match, please unconstrain if you REALLY wish to use this prior
                #    model[name].set_prior(gpy.priors.LogGaussian(mu=value[0], sigma=PRIOR_STD))
                # for i, param in enumerate(model[name]):
                #    param.set_prior(gpy.priors.Gaussian(mu=value, sigma=PRIOR_STD))

        if self.verbose:
            # print(model.mean_function)
            print('X: {} | Y: {}'.format(xx.shape, yy.shape))
            # print('Success: {:.3f}'.format(np.sum((THRESHOLD < yy) / len(yy))))
            # print('Mean: {:.3f}'.format(np.average(yy).tolist()))

        if model.is_fixed: # or (xx.shape[0] == 0)
            return model

        # These GP hyper parameters need to be calibrated for good uncertainty predictions.
        optimizer = None  # None | scg | fmin_tnc | simplex | lbfgsb | lbfgs | sgd
        # TODO: randomize on a logarithmic scale
        assert 1 <= num_restarts
        num_attempts = 10
        for attempt in range(num_attempts):
            model.randomize()
            try:
                if num_restarts == 1:
                    model.optimize(optimizer=optimizer, max_iters=iterations, messages=False)  # scg, bfgs, tnc
                else:
                    # Restarting is important because the initial hyperparameter guesses are often bad
                    parallel = (1 < num_processes)
                    model.optimize_restarts(optimizer=optimizer, num_restarts=num_restarts, parallel=parallel,
                                            num_processes=num_processes, max_iters=iterations, verbose=self.verbose)
                break
            except np.linalg.LinAlgError as e:
                print('Retraining attempt', attempt)
                if attempt == num_attempts - 1:
                    raise e
        return model

    def retrain(self, newx=None, newy=None, new_w=None, **kwargs):
        """
        Train the GP on all the training data again.
        """
        self.add_data(newx, newy, new_w)
        xx, yy = self.xx, self.yy[:, None]

        mean_fn, hyperparameters = self.mean_fn, self.hyperparameters
        normal_indices = np.argwhere(self.weights == NORMAL_WEIGHT).flatten()
        transfer_indices = np.argwhere(self.weights == TRANSFER_WEIGHT).flatten()
        print('Total: {} | Normal: {} | Transfer: {} | Other: {}'.format(
            len(self.weights), len(normal_indices), len(transfer_indices),
            len(self.weights) - len(normal_indices) - len(transfer_indices)))

        if (self.transfer_weight != 0) and (len(transfer_indices) != 0) and (self.sim_model is None):
            # TODO: assumes that transfer_indices remains constant
            self.sim_model = self.train_model(xx[transfer_indices], yy[transfer_indices], **kwargs)

        if self.sim_model is not None:
            if self.transfer_weight in [1, 3]: # More important than hyperparameters
                mean_fn = lambda X: self.sim_model.predict(X, include_likelihood=True)[0]
                #def mean_fn(X):
                #    mu, var = self.sim_model.predict(X, include_likelihood=True)
                #    return mu
                #sim_yy, _ = self.sim_model.predict(xx, include_likelihood=True) # Equivalent to mean_fn
                #yy = yy - sim_yy
            if self.transfer_weight in [2, 3]: # Treated as an enum
                hyperparameters = get_parameters(self.sim_model)

        # TODO: could add a small part of the data here (such as 0,0)
        self.model = self.train_model(xx[normal_indices], yy[normal_indices],
                                      mean_fn=mean_fn, hyperparameters=hyperparameters, **kwargs)

        self.reset_sample()
        self.trained = True
        #self.loo_train(num_restarts=num_restarts)
        if self.verbose:
            self.dump_model()

    def dump_parameters(self):
        print('Parameters:', get_parameters(self.model))
        # TODO: could split into context vs parameter
        print('Significant (most to least): {{{}}}'.format(', '.join(
            '{}: {:.3f}'.format(n, s) for s, n in sorted(zip(self.lengthscale, self.func.inputs)))))
        print('Magnitudes:')
        min_exp, max_exp = map(math.floor, map(math.log10, self.kern_var_range))
        bins = {}
        for n, s in zip(self.func.inputs, self.lengthscale):
            for exp in np.arange(min_exp, max_exp+1):
                if s < math.pow(10, exp+1):
                    bins.setdefault(exp, []).append(n)
                    break
        for exp in sorted(bins):
            print('{}: {}'.format(math.pow(10, exp), sorted(bins[exp])))

    def dump_model(self):
        print(self.model)
        print(self.ard_parameters)

        #print('Significant dimensions:', self.model.get_most_significant_input_dimensions())
        # self.variance*np.ones(self.input_dim)/self.lengthscale**2
        #print('Input sensitivity:', self.model.input_sensitivity().tolist())
        #print(self.model.likelihood)
        #print(self.model.posterior.K_chol) # woodbury_chol
        yy = self.yy[:, None]
        print('Prior:', self.model.log_prior())
        print('Train objective:', self.model.objective_function())
        print('Train log likelihood:', self.model.log_likelihood() / len(yy))
        print('Train log predictive density:',
              np.average(self.model.log_predictive_density(self.xx, yy)))
        print('Train sampled log predictive density:',
              np.average(self.model.log_predictive_density_sampling(self.xx, yy, num_samples=1000)))
        # TODO: account for differences in data sizes

        loo_likelihoods = self.model.inference_method.LOO(
            self.model.kern, self.xx, yy, self.model.likelihood, self.model.posterior) # marginal LOO
        # https://arxiv.org/abs/1412.7461
        #print(loo_likelihoods.tolist())
        print('LOO log likelihood:', np.average(loo_likelihoods))
        #mean, covar = self.model.predict(self.xx, full_cov=True, include_likelihood=True)
        #print(mean.shape, covar.shape) # full_cov returns full multivariate Gaussian (n times n) rather than just one
        #print(self.model.likelihood.log_predictive_density(yy, mean, var).tolist())
        #print(scipy.stats.multivariate_normal.logpdf(yy.flatten(), mean.flatten(), covar)) # Predictive
        self.dump_parameters()

        #loo = LeaveOneOut()
        #likelihoods = []
        #for train_index, test_index in loo.split(self.xx, yy):
        #    self.model.set_XY(self.xx[train_index], yy[train_index])
        #    likelihoods.append(self.model.log_predictive_density(self.xx[test_index], yy[test_index]).flatten())
        #print(np.array(likelihoods).tolist())

        #kernel = self.model.kern.K(self.xx, self.xx) + self.model.likelihood.variance*np.eye(len(self.xx))
        #print(scipy.stats.multivariate_normal.logpdf(yy.flatten(), np.zeros(len(yy)), kernel)) # Evidence

    def loo_train(self, num_restarts=10):

        def loss_fn(parameters):
            self.model[:] = parameters
            # TODO: this seems to just ignore its inputs...
            log_likelihood = self.model.inference_method.LOO(self.model.kern, self.xx, self.yy,
                                                             self.model.likelihood, self.model.posterior)
            # Maximize likelihood, maximize log likelihood, minimize neg log likelihood
            return -np.average(log_likelihood)

        best_parameters = self.model[:].copy() # self.model.param_array
        best_likelihood = loss_fn(best_parameters)
        print('Initial: {:.3f}'.format(best_likelihood))
        for i in range(num_restarts):
            #self.model.unfix() # unconstrain_fixed
            self.model.randomize()
            #self.model.fix() # constrain_fixed
            #for p in self.model.parameters:
            #   print(p)
            x0 = self.model[:].copy()
            bounds = list(zip(1e-4*np.ones(x0.shape), 1e4*np.ones(x0.shape))) # TODO: use bounds from above
            #bounds = None
            result = scipy.optimize.minimize(fun=loss_fn, x0=x0, bounds=bounds, method='L-BFGS-B', callback=None)
            x_star, y_star = result.x, result.fun
            #x_star, y_star = x0, loss_fn(x0)
            print('Iteration: {} | Initial: {:.3f} | Final: {:.3f} | Score: {:.3f}'
                  .format(i, loss_fn(x0), loss_fn(x_star), y_star)) # x0.tolist(), x_star.tolist()
            if y_star < best_likelihood:
                best_likelihood = y_star
                best_parameters = x_star.copy()
            #self.model.optimize(optimizer=optimizer, max_iters=iterations, messages=False) # No need to call
            #likelihood = -self.model.inference_method.LOO(self.model.kern, self.xx, self.yy,
            #                                              self.model.likelihood, self.model.posterior) # negative marginal LOO
            # average_likelihood = np.average(likelihood)
            #if best_likelihood < average_likelihood:
            #    best_likelihood = average_likelihood
            #    best_parameters = self.model[:].tolist()
            #print('LOO likelihood:', average_likelihood)
        print('Best LOO likelihood:', -best_likelihood)
        print('Best parameters:', best_parameters.tolist())
        self.model[:] = best_parameters

        # TODO: metadata flag
        # TODO: incorporate a prior so unsampled parts of the space are avoided
        # posterior_covariance_between_points
