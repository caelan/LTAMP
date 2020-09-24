# Author: Zi Wang

from __future__ import print_function

import os
import pickle
import shutil

import numpy as np
import scipy
import scipy.optimize

from functools import partial
from scipy.stats import truncnorm

try:
    user_input = raw_input
except NameError:
    user_input = input

def safe_rmdir(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)

class WorkingDirSaver(object):
    def __init__(self, tmp_working_dir, remove_tmp_dir=False):
        # TODO: this changes the abspath of __file__ for python2
        self.working_dir = os.getcwd()
        self.tmp_working_dir = tmp_working_dir
        self.remove_tmp_dir = remove_tmp_dir
        if not os.path.isdir(self.tmp_working_dir):
            os.makedirs(self.tmp_working_dir)
        elif self.remove_tmp_dir:
            raise ValueError('Working directory already exists. Other processes may be using it now.')

    def __enter__(self):
        os.chdir(self.tmp_working_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.working_dir)
        if self.remove_tmp_dir:
            safe_rmdir(self.tmp_working_dir)


def is_close(a, b):
    return np.abs(a - b) < 1e-3


def get_xx_yy(expid, method, exp='pour'):
    '''
    Returns the training data {xx, yy} and the context c of an experiment.
    Args:
        expid: experiment ID.
        method: training method (e.g. 'gp_lse', 'nn_classification', 'nn_regression', 'random').
        exp: experimented action (e.g. 'scoop', 'pour', 'push').
    '''
    dirnm = 'data/'
    fnm_prefix = os.path.join(dirnm, exp)
    initx, inity = pickle.load(open('{}_init_data_{}.pk'.format(fnm_prefix, expid)))
    fnm = '{}_{}_{}.pk'.format(fnm_prefix, method, expid)
    xx, yy, c = pickle.load(open(fnm, 'rb'))
    xx = np.vstack((initx, xx))
    yy = np.hstack((inity, yy))
    return xx, yy, c


def diversity(xx, active_dim):
    '''
    Returns the diversity of the list xx, with active dimensions active_dim.
    Diversity is measured by log |K/0.01 + I|, where K is the squared 
    exponential gram matrix on xx, with length scale 1.
    '''
    n = len(xx)
    xx = xx[:, active_dim]
    l = np.ones(xx.shape[1])
    K = se_kernel(xx, xx, l)
    return np.log(scipy.linalg.det(K / 0.01 + np.eye(n)))


def sample_tgmm(center, scale, n, xmin, xmax, epsilon=1e-4):
    '''
    Sample from a truncated Gaussian mixture model (TGMM).
    Returns the samples and their importance weight.
    Args:
        center: center of TGMM.
        scale: scale of TGMM.
        n: number of samples.
        xmin: smallest values of the truncated interval. 
        xmax: largest values of the truncated interval. 
    '''
    dx = len(xmin)
    slen = len(center)
    rd_centers = np.random.choice(slen, (n))
    ta = (xmin - center[rd_centers]) / scale
    tb = (xmax - center[rd_centers]) / scale
    x_samples_gmm = truncnorm.rvs(ta, tb, loc=center[rd_centers], scale=scale)

    ta = (xmin - center) / scale
    tb = (xmax - center) / scale

    def truncpdf(j, i):
        return truncnorm.pdf(x_samples_gmm[:, j], ta[i][j], tb[i][j], center[i][j], scale[j])

    # TypeError: unsupported operand type(s) for +: 'map' and 'map'
    prob = [np.prod(list(map(partial(truncpdf, i=i), range(dx))), axis=0) for i in range(slen)]
    prob = np.sum(prob, axis=0) / slen
    np.clip(prob, epsilon, 1. / epsilon)
    return x_samples_gmm, 1. / prob


def grid_around_point(p, grange, n, x_range):
    '''
    Returns a list of the points on the grid around point p.
    Args:
        p: the point around which the grid is generated
        grange: a dx vector, each element denotes half length of the grid on dimension d
        n: the number of points on each dimension
    '''
    dx = len(p)
    if not hasattr(n, "__len__"):
        n = [n] * dx
    xmin = [max(p[d] - grange[d], x_range[0, d]) for d in range(dx)]
    xmax = [min(p[d] + grange[d], x_range[1, d]) for d in range(dx)]
    mg = np.meshgrid(*[np.linspace(xmin[d], xmax[d], n[d]) for d in range(dx)])
    grids = list(map(np.ravel, mg))
    return np.array(grids).T


def grid(n, x_range):
    '''
    p is the point around which the grid is generated
    grange is a dx vector, each element denotes half length of the grid on dimension d
    n is the number of points on each dimension
    '''
    dx = x_range.shape[1]
    if not hasattr(n, "__len__"):
        n = [n] * dx
    xmin, xmax = x_range
    mg = np.meshgrid(*[np.linspace(xmin[d], xmax[d], n[d]) for d in range(dx)])
    grids = list(map(np.ravel, mg))
    return np.array(grids).T


def global_minimize(f, fg, x_range, n=1, guesses=None, iterations=100, **kwargs):
    """
    :param f: function
    :param fg: function and gradients
    :param x_range:
    :param n:
    :param guesses:
    :param iterations:
    :return:
    """
    dx = x_range.shape[1]
    tx = np.random.uniform(x_range[0], x_range[1], (n, dx))
    if guesses is not None:
        tx = np.vstack((tx, guesses))
    ty = f(tx)
    #ty = np.array([f(x) for x in tx])
    index = ty.argmin()
    x0 = tx[index]  # 2d array 1*dx
    #print([np.min(ty[:10**i]) for i in range(1, 6)])
    if iterations is None:
        x_star, y_star = x0, float(ty[index])
    elif fg is None:
        fg = lambda *args, **kw: np.array(f(*args, **kw)).flatten()[0]
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        #scipy.optimize.show_options()
        res = scipy.optimize.minimize(fun=fg, x0=x0, bounds=x_range.T, method='L-BFGS-B', callback=None, options={}, **kwargs)
        x_star, y_star = res.x, res.fun
    else:
        # TODO: problems with BLAS when using multiprocessing
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
        x_star, y_star, _ = scipy.optimize.fmin_l_bfgs_b(
            func=fg, x0=x0, bounds=x_range.T, maxiter=iterations, callback=None, **kwargs)
    return x_star, y_star


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def l2_squared(X, X2, l):
    '''
    l2 distance between each pair of items from X, X2, rescaled by l.
    '''
    if X.ndim == 1:
        X = X[None, :]
    if X2.ndim == 1:
        X2 = X2[None, :]
    X = X * l
    X2 = X2 * l
    X1sq = np.sum(np.square(X), 1)
    X2sq = np.sum(np.square(X2), 1)
    r2 = -2. * np.dot(X, X2.T) + X1sq[:, None] + X2sq[None, :]
    r2 = np.clip(r2, 0, np.inf)
    return r2


def argmax_min_dist(X, X2, l=None):
    if l is None:
        l = np.ones(X.shape[1])
    r2 = l2_squared(X, X2, l)
    r2 = r2.min(axis=1)
    return r2.argmax()


def se_kernel(X, X2, l):
    '''
    Squared exponential kernel, with inverse lengthscale l.
    '''
    dist = l2_squared(X, X2, l)
    return np.exp(-0.5 * dist)


def argmax_condvar(X, X2, l=None):
    '''
    Returns the argmax of conditional variance on X2. The candidates are X.
    l is the inverse length scale of a squared exponential kenel.
    '''
    if l is None:
        l = np.ones(X.shape[1])
    kxx2 = se_kernel(X, X2, l)
    kx2x2 = se_kernel(X2, X2, l) + 0.01*np.eye(X2.shape[0])
    factor = scipy.linalg.cholesky(kx2x2)
    negvar = (kxx2 * scipy.linalg.cho_solve((factor, False), kxx2.T).T).sum(axis=1, keepdims=1)
    return negvar.argmin()


def important_d(s, X, l):
    '''
    Returns the most important dimension given that the last sample is s and the samples before
    s is X. l is the inverse length scale of a squared exponential kenel.
    '''
    dx = X.shape[1]
    importance = np.zeros(dx)
    kxx = se_kernel(X, X, l) + 0.01*np.eye(X.shape[0])
    factor = scipy.linalg.cholesky(kxx)
    for d in range(dx):
        l2 = np.zeros(l.shape)
        l2[d] = l[d]
        ksx = se_kernel(s, X, l2)
        importance[d] = ksx.dot(scipy.linalg.cho_solve((factor, False), ksx.T))
    return importance.argmin()


def regression_acc(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    label = (0 < y_true)
    pred = (0 < y_pred)
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    acc = (tn + tp) * 1.0 / len(label)
    fpr = fp * 1.0 / (tn + fp)
    fnr = fn * 1.0 / (tp + fn)
    return acc, fpr, fnr


def gen_data(func, N, parallel=False):
    '''
    Generate N data points on function func.
    Use multiprocessing if parallel is True; otherwise False.
    '''
    X = np.random.uniform(
        func.x_range[0], func.x_range[1], (N, func.x_range.shape[1]))
    if parallel:
        from multiprocessing import Pool
        import multiprocessing
        cpu_n = multiprocessing.cpu_count()
        p = Pool(cpu_n)
        y = np.array(list(p.map(func, X)))
    else:
        y = np.array(list(map(func, X)))
    return X, y


def gen_context(func, N=1):
    '''
    Generate N random contexts associated with function func. 
    '''
    xmin = func.x_range[0, func.context_idx]
    xmax = func.x_range[1, func.context_idx]
    if N == 1:
        return np.random.uniform(xmin, xmax)
    return np.random.uniform(xmin, xmax, (N, len(func.context_idx)))


def find_closest_positive_context_param(context, xx, yy, param_idx, context_idx, weights=None):
    '''
    Find the closest data point (in terms of context distance) that has a positive label.
    Args:
        context: current context
        xx: training inputs
        yy: training outpus
        param_idx: index of parameters in an input
        context_idx: index of contexts in an input
        weights:
    '''
    if yy.ndim == 2:
        yy = np.squeeze(yy)
    if weights is None:
        weights = np.ones(context.shape)
    else:
        weights = weights[context_idx]
    positive_idx = (0 < yy)
    if np.sum(positive_idx) == 0:
        idx = 0
    else:
        xx = xx[positive_idx] #yy = yy[positive_idx]
        differences = xx[:, context_idx] - context
        distances = np.dot(differences*differences, weights)
        idx = distances.argmin()
    return xx[idx, param_idx], xx[idx, context_idx]


def gen_biased_data(func, pos_ratio, N):
    '''
    Generate N data points on function func, with pos_ratio percentage of the 
    data points to have a positive label.
    '''
    from sklearn.utils import shuffle
    pos = []
    neg = []
    while (len(pos) < pos_ratio * N) or (len(neg) < N - pos_ratio * N):
        x = np.random.uniform(func.x_range[0], func.x_range[1])
        y = func(x)
        if y > 0:
            if len(pos) < (pos_ratio * N):
                pos.append(np.hstack((x, y)))
        elif len(neg) < (N - pos_ratio * N):
            neg.append(np.hstack((x, y)))
    xy = np.vstack((pos, neg))
    xy = shuffle(xy)
    return xy[:, :-1], xy[:, -1]


def run_ActiveLearner(active_learner, context, iters, save_fnm=None):
    '''
    Actively query a function with active learner.
    Args:
        active_learner: an ActiveLearner object.
        context: the current context we are testing for the function.
        iters: total number of queries.
        save_fnm: a file name string to save the queries.
    '''
    # Retrieve the function associated with active_learner
    func = active_learner.func
    # Queried x and y
    xq, yq = None, None
    # All the queries x and y
    xx = np.zeros((0, func.x_range.shape[1]))
    yy = np.zeros(0)
    # Start active queries
    for i in range(iters):
        active_learner.retrain(xq, yq)
        xq = active_learner.query(context)
        yq = func(xq)
        xx = np.vstack((xx, xq))
        yy = np.hstack((yy, yq))
        print('i={}, xq={}, yq={}'.format(i, xq, yq))
        if save_fnm is not None:
            data = (xx, yy, context)
            with open(save_fnm, 'wb') as f:
                pickle.dump(data, f)
    return xx, yy