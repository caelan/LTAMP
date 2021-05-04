import argparse
import os
import time
from collections import defaultdict
from collections import namedtuple

import numpy as np

from learn_tools.active_gp import ActiveGP, POUR_MLP_HYPERPARAM_3000, SCOOP_MLP_HYPERPARAM_3000
from learn_tools.collect_simulation import start_task, complete_task, sample_task, get_parameter_result
from learn_tools.helper import diversity
from learn_tools.learnable_skill import load_data
from learn_tools.learner import FEATURE, PARAMETER, SCORE, FAILURE
from learn_tools.learner import get_trial_parameter_fn
from learn_tools.uncertain_learner import DIVERSELK, SAMPLE_STRATEGIES
from pddlstream.utils import get_python_version, mkdir, elapsed_time
from plan_tools.common import set_seed
from pybullet_tools.utils import write_pickle, read_pickle

Result = namedtuple('Result', ['samples', 'sample_time', 'diversity', 'diversity_5', 'num_samples_5',
                               'scores', 'plan_results', 'precision', 'context', 'n_samples', 'beta',
                               'best_beta'])


def sample_task_with_seed(problem, seed=0, **kwargs):
    set_seed(seed)
    return sample_task(problem, collect_kwargs={}, **kwargs)


def evaluate_samples(sim_world, collector, task, feature, domain, samples, evalfunc, saver):
    scores = []
    plan_results = []
    for sample in samples:
        parameter = domain.parameter_from_sample(sample[domain.param_idx])
        parameter_fn = get_trial_parameter_fn(parameter)
        parameter_fns = {collector.gen_fn: parameter_fn}
        saver.restore()
        # TODO(caelan): update given new function parameterization
        result = get_parameter_result(sim_world, task, parameter_fns, evalfunc, max_time=150)
        if result is None or result[SCORE] is None:
            score = FAILURE
        else:
            score = domain.score_fn(feature, None, result[SCORE])
        # if score > 0:
        #    import pdb; pdb.set_trace()
        scores.append(score)
        plan_results.append(result)
    scores = np.array(scores)
    return scores, plan_results


def get_sample_result(sample_strategy, learner, context, sim_world, collector, task, feature, evalfunc, saver,
                      n_samples):
    learner.sample_strategy = sample_strategy
    learner.reset_sample()
    learner.set_world_feature(sim_world, feature)
    start_time = time.time()
    for i in range(n_samples):
        learner.sample(context)
    sample_time = elapsed_time(start_time)
    div = diversity(learner.sampled_xx, learner.func.param_idx)

    print('len samples = {}   |   diversity = {}'.format(len(learner.sampled_xx), div))

    scores, plan_results = evaluate_samples(sim_world, collector, task, feature, learner.func, learner.sampled_xx,
                                            evalfunc, saver)
    good_samples = learner.sampled_xx[scores > 0]
    if good_samples.shape[0] >= 5:
        div_5 = diversity(good_samples[:5], learner.func.param_idx)
        num_samples_5 = 0  # find number of samples to get first good 5 samples
        cnt = 0
        for s in scores:
            if s > 0:
                cnt += 1
            num_samples_5 += 1
            if cnt >= 5:
                break
    else:
        div_5 = None
        num_samples_5 = None
    precision = sum(scores > 0) / len(scores)
    return Result(learner.sampled_xx, sample_time, div, div_5, num_samples_5, scores, plan_results, precision,
                  context, n_samples, learner.beta, learner.best_beta)


def save_experiments(experiments_dir, expid, experiments):
    if experiments_dir is None:
        return None
    mkdir(experiments_dir)
    data_path = os.path.join(experiments_dir, 'experiments_{}.pk{}'.format(expid, get_python_version()))
    write_pickle(data_path, experiments)
    print('Saved', data_path)
    return data_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trainsize', default=2000, type=int, help='training set size')
    parser.add_argument('expid', default=0, type=int, help='experiment ID')
    parser.add_argument('paths', default=[os.path.join(get_data_dir('pour'), 'trials_n=10000.json')], nargs='*',
                        help='Paths to the data.')
    parser.add_argument('beta_lambda', default=0.99, type=float,
                        help='lambda parameter for computing beta from best beta')
    parser.add_argument('-n', '--include_none', action='store_true',
                        help='include none in the dataset')
    args = parser.parse_args()
    beta_lambda = args.beta_lambda
    n_samples = 20
    seed = hash((os.getpid(), args.expid))
    set_seed(seed)
    print('loading data')
    domain = load_data(args.paths)
    data = domain.create_dataset(include_none=args.include_none, binary=False)
    data.shuffle()
    X, Y, W = data.get_data()

    print('finished obtaining x y data')
    n_train = args.trainsize  # len(X)
    X = X[:n_train]
    Y = Y[:n_train]
    print('initializing ActiveGP')
    if 'pour' in args.paths[0]:
        hype = POUR_MLP_HYPERPARAM_3000
    elif 'scoop' in args.paths[0]:
        hype = SCOOP_MLP_HYPERPARAM_3000

    learner = ActiveGP(domain, initx=X, inity=Y, hyperparameters=hype, sample_time_limit=60, beta_lambda=beta_lambda)
    learner.retrain(num_restarts=10)

    print('sample a new task')
    current_wd, trial_wd = start_task()
    sim_world, collector, task, feature, evalfunc, saver = sample_task_with_seed(domain.skill, seed=seed,
                                                                                 visualize=False)
    context = domain.context_from_feature(feature)

    # date_name = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    exp_dir = 'sampling_trainsize={}_samples={}_beta_lambdda={}_includenone_{}'.format(len(X),
                                                                                       n_samples,
                                                                                       beta_lambda,
                                                                                       int(args.include_none))
    exp_dir = os.path.join(os.path.dirname(args.paths[0]), exp_dir)
    results = {}
    SAMPLE_STRATEGIES.remove(DIVERSELK)
    for sample_strategy in SAMPLE_STRATEGIES:
        results[sample_strategy] = get_sample_result(sample_strategy, learner, context, sim_world, collector, task,
                                                     feature, evalfunc, saver, n_samples)
    complete_task(sim_world, current_wd, trial_wd)

    save_experiments(exp_dir, args.expid, (results, seed))
    for ss in results:
        print(ss, results[ss].precision, results[ss].diversity, results[ss].diversity_5, results[ss].sample_time)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`
    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    x = np.array(x)
    y = np.array(y)
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot(trainsize, n_samples, beta_lambda, includenone=1, ignore_plan_fail=True, show=False, skill='pour'):
    exp_dir = 'sampling_trainsize={}_samples={}_beta_lambdda={}_includenone_{}'.format(trainsize, n_samples,
                                                                                       beta_lambda, includenone)
    if skill is 'pour':
        exp_dir = os.path.join(get_data_dir('pour'), exp_dir)
    elif skill is 'scoop':
        exp_dir = os.path.join(get_data_dir('scoop'), exp_dir)
    acc = defaultdict(list)
    div5 = defaultdict(list)
    div = defaultdict(list)
    sample_time = defaultdict(list)
    num_samples_5 = defaultdict(list)
    npr = defaultdict(list)
    for expid in range(50):
        data_path = os.path.join(exp_dir, 'experiments_{}.pk{}'.format(expid, get_python_version()))
        if not os.path.exists(data_path):
            print('{} does not exist'.format(data_path))
            continue
        # print('processing {}'.format(data_path))
        data, seed = read_pickle(data_path)
        for strategy in data:
            if data[strategy].diversity_5 is None:
                continue
            none_res = np.sum([1 if pr['score'] is None else 0 for pr in data[strategy].plan_results])
            if data[strategy].precision < 1.:
                print('low precision - score 0:{}, <0:{}, >0:{}'.format(none_res,
                                                                        np.sum(data[strategy].scores < 0),
                                                                        np.sum(data[strategy].scores > 0)))

            if ignore_plan_fail:
                acc[strategy].append( (np.sum(data[strategy].scores > 0) + none_res) / len(data[strategy].scores))
            else:
                acc[strategy].append(data[strategy].precision)
            npr[strategy].append(none_res / len(data[strategy].scores))
            div5[strategy].append(data[strategy].diversity_5)
            div[strategy].append(data[strategy].diversity)
            sample_time[strategy].append(data[strategy].sample_time)
            num_samples_5[strategy].append(data[strategy].num_samples_5)

    sample_time_plot = []
    sample_time_err = []
    strategies = []
    num_samples_5_plot = []
    num_samples_5_err = []

    for strategy in sample_time:
        acc[strategy] = np.array(acc[strategy])
        print(
            '{}     fpr = {:.2f} \pm {:.2f}     npr = {:.2f} \pm {:.2f}     time = {:.2f} \pm {:.2f}     n5 = {:.2f} \pm {:.2f}     div5 = {:.2f} \pm {:.2f}'.format(
                strategy,
                np.mean(1 - acc[strategy]),
                np.std(1 - acc[strategy]),
                np.mean(npr[strategy]),
                np.std(npr[strategy]),
                np.mean(sample_time[strategy]),
                np.std(sample_time[strategy]),
                np.mean(num_samples_5[strategy]),
                np.std(num_samples_5[strategy]),
                np.mean(div5[strategy]),
                np.std(div5[strategy])
                ))
        sample_time_plot.append(np.mean(sample_time[strategy]))
        sample_time_err = np.std(sample_time[strategy])
        num_samples_5_plot.append(np.mean(num_samples_5[strategy]))
        num_samples_5_err.append(np.std(num_samples_5[strategy]))
        strategies.append(strategy)

    exp_dir = os.path.join(exp_dir, 'figures')
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.size': 20, 'legend.fontsize': 15})
    markers = ['o', 'v', '<', '>', '1', '2', '3']
    colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#cab2d6', '#fb9a99']

    # bar plot for sample time
    return
    fig, ax = plt.subplots()
    ax.bar(range(3), sample_time_plot, yerr=sample_time_err, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Sample time (seconds)')
    ax.set_xticks(range(3))
    ax.set_xticklabels(strategies)
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'sampletime.pdf'))
    if show:
        plt.show()
    else:
        plt.clf()

    # bar plot for num sample 5
    fig, ax = plt.subplots()
    ax.bar(range(3), num_samples_5_plot, yerr=num_samples_5_err, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Number of samples needed to generate 5 good ones')
    ax.set_xticks(range(3))
    ax.set_xticklabels(strategies)
    ax.yaxis.grid(True)

    # Save the figure and show
    # plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'numsample5.pdf'))
    if show:
        plt.show()
    else:
        plt.clf()

    # div-acc using all data points
    i = 0
    for strategy in acc:
        # plt.scatter(np.mean(div5[strategy]), np.mean(acc[strategy]), marker=markers[i], color=colors[i], label=strategy)
        plt.scatter(div5[strategy], acc[strategy], marker=markers[i], color=colors[i], label=strategy)
        plt.xlabel('Diversity of the first 5 samples')
        plt.ylabel('Accuracy ')
        i += 1
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'all_data_div5_ignorePlanFail_{}.pdf'.format(ignore_plan_fail)))
    if show:
        plt.show()
    else:
        plt.clf()

    i = 0
    for strategy in acc:
        # plt.scatter(np.mean(div[strategy]), np.mean(acc[strategy]), marker=markers[i], color=colors[i], label=strategy)
        plt.scatter(div[strategy], acc[strategy], marker=markers[i], color=colors[i], label=strategy)
        plt.xlabel('Diversity of all 20 samples')
        plt.ylabel('Accuracy ')
        i += 1
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'all_data_div_ignorePlanFail_{}.pdf'.format(ignore_plan_fail)))
    if show:
        plt.show()
    else:
        plt.clf()

    # plot the average
    i = 0
    fig, ax = plt.subplots()
    for strategy in acc:
        confidence_ellipse(div5[strategy], acc[strategy], ax,
                           alpha=0.5, facecolor=colors[i], edgecolor=colors[i], zorder=0)
        plt.scatter(np.mean(div5[strategy]), np.mean(acc[strategy]), marker='.', s=10, color=colors[i], label=strategy)
        plt.xlabel('Diversity of the first 5 samples')
        plt.ylabel('Accuracy ')
        i += 1
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'mean_div5_ignorePlanFail_{}.pdf'.format(ignore_plan_fail)))
    if show:
        plt.show()
    else:
        plt.clf()

    i = 0
    fig, ax = plt.subplots()
    for strategy in acc:
        confidence_ellipse(div[strategy], acc[strategy], ax,
                           alpha=0.5, facecolor=colors[i], edgecolor=colors[i], zorder=0)
        plt.scatter(np.mean(div[strategy]), np.mean(acc[strategy]), marker='.', s=10, color=colors[i], label=strategy)
        plt.xlabel('Diversity of all 20 samples')
        plt.ylabel('Accuracy ')
        i += 1
    plt.legend()
    plt.savefig(os.path.join(exp_dir, 'mean_div_ignorePlanFail_{}.pdf'.format(ignore_plan_fail)))
    if show:
        plt.show()
    else:
        plt.clf()


def get_data_dir(skill):
    if skill == 'pour':
        # old path: skill_dir = 'data/pour_19-06-13_00-59-21/'
        # new path
        # skill_dir = 'data/pour_19-11-22_13-08-55'
        skill_dir = 'data/pour_19-12-09_22-16-07/'
    elif skill == 'scoop':
        skill_dir = 'data/scoop_19-12-07_16-02-40/'
    return skill_dir


def gen_exp_script(trainsize, n_samples, beta_lambda, nexp, nparallel=15, skill='pour'):
    with open('run_sample_exp.sh', 'w') as f:
        for beta_lambda in [.99, .98]:
            for include_none in [1]:
                exp_dir = 'sampling_trainsize={}_samples={}_beta_lambdda={}_includenone_{}'.format(trainsize,
                                                                                               n_samples,
                                                                                               beta_lambda,
                                                                                               int(include_none))
                skill_dir = get_data_dir(skill)
                exp_dir = os.path.join(skill_dir, exp_dir)
                lines = []
                for i in range(nexp):
                    expfile = os.path.join(exp_dir, 'experiments_{}.pk{}'.format(i, get_python_version()))
                    outfile = exp_dir + '_experiments_{}.pk{}'.format(i, get_python_version())
                    if os.path.exists(expfile):
                        continue
                    line = 'python3 -m learn_tools.run_sample {} {} {}trials_n=10000.json {} '.format(
                        trainsize, i, skill_dir, beta_lambda)
                    if include_none:
                        line += '-n '

                    line += '> {}.out &\n'.format(outfile)
                    lines.append(line)

                for i in range(nparallel - 1, len(lines), nparallel):
                    lines[i] = lines[i].replace('&', '')
                f.writelines(lines)


if __name__ == '__main__':
    main()
    # plot(3000, 20, 0.98, includenone=1, ignore_plan_fail=False, show=False, skill='scoop')
    # gen_exp_script(3000, 20, 0.98, 50, skill='scoop', nparallel=15)
    # scp -r ziw@shakey.csail.mit.edu:ltamp-pr2/data/pour_19-06-13_00-59-21/sampling* data/pour_19-06-13_00-59-21/
