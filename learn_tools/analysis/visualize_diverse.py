from learn_tools.collect_simulation import start_task
import argparse
import os

from pybullet_tools.utils import read_pickle
from pddlstream.utils import get_python_version
from learn_tools.retired.run_sample import evaluate_samples, sample_task_with_seed
from learn_tools.learnable_skill import load_data
from learn_tools.uncertain_learner import UNIFORM


def main(skill='pour'):
    parser = argparse.ArgumentParser()
    parser.add_argument('expid', type=int, help='experiment ID')
    parser.add_argument('beta_lambda', type=float, default=0.99, help='lambda parameter for computing beta from best beta')
    args = parser.parse_args()
    beta_lambda = args.beta_lambda
    expid = args.expid
    n_samples = 20
    trainsize = 1000
    exp_dir = 'sampling_trainsize={}_samples={}_beta_lambdda={}'.format(trainsize, n_samples, beta_lambda)
    if skill is 'pour':
        exp_dir = os.path.join('data/pour_19-06-13_00-59-21/', exp_dir)
        domain = load_data(['data/pour_19-06-13_00-59-21/trials_n=10000.json'])
    elif skill is 'scoop':
        exp_dir = os.path.join('data/scoop_19-06-10_20-16-59_top-diameter/', exp_dir)
        domain = load_data(['data/scoop_19-06-10_20-16-59_top-diameter/trials_n=10000.json'])

    data_path = os.path.join(exp_dir, 'experiments_{}.pk{}'.format(expid, get_python_version()))
    if not os.path.exists(data_path):
        print('{} does not exist'.format(data_path))
        return
    data, seed = read_pickle(data_path)

    print('sample a new task')
    current_wd, trial_wd = start_task()
    sim_world, collector, task, feature, evalfunc, saver = sample_task_with_seed(skill, seed=seed,
                                                                                 visualize=True)

    samples = data[UNIFORM].samples
    scores = data[UNIFORM].scores
    good_samples = samples[scores > 0]

    scores, plan_results = evaluate_samples(sim_world, collector, task, feature, domain, good_samples[:5],
                                            evalfunc, saver)


if __name__ == '__main__':
    main()