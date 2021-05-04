from __future__ import print_function, division

import argparse
import os
import sys

sys.path.extend([
    os.path.join(os.getcwd(), 'pddlstream'),
    os.path.join(os.getcwd(), 'ss-pybullet'),
])

from pybullet_tools.utils import read_pickle, clip
from learn_tools.run_active import evaluate_confusions, create_learner, SCOOP_TEST_DATASETS \
    , BEST_DATASETS, TRAIN_DATASETS, TEST_DATASETS
from learn_tools.learner import TRAINING, SCORE, FEATURE, PARAMETER, BAD, REAL_PREFIX
from learn_tools.learnable_skill import load_data, read_data
from learn_tools.analyze_experiment import plot_experiments, Experiment, Algorithm
from learn_tools.active_gp import BATCH_GP
from learn_tools.collect_simulation import SKILL_COLLECTORS
from learn_tools.collectors.collect_pour import compute_fraction_filled
from learn_tools.collectors.collect_scoop import scoop_fraction_filled
from learn_tools.statistics import is_successful
from learn_tools.active_nn import NN_MODELS
from learn_tools.active_rf import RF_REGRESSOR

POUR_LEARNERS = [
    #'pour_continuous_real=80.pk2',
    #'pour_sim=1000_continuous_real=80.pk2',
    #'pour_continuous_real=50.pk2',
    'pour_continuous_real=100.pk2',
    #'gp_batch_k=mlp_v=true_t=2_20-01-06_14-24-44.pk2', # Active feature
    #'gp_batch_k=mlp_v=true_t=2_20-01-06_17-25-04.pk2', # Active feature
    #'gp_batch_k=mlp_v=true_t=2_20-01-07_12-19-02.pk2', # Random feature
    #'gp_batch_k=mlp_v=true_t=2_20-01-07_13-39-08.pk2', # Random feature
    #'gp_batch_k=mlp_v=true_20-01-09_14-44-18.pk2', # Active feature
    'gp_batch_k=mlp_v=true_20-01-09_16-26-02.pk2', # Random feature
    #'gp_batch_k=mlp_v=true_20-01-10_12-35-40.pk2' # Random feature (short extension of last)
    #'gp_batch_k=mlp_v=true_t=3_20-01-10_13-20-11.pk2', # Random feature
    #'gp_batch_k=mlp_v=true_20-01-10_14-43-49.pk2', # Random feature, include_none=True
    #'gp_batch_k=mlp_v=true_20-01-10_16-17-15.pk2', # Active feature, include_none=True
]

SCOOP_LEARNERS = [
    #'scoop_real=50.pk2',
    'scoop_real=96.pk2',
    #'gp_batch_k=mlp_v=true_20-02-12_17-11-16.pk2', # orange_spoon
    #'gp_batch_k=mlp_v=true_20-02-13_15-48-24.pk2', # grey_spoon
    #'gp_batch_k=mlp_v=true_20-02-13_16-19-49.pk2', # green_spoon
    'gp_batch_k=mlp_v=true_20-02-13_16-50-25.pk2', # final
    #'gp_batch_k=mlp_v=true_20-02-17_16-25-33.pk2',
    #'gp_batch_k=mlp_v=true_20-02-17_16-36-49.pk2',
    #'gp_batch_k=mlp_v=true_20-02-17_16-48-40.pk2',
]

LABELS = {
    'pour_continuous_real=100.pk2': 'GP', # 'Batch GP',
    'gp_batch_k=mlp_v=true_20-01-09_14-44-18.pk2': 'GP-LSE',
    'gp_batch_k=mlp_v=true_20-01-09_16-26-02.pk2': 'GP-LSE2', # 'Active GP',
    'scoop_real=96.pk2': 'GP', # 'Batch GP',
    'gp_batch_k=mlp_v=true_20-02-13_16-50-25.pk2': 'GP-LSE2', # 'Active GP',
}

BINARY_LEARNERS = [
    'gp_batch_k=mlp_v=true_t=3_19-12-09_18-41-13.pk2',
    'gp_batch_k=mlp_v=true_t=3_19-12-16_15-48-15.pk2',
    'gp_batch_k=mlp_v=true_t=3_19-12-18_13-43-37.pk2',
]

BAD_LEARNERS = [
    'gp_batch_k=mlp_v=true_t=4_19-12-18_13-17-31.pk2',
]

##################################################

#ALPHAS = np.linspace(0.0, 0.9, num=5, endpoint=True)
ALPHAS = [0., 0.25, 0.5, 0.75, 0.9, 0.95]
#ALPHAS = [0.25]

#TRANSFERS = [0, 1, 2, 3]
TRANSFERS = [None]

MIN_TRAIN = 50
TRAIN_STEP = 5

BATCH_ALGORITHMS = []
# BATCH_ALGORITHMS += [Algorithm(nn_model) for nn_model in NN_MODELS]
# BATCH_ALGORITHMS += [Algorithm(RF_REGRESSOR, variance=True, transfer_weight=transfer)
#                      for transfer in TRANSFERS]  # transfer_weights
# BATCH_ALGORITHMS += [Algorithm(BATCH_GP, kernel='MLP', variance=True, transfer_weight=transfer)
#                      for transfer in TRANSFERS]

METRICS = [
    'rmse',
    #'accuracy',
    #'precision',
    #'recall',
    'f1',
]

##################################################

def test_learner(algorithm, learner, training_sizes, test_data, num_train=10):
    xx, yy, ww = learner.xx, learner.yy, learner.weights
    #all_indices = list(range(len(learner.xx)))
    # all_indices = batch_indices + active_indices
    # all_indices = randomize(all_indices)
    all_indices = [i for i, result in enumerate(learner.results)
                   if not result.get('annotation', '')]

    experiments = []
    for size in training_sizes:
        indices = all_indices[:size]
        learner.xx, learner.yy, learner.weights = xx[indices], yy[indices], ww[indices]
        for _ in range(num_train):
            learner.retrain()
            test_confusion = evaluate_confusions(test_data, learner, alphas=ALPHAS, serial=True,
                                                 header='Metrics:', verbose=False)
            confusion = {'test': test_confusion}
            experiments.append(Experiment(algorithm, size, confusion, []))
    evaluate_confusions(test_data, learner, alphas=ALPHAS, serial=True,
                        header='Metrics:', verbose=True)
    learner.xx, learner.yy, learner.weights = xx, yy, ww
    return experiments

##################################################

def evaluate_batch(training_sizes, test_data, num_trials=3):
    if not BATCH_ALGORITHMS:
        return []
    train_paths = TRAIN_DATASETS
    train_domain = load_data(train_paths, verbose=True)
    train_domain.name = 'Train Data'
    train_data = train_domain.create_dataset(include_none=False, binary=False, scale_paths=None)
    #training_sizes = list(range(5, len(train_data), 5))

    experiments = []
    for _ in range(num_trials):
        train_data.shuffle()
        selected_data, _ = train_data.partition(max(training_sizes))
        for algorithm in BATCH_ALGORITHMS:
            learner, _ = create_learner(train_domain, selected_data, algorithm, verbose=False)
            #learner_name = 'Batch GP'
            learner_name = algorithm.name
            algorithm = Algorithm(learner_name)
            #algorithm = Algorithm(learner_name, *learner.algorithm[1:])
            experiments.extend(test_learner(algorithm, learner, training_sizes, test_data))
    return experiments

##################################################

def main(pour=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('learners', nargs='*',
                        help='Path to the learner to be analyzed')
    #parser.add_argument('-d', '--dump', action='store_true',
    #                    help='When enabled, dumps trial statistics.')
    args = parser.parse_args()

    learner_paths = args.learners
    if not learner_paths:
        LEARNERS = POUR_LEARNERS if pour else SCOOP_LEARNERS
        learner_paths.extend(os.path.join('learners', learner_file) for learner_file in LEARNERS)

    test_paths = TEST_DATASETS if pour else SCOOP_TEST_DATASETS
    # test_paths = TRAIN_DATASETS + TEST_DATASETS
    test_domain = load_data(test_paths, verbose=False)
    #test_domain.name = 'Test Data'
    test_data = test_domain.create_dataset(include_none=False, binary=False, scale_paths=None)
    # TODO: test best per discrete context
    print('Test: {} (n={})'.format(test_domain.name, len(test_data)))

    max_train = 0
    experiments = []
    for learner_path in learner_paths:
        learner = read_pickle(learner_path)
        #print(learner.func.skill, learner_path)
        #learner.dump_model()
        #learner.dump_parameters()
        #write_json('test.json', learner.results)
        #return

        #learner.transfer_weight = 3
        #learner.hyperparameters = None
        #learner.mean_fn = None
        #learner.transfer_weight = None
        #learner.sim_model = None
        learner.verbose = False
        # TODO: TypeError: not enough arguments for format string
        # due to learner.algorithm namedtuple
        #print(learner, learner.algorithm, learner.nx)
        print(learner.results[-1])
        #skill = learner.func.skill

        # TODO: these aren't really needed anymore
        bad_indices = [i for i, result in enumerate(learner.results)
                       if BAD in result.get('annotation', '')]
        invalid_indices = [i for i, result in enumerate(learner.results)
                           if ('policy' not in result) and not result['valid']]
        valid_indices = [i for i, result in enumerate(learner.results)
                         if ('policy' not in result) and (i not in invalid_indices)]
        sim_indices = invalid_indices + valid_indices + bad_indices
        batch_indices = [i for i, result in enumerate(learner.results)
                         if (i not in sim_indices) and (result['policy'] == TRAINING)]
        active_indices = [i for i, result in enumerate(learner.results)
                          if i not in (sim_indices + batch_indices)]
        real_indices = batch_indices + active_indices

        #indices = list(range(len(learner.results)))
        training_indices = valid_indices + real_indices
        print('Bad: {} | Invalid: {} | Valid: {} | Batch: {} | Active: {} | Total: {}'.format(
            len(bad_indices), len(invalid_indices), len(valid_indices), len(batch_indices), len(active_indices), learner.nx))

        max_train = max(max_train, len(training_indices))
        training_sizes = list(range(MIN_TRAIN, len(training_indices), TRAIN_STEP))
        #training_sizes = list(range(len(sim_indices) + len(batch_indices), len(learner.xx), 1))

        learner_filename = os.path.basename(learner_path)
        learner_name, _ = os.path.splitext(learner_filename)
        if learner_filename in LABELS:
            learner_name = LABELS[learner_filename]
        #algorithm = Algorithm(learner_name, *learner.algorithm[1:])
        algorithm = Algorithm(learner_name, label=learner_name)

        if TRANSFERS == [None]:
            experiments.extend(test_learner(algorithm, learner, training_sizes, test_data))
            continue
        for transfer in TRANSFERS:
            if transfer is not None:
                learner.transfer_weight = transfer
            #if RENAME:
            #algorithm = learner.algorithm
            algorithm = Algorithm(name='transfer={}'.format(learner.transfer_weight)) # copy args?
            #learner.algorithm = algorithm
            experiments.extend(test_learner(algorithm, learner, training_sizes, test_data))

    training_sizes = list(range(MIN_TRAIN, max_train, 1))
    experiments = evaluate_batch(training_sizes, test_data) + experiments

    for metric in METRICS:
        #experiments_name = '{} {}'.format(REAL_PREFIX, skill)
        experiments_name = None
        plot_experiments(test_domain, experiments_name, None, experiments, #scale=1,
                         prediction_metric=metric, include_none=False, verbose=False)

##################################################

ACTIVE_SCOOP_DATASETS = [
    'data/pr2_scoop_20-02-12_17-11-14/trials.json',
    'data/pr2_scoop_20-02-13_15-48-23/trials.json',
    'data/pr2_scoop_20-02-13_16-19-47/trials.json',
    'data/pr2_scoop_20-02-13_16-50-22/trials.json',
    'data/pr2_scoop_20-02-17_16-25-33/trials.json',
    'data/pr2_scoop_20-02-17_16-36-49/trials.json',
    'data/pr2_scoop_20-02-17_16-48-39/trials.json',
]

BEST_SCOOP_DATASETS = [
    ('data/pr2_scoop_20-02-13_17-45-13/trials.json',
     'data/pr2_scoop_20-02-13_17-52-42/trials.json',
     'data/pr2_scoop_20-02-13_18-01-43/trials.json'),  # 50 batch

    ('data/pr2_scoop_20-02-12_17-47-34/trials.json',
     'data/pr2_scoop_20-02-12_17-57-15/trials.json',
     'data/pr2_scoop_20-02-12_18-08-20/trials.json'), # 96 batch

    # 96 active
    (#'data/pr2_scoop_20-02-13_17-11-26/trials.json', # wrong learner
     #'data/pr2_scoop_20-02-13_17-21-06/trials.json', # wrong_learner
     'data/pr2_scoop_20-02-13_17-31-58/trials.json',
     'data/pr2_scoop_20-02-17_17-20-52/trials.json',
     'data/pr2_scoop_20-02-17_17-32-57/trials.json',
    ),
]

def table(pour=False):
    #best_domain = load_data(BEST_DATASETS, verbose=False)
    datasets = BEST_DATASETS if pour else BEST_SCOOP_DATASETS

    for group in datasets:
        if isinstance(group, str):
            group = [group]
        results = [result for path in group for result in read_data(path)]
        skill = results[0]['skill']
        collector = SKILL_COLLECTORS[skill]

        num_scored = total_successful = 0
        total_score = total_filled = 0.
        for result in results:
            if result.get(SCORE, None) is None:
                continue
            num_scored += 1
            score = collector.score_fn(result[FEATURE], result[PARAMETER], result[SCORE])
            total_score += score
            #total_successful += threshold_score(score)
            total_successful += is_successful(result)
            if skill == 'pour':
                filled = compute_fraction_filled(result[SCORE])
            elif skill == 'scoop':
                filled = scoop_fraction_filled(result[FEATURE], result[SCORE])
            else:
                raise ValueError(skill)
            total_filled += clip(filled, min_value=0, max_value=1)

        print('{} {} {} | trials={} | %scored={:.3f} | score={:.3f} | successful={:.3f} | filled={:.3f}'.format(
            group[0], skill, results[0]['policy'], len(results), num_scored / len(results), total_score / num_scored,
            total_successful / num_scored, total_filled / num_scored))

if __name__ == '__main__':
    main()
    #table()

# python2 -m learn_tools.analyze_learner learners/gp_batch_k=mlp_v=true_t=3_19-12-09_18-41-13.pk2
# python2 -m learn_tools.analyze_learner learners/gp_batch_k=mlp_v=true_t=3_19-12-16_15-48-15.pk2
# python2 -m learn_tools.analyze_learner