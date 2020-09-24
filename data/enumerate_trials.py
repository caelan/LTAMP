from __future__ import print_function

import argparse
import os
import sys

sys.path.extend([
    'pddlstream',
    'ss-pybullet',
])

from collections import Counter

from learn_tools.learner import FEATURE, SCORE, CATEGORIES, TRAINING, POLICIES, SKILL
from learn_tools.learnable_skill import read_data
from learn_tools.statistics import get_category_values, DISCRETE_FEATURES
#from learn_tools.collect_simulation import REAL_PREFIX
from learn_tools.collect_simulation import write_results
from pddlstream.utils import safe_rm_dir, INF, ensure_dir

TRIAL_PREFIX = 'trials'

SPECIAL_CATEGORIES = (SKILL, 'material', 'policy')

DATA_DIR = 'data/'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', default=None,
                        help='The prefix of trials')
    parser.add_argument('-l', '--lower', default=0,
                        help='The minimum number of trials')
    parser.add_argument('-u', '--upper', default=INF,
                        help='The minimum number of trials')
    # TODO: select the training policy
    # TODO: date range
    args = parser.parse_args()

    #data_path = os.path.dirname(__file__)
    all_dirname = os.path.join(DATA_DIR, '{}_all/'.format(args.prefix))

    selected_trials = []
    all_data = []
    for trial_dirname in sorted(os.listdir(DATA_DIR)):
        if (args.prefix is not None) and not trial_dirname.startswith(args.prefix):
            continue
        trial_directory = os.path.join(DATA_DIR, trial_dirname)
        if not os.path.isdir(trial_directory) or (trial_directory == all_dirname[:-1]):  # TODO: sloppy
            continue
        for trial_filename in sorted(os.listdir(trial_directory)):
            if trial_filename.startswith(TRIAL_PREFIX):
                trial_path = os.path.join(trial_directory, trial_filename)
                data = list(read_data(trial_path))
                if (len(data) < args.lower) or (args.upper < len(data)):
                    continue
                print('\n{}'.format(os.path.join(DATA_DIR, trial_dirname, trial_filename)))
                # TODO: record the type of failure
                num_scored = sum(result[SCORE] is not None for result in data)
                print('Trials: {} | Scored: {}'.format(len(data), num_scored))

                category_frequencies = {category: Counter(field for result in data if result.get(category, {})
                                                          for field in result.get(category, {}))
                                        for category in CATEGORIES}
                category_frequencies.update({category: Counter(result[category] for result in data if category in result)
                                             for category in SPECIAL_CATEGORIES})
                #if list(category_frequencies.get('policy', {})) != [TRAINING]:
                #    continue
                for category in CATEGORIES + SPECIAL_CATEGORIES:
                    frequencies = category_frequencies.get(category, {})
                    if frequencies:
                        print('{} ({}): {}'.format(category, len(frequencies), sorted(frequencies.keys())))

                category_values = get_category_values(data, include_score=False)
                context_values = category_values[FEATURE]
                for name in sorted(DISCRETE_FEATURES):
                    if name in context_values:
                        print('{}: {}'.format(name, Counter(context_values[name])))
                selected_trials.append(trial_path)
                all_data.extend(data)

                # TODO: print num of successful trials
                #if len(data) < MIN_TRIALS:
                #    response = user_input('Remove {}?'.format(trial_path))
                #    safe_remove(trial_path)
                # Get most recent of each skill
        if not os.listdir(trial_directory):
            print('Removed {}'.format(trial_directory))
            safe_rm_dir(trial_directory)

    print()
    print(len(all_data))
    ensure_dir(all_dirname)
    #write_results(os.path.join(all_dirname, TRIAL_PREFIX), all_data)
    #write_json(path, all_data)
    print(' '.join(selected_trials))

if __name__ == '__main__':
    main()
