from __future__ import print_function, division

from collections import OrderedDict, Counter

import numpy as np
import scipy
from scipy.stats import ks_2samp

from learn_tools.collect_simulation import SKILL_COLLECTORS
from learn_tools.learner import SKILL, CATEGORIES, SCORE, FEATURE, PARAMETER, THRESHOLD, RANGES
from pddlstream.utils import str_from_object

DISCRETE_FEATURES = ['bowl_type', 'spoon_type', 'cup_type']

DEFAULT_ALPHAS = np.linspace(0.0, 1.0, num=11, endpoint=True)

def is_string(s):
    try:
        return isinstance(s, basestring) # Python2 + unicode
    except NameError:
        return isinstance(s, str) # Python3


def compute_percentiles(values, alphas=DEFAULT_ALPHAS):
    if not values:
        return np.zeros(alphas.shape)
    return np.array([scipy.stats.scoreatpercentile(
        values, 100 * p, interpolation_method='lower') for p in alphas])  # numpy.percentile


def histogram(data, name=None, x_axis=None):
    import matplotlib.pyplot as plt
    if not isinstance(data, dict):
        data = {None: data}
    num_samples = sum(len(samples) for samples in data.values())
    for label in sorted(data):
        samples = data[label]
        print(label, len(samples), num_samples)
        weights = np.ones(len(samples)) / num_samples
        n, bins, patches = plt.hist(samples, bins=None, density=False, weights=weights, # facecolor='b',
                                    alpha=0.5, cumulative=False, label=label) # normed=False,
    plt.xlabel(x_axis)
    plt.ylabel('Fraction')
    plt.title(name)
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()
    return plt

##################################################

def get_category_values(results, include_score=True):
    category_values = {}
    for result in results:
        skill = result[SKILL]
        category_values.setdefault(SKILL, set()).add(skill)
        for category in CATEGORIES:
            d = result.get(category, None)
            if d is None:
                continue
            if include_score and (category == SCORE):
                d['score'] = SKILL_COLLECTORS[skill].score_fn(result[FEATURE], result[PARAMETER], d)
            for name, value in d.items():
                if name != RANGES: # Parameter has ranges
                    category_values.setdefault(category, {}).setdefault(name, []).append(value)
    return category_values

def dump_statistics(name, values, alphas=DEFAULT_ALPHAS):
    print('{}: num={}, mean={:.3f}, std={:.3f}, percentiles={}'.format(
        name, len(values), np.mean(values), np.std(values),
        compute_percentiles(values, alphas).round(3).tolist()))

def get_category_ranges(category_values, alphas=DEFAULT_ALPHAS, verbose=True):
    if verbose:
        print('Category ranges:', CATEGORIES)
        print('Alphas:', alphas.round(3).tolist())
    category_ranges = {}
    for category in CATEGORIES:
        if verbose: print('\n{}:'.format(category))
        category_ranges[category] = OrderedDict()
        for i, name in enumerate(sorted(category_values.get(category, {}))):
            if name in ['policy', 'rotate_bowl']:
                continue
            values = category_values[category][name]
            if is_string(values[0]):
                if verbose: print('{}) {}: {}'.format(i, name, Counter(values)))
            elif isinstance(values[0], dict):
                # TODO: handle dicts
                pass
            else:
                # if category in (SCORE,): # Everything else should be about uniform
                #    histogram(values, category, name)
                # lower, upper = scipy.stats.norm.interval(alpha, np.mean(values), np.std(values))
                if verbose:
                    dump_statistics('{}) {}'.format(i, name), values)
                # range=[{:.3f}, {:.3f}], np.min(values), np.max(values)
                min_value, max_value = np.min(values), np.max(values)
                if min_value != max_value:
                    category_ranges[category][name] = (min_value, max_value)
    return category_ranges

##################################################

def get_scored_results(results):
    return [result for result in results
            if result.get(SCORE, None) is not None]

def is_successful(result):
    return THRESHOLD < SKILL_COLLECTORS[result[SKILL]].score_fn(
        result[FEATURE], result[PARAMETER], result.get(SCORE, None))

def get_successful_results(results):
    return [result for result in results if is_successful(result)]


def get_unsuccessful_results(results):
    return [result for result in results if not is_successful(result)]

##################################################

def decompose_discrete(results, **kwargs):
    data_from_discrete = {}
    for result in results:
        key = frozenset((feature, result[FEATURE][feature]) for feature in DISCRETE_FEATURES
                        if feature in result[FEATURE])
        data_from_discrete.setdefault(key, []).append(result)

    sorted_triplets = []
    for key, key_data in data_from_discrete.items():
        category_values = get_category_values(key_data, **kwargs)
        scores = category_values[SCORE]['score'] if SCORE in category_values else []
        sorted_triplets.append((dict(key), len(scores), compute_percentiles(scores, **kwargs)))
    for i, (discrete, num, percentiles) in enumerate(
            sorted(sorted_triplets, key=lambda t: (t[-1][-1], t[1]), reverse=True)):
        print('{}) {} | Num: {} | %: {}'.format(
            i, str_from_object(discrete), num, percentiles.round(3).tolist()))
        #analyze_data(data_name, data_from_discrete[key])
    return data_from_discrete


def compare_distributions(results):
    scored_results = get_scored_results(results)
    #scored_results = results
    unsuccessful_results = get_unsuccessful_results(scored_results)
    unsuccessful_values = get_category_values(unsuccessful_results)
    # analyze_data(None, unsuccessful_results)
    # print('Unsuccessful')
    # print(SEPARATOR)
    successful_results = get_successful_results(scored_results)
    successful_values = get_category_values(successful_results)
    # analyze_data(None, successful_results)
    # print('Successful')
    # print(SEPARATOR)
    print('{} Scored | {} Successful | {} Unsuccessful'.format(
        len(scored_results), len(successful_results), len(unsuccessful_results)))

    discrete_values = {}
    continuous_values = {}
    for category in CATEGORIES:
        names = set(unsuccessful_values.get(category, {})) | \
                set(successful_values.get(category, {}))
        for i, name in enumerate(sorted(names)):
            if name in ['policy']: #, 'rotate_bowl']:
                continue
            unsuccessful = unsuccessful_values[category].get(name, [])
            successful = successful_values[category].get(name, [])
            combined = unsuccessful + successful
            if is_string(combined[0]):
                discrete_values[category, name] = (Counter(unsuccessful), Counter(successful))
            elif isinstance(combined[0], dict):
                # TODO: handle dicts
                pass
            else:
                continuous_values[category, name] = (unsuccessful, successful)

    print('\nDiscrete Percent Successful:')
    for (category, name), (unsuccessful, successful) in sorted(
            discrete_values.items(), key=lambda pair: sum((pair[1][0] & pair[1][1]).values()), reverse=True):
        #intersection = unsuccessful & successful
        #union = unsuccessful | successful
        total = Counter(unsuccessful)
        total.update(successful)
        # Important that this is relative to number of attempts
        # Cups seem to not impact pouring (expected)
        # Easier to pour into large bowls
        #delta = {k: round((successful[k] - unsuccessful[k]) / union[k], 3) for k in union}
        delta = {k: '{}/{} ({:.0f}%)'.format(successful[k], total[k], 100 * successful[k] / total[k]) for k in total}
        print(category, name, sum(total.values()), delta)

    print('\nContinuous Percentile Successful:')
    #alphas = np.linspace(0.0, 1.0, num=5, endpoint=True)
    alphas = np.array([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    print('Alphas:', alphas.round(3).tolist())
    for (category, name), (unsuccessful, successful) in sorted(
            continuous_values.items(), key=lambda pair: ks_2samp(*pair[1])[1], reverse=True):
        #print(sorted(unsuccessful))
        #print(sorted(successful))
        statistic, pvalue = ks_2samp(unsuccessful, successful) # statistic is the max difference in CDF
        print('{:10}{:25} ks-stat={:.3f}, p-value={:.3f}, F={}, T={}'.format(
            category, name, statistic, pvalue,
            compute_percentiles(unsuccessful, alphas=alphas).round(3).tolist(),
            compute_percentiles(successful, alphas=alphas).round(3).tolist()))
        #if pvalue < 1e-3:
        #    histogram({'unsuccessful': unsuccessful, 'successful': successful}, name=category, x_axis=name)
