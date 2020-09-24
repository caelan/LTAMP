from __future__ import print_function

import os
import numpy as np
import datetime
import random

#from ast import literal_eval

from learn_tools.collect_simulation import SKILL_COLLECTORS
from learn_tools.learner import DEFAULT_INTERVAL, rescale, sample_parameter, x_from_context_sample, SKILL, FEATURE, \
    PARAMETER, SCORE, SEPARATOR, FAILURE_WEIGHT, threshold_score, PLANNING_FAILURE, SUCCESS, FAILURE, THRESHOLD, \
    REAL_PREFIX, DYNAMICS, TRANSFER_WEIGHT, NORMAL_WEIGHT
from learn_tools.common import DATE_FORMAT
from learn_tools.statistics import get_category_values, get_category_ranges, compare_distributions, decompose_discrete
from pddlstream.utils import str_from_object, implies, find_unique
from pybullet_tools.utils import read_pickle, read_json, safe_zip, user_input

def is_real(path):
    return REAL_PREFIX in path

def get_skill(path):
    return find_unique(path.__contains__, SKILL_COLLECTORS)

##################################################

class Dataset(object):
    def __init__(self, func, results=[], scale_paths=None, **kwargs):
        self.func = func # important to use the same ranges
        self.results = list(results)
        self.scale_paths = scale_paths
        self.kwargs = dict(kwargs)
        # TODO: order by path
    @property
    def domain(self):
        return self.func
    def __len__(self):
        return len(self.results)
    def clone(self):
        return Dataset(self.func, self.results, scale_paths=self.scale_paths, **self.kwargs)
    def shuffle(self):
        random.shuffle(self.results)
    def partition(self, index):
        partition = []
        retain = []
        for result in self.results:
            if (self.scale_paths is None) or (result['path'] in self.scale_paths):
                partition.append(result)
            else:
                retain.append(result)
        return Dataset(self.func, (retain + partition[:index]), scale_paths=self.scale_paths, **self.kwargs), \
               Dataset(self.func, partition[index:], scale_paths=self.scale_paths, **self.kwargs)
    def get_data(self):
        return self.func.examples_from_results(self.results, **self.kwargs)
    def __repr__(self):
        return '{}(n={})'.format(self.__class__.__name__, len(self.results))

##################################################

class LearnableSkill(object):

    def __init__(self, name, skill, results=[], feature_ranges={}, parameter_ranges={},
                 lengthscale=1e-1, lengthscale_range=(1e-3, 1e3)):
        self.name = name
        self.skill = skill
        self.results = list(results) # TODO: could do this within the constructor
        self.score_fn = self.collector.score_fn

        # Param must come before context
        self.parameters = tuple(sorted(p for p, (l, u) in parameter_ranges.items() if l != u))
        self.parameter_ranges = parameter_ranges # parameter['ranges']
        self.param_idx = [0 + n for n in range(self.num_params)]

        self.features = tuple(sorted(f for f in self.collector.features if f in feature_ranges))
        #self.features = tuple(filter(lambda f: feature_ranges[f][0] < feature_ranges[f][1],
        #                             sorted(self.collector.features)))
        self.feature_ranges = feature_ranges # feature['ranges']
        self.context_idx = [self.param_idx[-1] + 1 + n for n in range(self.num_context)]

        lower, upper = DEFAULT_INTERVAL
        self.x_range = np.array([
            lower * np.ones(self.dx),
            upper * np.ones(self.dx),
        ])

        self.task_lengthscale = lengthscale*np.ones(self.dx)
        # TODO: maybe these are too large after scaling
        lower, upper = lengthscale_range
        self.lengthscale_bound = np.array([
            lower*np.ones(self.dx),
            upper*np.ones(self.dx),
        ])
        self.sim_only = all(not result[REAL_PREFIX] for result in self.results)
        self.real_only = all(result[REAL_PREFIX] for result in self.results)

    @property
    def collector(self):
        return SKILL_COLLECTORS[self.skill]

    @property
    def num_params(self):
        return len(self.parameters)

    @property
    def num_context(self):
        return len(self.features)

    @property
    def inputs(self):
        return list(x_from_context_sample(self.features, self.parameters))

    @property
    def num_inputs(self):
        return len(self.inputs)

    @property
    def dx(self):
        return self.num_inputs

    @property
    def num_results(self):
        return len(self.results)

    @property
    def nx(self):
        return self.num_results

    @property
    def sim_and_real(self):
        return not self.sim_only and not self.real_only

    def context_from_feature(self, feature):
        context = np.array([rescale(feature[name], self.feature_ranges[name])
                            for name in self.features])
        assert len(context) == len(self.context_idx)
        return context

    def sample_from_parameter(self, parameter):
        sample = np.array([rescale(parameter[name], self.parameter_ranges[name], new_interval=DEFAULT_INTERVAL)
                           for name in self.parameters])
        assert len(sample) == len(self.param_idx)
        return sample

    def parameter_from_sample(self, sample):
        assert len(sample) == len(self.param_idx)
        parameter = sample_parameter(self.parameter_ranges) # In the event that self.parameters is a subset
        parameter.update({name: rescale(value, DEFAULT_INTERVAL, new_interval=self.parameter_ranges[name])
                          for name, value in safe_zip(self.parameters, sample)})
        return parameter

    def x_from_feature_parameter(self, feature, parameter):
        context = self.context_from_feature(feature)
        sample = self.sample_from_parameter(parameter)
        return x_from_context_sample(context, sample) # Important! Sample must come before context

    def example_from_result(self, result, validity=False, binary=False):
        feature = result[FEATURE]
        parameter = result[PARAMETER]
        x = self.x_from_feature_parameter(feature, parameter)
        score = result.get(SCORE, None)
        w = NORMAL_WEIGHT
        if validity:
            y = SUCCESS if score is not None else FAILURE
            return x, y, w

        if score is None:
            y = PLANNING_FAILURE
            #if result.get('valid', True):
            #    w = FAILURE_WEIGHT # Soft failure
        else:
            y = self.score_fn(feature, parameter, score)
            if binary:
                #y = threshold_score(y)
                y = SUCCESS if y > THRESHOLD else FAILURE
            if self.sim_and_real:
                if not result[REAL_PREFIX]:
                    w = TRANSFER_WEIGHT
        return x, y, w

    def examples_from_results(self, results, **kwargs):
        if not results:
            return np.empty([0, self.dx]), np.empty([0]), np.empty([0])
        return tuple(map(np.array, zip(*[self.example_from_result(result, **kwargs)
                                         for result in results])))

    def create_dataset(self, include_invalid=False, include_none=True, **kwargs):
        # if (result.get(PARAMETER, None) is not None)
        return Dataset(self, [result for result in self.results if
                              implies(not result.get('valid', True), include_invalid) and
                              implies(result.get(SCORE, None) is None, include_none)], **kwargs)

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__, self.skill,
                                       str_from_object(self.parameters), str_from_object(self.features))

##################################################

def read_data(path):
    # TODO: move to pybullet-tools
    _, extension = os.path.splitext(path)
    if extension == '.json':
        return read_json(path)
    if extension in ['.pp2', '.pp3', '.pkl', '.pk2', '.pk3']:
        return read_pickle(path)
    raise NotImplementedError(extension)


def analyze_data(data_name, results, verbose=True, **kwargs):
    category_values = get_category_values(results)
    skills = category_values[SKILL]
    if verbose: print('\nSkills:', skills)
    assert len(skills) == 1
    [skill] = skills
    if verbose: print('Examples:', len(results))
    #print('Percentiles:', alphas.round(3).tolist())
    category_ranges = get_category_ranges(category_values, verbose=verbose, **kwargs)
    return LearnableSkill(data_name, skill, results, category_ranges[FEATURE], category_ranges[PARAMETER])

# Relative pour
# https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/commit/3accad1ed6c035f97af030fbe0dd9b1fde9d4d41

# Relative scoop
# https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/commit/8f00bca2591b74e60155fa0c8af843f0691526da

RELATIVE_DATE = datetime.datetime(year=2019, month=6, day=13)  # 2019-06-10 for scoop and 2019-06-13 for pour

def ensure_relative(path, result):
    from plan_tools.samplers.pour import RELATIVE_POUR, scale_parameter, RELATIVE_POUR_SCALING, POUR_PARAMETER_RANGES
    from plan_tools.samplers.scoop import RELATIVE_SCOOP, RELATIVE_SCOOP_SCALING, SCOOP_PARAMETER_RANGES
    # result['path'] = path
    is_pr2 = is_real(path)
    date = datetime.datetime.strptime(result['date'], DATE_FORMAT)
    result[REAL_PREFIX] = is_pr2
    if is_pr2 and (date <= RELATIVE_DATE):
        print('Rescaling because {} <= {}'.format(date, RELATIVE_DATE))
        ranges = result['parameter'].get('ranges', None)
        assert ranges is not None
        #print(result['parameter'])
        if result[SKILL] == 'pour':
            assert RELATIVE_POUR
            #print(ranges == POUR_PARAMETER_RANGES)
            result['parameter'] = scale_parameter(result['feature'], result['parameter'], RELATIVE_POUR_SCALING)
        elif result[SKILL] == 'scoop':
            assert RELATIVE_SCOOP
            #print(ranges == SCOOP_PARAMETER_RANGES)
            result['parameter'] = scale_parameter(result['feature'], result['parameter'], RELATIVE_SCOOP_SCALING)
        #print(result['parameter'])

def load_data(data_paths, verbose=True):
    #data_name = '+'.join(os.path.splitext(os.path.basename(path))[0]
    #                     for path in data_paths)
    #data_name = '+'.join(sorted(os.path.basename(os.path.dirname(path))
    #                     for path in data_paths))
    data_name = data_paths[-1]

    results = []
    for path in data_paths:
        for result in read_data(path):
            if result['parameter'] is None:
                continue
                #assert result['parameter'] is not None
            if not result.get('execution', True):
                continue
            if SKILL not in result:
                result[SKILL] = get_skill(path)
            if 'scoop' == result[SKILL]:
                from learn_tools.collectors.collect_scoop import GOOD_BOWLS, BAD_BOWLS
                if result['feature']['bowl_type'] in BAD_BOWLS:
                    continue

            result['path'] = path
            result['directory'] = os.path.basename(os.path.dirname(path))
            annotation = result.get('annotation', '').strip()
            # TODO: prune escape sequences
            #annotation = literal_eval('"{}"'.format(annotation))
            if ('bad' in annotation) or annotation:
                print('Skipping "{}"'.format(repr(annotation)))
                continue
            ensure_relative(path, result)
            if (result.get(SCORE, None) is not None) and (DYNAMICS in result[SCORE]):
                # TODO: localInertiaDiagonal is still an array
                result[DYNAMICS] = result[SCORE].pop(DYNAMICS)
                result[DYNAMICS] = {'{} {}'.format(name.split('_')[-1], parameter): value for name, parameters in result[DYNAMICS].items()
                                    for parameter, value in parameters.items() if parameter != 'localInertiaDiagonal'}
            results.append(result)
    #decompose_discrete(results, **kwargs)
    #print(SEPARATOR)
    if verbose:
        compare_distributions(results)

    # TODO: filter out test objects
    if verbose:
        print(SEPARATOR)
    return analyze_data(data_name, results, verbose=verbose)
