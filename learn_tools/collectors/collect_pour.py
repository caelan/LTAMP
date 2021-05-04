import random

import numpy as np

from dimensions.common import CUPS, BOWLS
from learn_tools.collectors.common import INFEASIBLE, get_contained_beads, fill_with_beads, InitialRanges, \
    create_table_bodies, randomize_dynamics, sample_norm, MAX_BEADS, \
    dump_ranges, MAX_SPILLED, MAX_TRANSLATION, create_beads, sample_bead_parameters
from learn_tools.learner import Collector, PLANNING_FAILURE, FAILURE, SUCCESS, DYNAMICS, REAL_PREFIX, rescale, THRESHOLD
from pddlstream.algorithms.constraints import PlanConstraints, WILD as X
from pddlstream.language.statistics import safe_ratio
from pddlstream.utils import find_unique
from perception_tools.common import create_name
from plan_tools.common import LEFT_ARM, COFFEE, CORRECTION
from plan_tools.planner import Task
from plan_tools.samplers.collision import body_pair_collision
from plan_tools.samplers.grasp import has_grasp
from plan_tools.samplers.pour import get_pour_feature, get_pour_gen_fn, POUR_PARAMETER_FNS, POUR_FEATURES, is_valid_pour
from plan_tools.simulated_problems import update_world
from pybullet_tools.utils import ClientSaver, get_distance, point_from_pose, \
    quat_angle_between, quat_from_pose, get_mass, LockRenderer, wait_for_user

MIN_BEADS = 10

POUR_CUPS = CUPS
POUR_BOWLS = BOWLS

#cup_index = -2
#POUR_CUPS = CUPS[cup_index:cup_index+1]
#bowl_index = -2
#POUR_BOWLS = BOWLS[bowl_index:bowl_index+1]

def interval(center=1.0, extent=0.0):
    return (center - extent, center + extent)

def collect_pour(world, bowl_type=None, cup_type=None, randomize=True, **kwargs):
    arm = LEFT_ARM
    bowl_type = random.choice(POUR_BOWLS) if bowl_type is None else bowl_type
    cup_type = random.choice(POUR_CUPS) if cup_type is None else cup_type

    # TODO: could directly randomize base_diameter and top_diameter
    scale = int(randomize)
    cup_name = create_name(cup_type, 1)
    bowl_name = create_name(bowl_type, 1)
    item_ranges = {
        cup_name: InitialRanges(
            width_range=interval(extent=0.2*scale), # the cups are already fairly small
            #height_range=(0.9, 1.1),
            height_range=(1.0, 1.2), # Difficult to grasp when this shrinks
            mass_range=interval(extent=0.2*scale),
            pose2d_range=([0.5, 0.3, -np.pi],
                          [0.5, 0.3, np.pi]), # x, y, theta
        ),
        bowl_name: InitialRanges(
            width_range=interval(extent=0.4*scale),
            height_range=interval(extent=0.4*scale),
            mass_range=interval(extent=0.4*scale),
            pose2d_range=([0.5, 0.0, -np.pi],
                          [0.5, 0.0, np.pi]),  # x, y, theta
        ),
    }
    if 'item_ranges' in kwargs:
        for item in kwargs['item_ranges']:
            item_ranges[item] = kwargs['item_ranges'][item]

    #dump_ranges(POUR_CUPS, item_ranges[cup_name])
    #dump_ranges(POUR_BOWLS, item_ranges[bowl_name])
    #for name, limits in item_ranges.items():
    #    lower, upper = aabb_from_points(read_obj(get_body_obj(name)))
    #    extents = upper - lower
    #    print(name, 'width', (extents[0]*np.array(limits.width_range)).tolist())
    #    print(name, 'height', (extents[2]*np.array(limits.height_range)).tolist())

    # TODO: check collisions/feasibility when sampling
    cup_fraction = random.uniform(0.75, 1.0)
    print('Cup fraction:', cup_fraction)
    bowl_fraction = random.uniform(1.0, 1.0)
    print('Bowl fraction:', bowl_fraction)
    bead_radius = sample_norm(mu=0.006, sigma=0.001*scale, lower=0.004)
    print('Bead radius:', bead_radius)
    num_beads = 100

    with ClientSaver(world.client):
        create_table_bodies(world, item_ranges, randomize=randomize)
        bowl_body = world.get_body(bowl_name)
        cup_body = world.get_body(cup_name)
        update_world(world, bowl_body)
        if body_pair_collision(cup_body, bowl_body):
            # TODO: automatically try all pairs of bodies
            return INFEASIBLE
        if not has_grasp(world, cup_name):
            return INFEASIBLE
        # TODO: flatten and only store bowl vs cup
        parameters_from_name = randomize_dynamics(world, randomize=randomize)
        parameters_from_name['bead'] = sample_bead_parameters() # bead restitution is the most important
        with LockRenderer():
            all_beads = create_beads(num_beads, bead_radius, parameters=parameters_from_name['bead'])
        bowl_beads = fill_with_beads(world, bowl_name, all_beads,
                                     reset_contained=True, height_fraction=bowl_fraction)
        init_beads = fill_with_beads(world, cup_name, bowl_beads,
                                     reset_contained=False, height_fraction=cup_fraction)
        #wait_for_user()
        init_mass = sum(map(get_mass, init_beads))
        print('Init beads: {} | Init mass: {:.3f}'.format(len(init_beads), init_mass))
        if len(init_beads) < MIN_BEADS:
            return INFEASIBLE
        world.initial_beads.update({bead: cup_body for bead in init_beads})

    latent = {
        'num_beads': len(init_beads),
        'total_mass': init_mass,
        'bead_radius': bead_radius,
        DYNAMICS: parameters_from_name,
    }

    init = [('Contains', cup_name, COFFEE)]
    goal = [('Contains', bowl_name, COFFEE)]
    skeleton = [
        ('move-arm', [arm, X, X, X]),
        ('pick', [arm, cup_name, X, X, X, X, X]),
        ('move-arm', [arm, X, X, X]),
        ('pour', [arm, bowl_name, X, cup_name, X, COFFEE, X, X, X]),
    ]
    constraints = PlanConstraints(skeletons=[skeleton], exact=True)
    task = Task(init=init, goal=goal, arms=[arm], constraints=constraints) #, init_holding=init_holding)

    ##################################################

    feature = get_pour_feature(world, bowl_name, cup_name)
    #feature['ranges'] = POUR_FEATURE_RANGES
    initial_pose = world.get_pose(bowl_name)

    def score_fn(plan):
        assert plan is not None
        final_pose = world.get_pose(bowl_name)
        point_distance = get_distance(point_from_pose(initial_pose),
                                      point_from_pose(final_pose)) #, norm=2)
        quat_distance = quat_angle_between(quat_from_pose(initial_pose),
                                           quat_from_pose(final_pose))
        print('Translation: {:.5f} m | Rotation: {:.5f} rads' .format(
            point_distance, quat_distance))

        with ClientSaver(world.client):
            # TODO: lift the bowl up (with particles around) to prevent scale detections
            final_bowl_beads = get_contained_beads(bowl_body, init_beads)
            fraction_bowl = safe_ratio(len(final_bowl_beads), len(init_beads), undefined=0)
            mass_in_bowl = sum(map(get_mass, final_bowl_beads))
            final_cup_beads = get_contained_beads(cup_body, init_beads)
            fraction_cup = safe_ratio(len(final_cup_beads), len(init_beads), undefined=0)
            mass_in_cup = sum(map(get_mass, final_cup_beads))
        print('In Bowl: {} | In Cup: {}'.format(fraction_bowl, fraction_cup))

        score = {
            # Displacements
            'bowl_translation': point_distance,
            'bowl_rotation': quat_distance,
            # Masses
            'mass_in_bowl': mass_in_bowl,
            'mass_in_cup': mass_in_cup,
            # Counts
            'bowl_beads': len(final_bowl_beads),
            'cup_beads': len(final_cup_beads),
            # Fractions
            'fraction_in_bowl': fraction_bowl,
            'fraction_in_cup': fraction_cup,
        }
        score.update(latent)
        # TODO: store the cup path length to bias towards shorter paths

        #_, args = find_unique(lambda a: a[0] == 'pour', plan)
        #control = args[-1]
        #feature = control['feature']
        #parameter = control['parameter']
        return score

    return task, feature, score_fn

##################################################

def compute_fraction_filled(score):
    #real = REAL_PREFIX
    #real = not feature['simulated']
    real = 'bowl_beads' not in score

    correction = CORRECTION if real else 0
    #correction = 0
    mass_in_bowl = score['mass_in_bowl' if real else 'bowl_beads'] + correction
    #mass_in_cup = score['mass_in_cup' if real else 'spoon_beads']
    #mass_in_cup = 0.0
    total_mass = score['total_mass' if real else 'num_beads'] - correction
    # TODO: different sim / real scoring functions
    #print('Bowl: {:.1f}g | Cup: {:.1f}g | Total {:.1f}g | Correction: {:.1f}g'.format(
    #    1000*mass_in_bowl, 1000*mass_in_cup, 1000*total_mass, 1000*correction))

    fraction_filled = float(mass_in_bowl) / total_mass
    return fraction_filled

def piecewise_score(value, threshold):
    if value <= threshold:
        return rescale(value, interval=(0, threshold), new_interval=(FAILURE, THRESHOLD))
    return rescale(value, interval=(threshold, 1), new_interval=(THRESHOLD, SUCCESS))

def pour_score(feature, parameter, score, alpha=0.9): # TODO: adjust back to 0.95?
    assert score is not None
    fraction_filled = compute_fraction_filled(score)
    #spilled_beads = total_mass - (mass_in_bowl + mass_in_cup)
    #fraction_spilled = float(spilled_beads) / total_mass
    #print('Filled: {:.1f}% | Spilled: {:.1f}%'.format(100*fraction_filled, 100*fraction_spilled))
    #if MAX_SPILLED < fraction_spilled: # TODO: need to enforce cup orientation constraint to measure spillage
    #    return FAILURE
    #if MAX_TRANSLATION < score['bowl_translation']:  # in meters
    #    return FAILURE
    return piecewise_score(fraction_filled, alpha)
    #return np.exp(2*10*(fraction_filled - alpha)) - 1


POUR_COLLECTOR = Collector(collect_pour, get_pour_gen_fn, POUR_PARAMETER_FNS,
                           is_valid_pour, POUR_FEATURES, pour_score)
