import random
import numpy as np

from collections import defaultdict

from dimensions.common import BOWLS
from dimensions.bowls.dimensions import BOWL, BOWL_PROPERTIES
from learn_tools.collectors.common import INFEASIBLE, get_contained_beads, fill_with_beads, InitialRanges, \
    create_table_bodies, randomize_dynamics, sample_norm, MAX_BEADS, \
    dump_ranges, MAX_SPILLED, MAX_TRANSLATION, estimate_spoon_capacity, create_beads, pour_beads, \
    sample_bead_parameters
from learn_tools.collectors.collect_pour import piecewise_score
from learn_tools.learner import Collector, PLANNING_FAILURE, FAILURE, SUCCESS, DYNAMICS, rescale, THRESHOLD
from pddlstream.algorithms.constraints import PlanConstraints, WILD as X
from pddlstream.utils import find_unique
from perception_tools.common import create_name
from plan_tools.common import LEFT_ARM, COFFEE, CORRECTION, KG_PER_OZ, SPOON_CAPACITIES_OZ, SPOON_CAPACITIES, SPOONS
from plan_tools.planner import Task
from plan_tools.samplers.grasp import hold_item, SPOON_DIAMETERS, hemisphere_volume, SPOON_THICKNESSES
from plan_tools.samplers.scoop import get_scoop_feature, get_scoop_gen_fn, \
    SCOOP_PARAMETER_FNS, SCOOP_FEATURES, is_valid_scoop
from plan_tools.simulated_problems import update_world
from pybullet_tools.utils import ClientSaver, get_distance, point_from_pose, \
    quat_angle_between, quat_from_pose, get_mass, LockRenderer, set_dynamics, INF, clip

MIN_CAPACITY = 5

# Easier to scoop from bowl than whitebowl due to the curved base
# Easy to scoop chickpeas, medium for red figures, hard for heavy bolts

# Too difficult to scoop from large items when only using 250 particles
BAD_BOWLS = {'large_red_bowl': '37/682 (5%)', 'tan_bowl': '76/660 (12%)', 'lime_bowl': '97/671 (14%)'}

GOOD_BOWLS = {'green_bowl': '299/672 (44%)', 'red_speckled_bowl': '288/668 (43%)', 'blue_bowl': '146/652 (22%)',
              'yellow_bowl': '393/684 (57%)', 'brown_bowl': '487/652 (75%)', 'small_blue_bowl': '278/639 (44%)',
              'orange_bowl': '384/645 (60%)', 'blue_white_bowl': '470/710 (66%)', 'small_green_bowl': '366/650 (56%)',
              'red_bowl': '380/685 (55%)', 'white_bowl': '373/639 (58%)', 'purple_bowl': '168/685 (25%)'}

# TODO: use the green_spoon instead because it has a larger capacity?
SCOOP_SPOONS = SPOONS
#SCOOP_SPOONS = ['orange_spoon'] # grey_spoon | orange_spoon | green_spoon
SCOOP_BOWLS = sorted(set(BOWLS) - set(BAD_BOWLS))
#SCOOP_BOWLS = ['red_speckled_bowl', 'brown_bowl', 'yellow_bowl', 'blue_white_bowl', 'green_bowl', 'blue_bowl']

def collect_scoop(world):
    arm = LEFT_ARM
    spoon_name = create_name(random.choice(SCOOP_SPOONS), 1)
    bowl_name = create_name(random.choice(SCOOP_BOWLS), 1)
    item_ranges = {
        spoon_name: InitialRanges(
            width_range=(1., 1.),
            height_range=(1., 1.),
            mass_range=(1., 1.), # TODO: randomize density?
            pose2d_range=([0.3, 0.5, -np.pi/2],
                          [0.3, 0.5, -np.pi/2]),
        ),
        bowl_name: InitialRanges(
            width_range=(0.8, 1.2),
            height_range=(0.8, 1.2),
            mass_range=(0.8, 1.2),
            pose2d_range=([0.5, 0.0, -np.pi],
                          [0.5, 0.0, np.pi]),  # x, y, theta
        ),
    }
    #dump_ranges(SCOOP_SPOONS, None)
    dump_ranges(SCOOP_BOWLS, item_ranges[bowl_name])

    ##################################################

    # TODO: check collisions/feasibility when sampling
    #bowl_fraction = random.uniform(0.5, 0.75)
    bowl_fraction = random.uniform(0.75, 0.75)
    print('Bowl fraction:', bowl_fraction)
    #bead_radius = sample_norm(mu=0.005, sigma=0.0005, lower=0.004, upper=0.007)
    bead_radius = 0.005
    print('Bead radius:', bead_radius) # Chickpeas have a 1cm diameter
    max_beads = 250

    # TODO: could give the bowl infinite mass
    with ClientSaver(world.client):
        create_table_bodies(world, item_ranges)
        bowl_body = world.get_body(bowl_name)
        update_world(world, target_body=bowl_body)
        parameters_from_name = randomize_dynamics(world)
        parameters_from_name['bead'] = sample_bead_parameters()
        with LockRenderer():
            all_beads = create_beads(max_beads, bead_radius, parameters=parameters_from_name['bead'])
        spoon_capacity = estimate_spoon_capacity(world, spoon_name, all_beads)
        print('{} | Capacity: {} | Mass: {:.3f}'.format(spoon_name, spoon_capacity, 0.0))
        if spoon_capacity < MIN_CAPACITY:
            return INFEASIBLE
        init_holding = hold_item(world, arm, spoon_name)
        if init_holding is None:
            return INFEASIBLE
        # TODO: relate to the diameter of the spoon head. Ensure fraction above this level
        init_beads = fill_with_beads(world, bowl_name, all_beads,
                                     reset_contained=False, height_fraction=bowl_fraction)
        #wait_for_user()
        masses = list(map(get_mass, init_beads))
        mean_mass = np.average(masses)
        init_mass = sum(masses)
        print('Init beads: {} | Init mass: {:.3f} | Mean mass: {:.3f}'.format(len(init_beads), init_mass, mean_mass))
        if len(init_beads) < 2*spoon_capacity:
            return INFEASIBLE
        world.initial_beads.update({bead: bowl_body for bead in init_beads})

    init = [('Contains', bowl_name, COFFEE)]
    goal = [('Contains', spoon_name, COFFEE)]
    skeleton = [
        ('move-arm', [arm, X, X, X]),
        ('scoop', [arm, bowl_name, X, spoon_name, X, COFFEE, X, X, X]),
    ]
    constraints = PlanConstraints(skeletons=[skeleton], exact=True)
    task = Task(init=init, goal=goal, arms=[arm],
                init_holding=init_holding, constraints=constraints)

    ##################################################

    feature = get_scoop_feature(world, bowl_name, spoon_name)
    initial_pose = world.get_pose(bowl_name)

    def score_fn(plan):
        assert plan is not None
        final_pose = world.get_pose(bowl_name)
        point_distance = get_distance(point_from_pose(initial_pose), point_from_pose(final_pose)) #, norm=2)
        quat_distance = quat_angle_between(quat_from_pose(initial_pose), quat_from_pose(final_pose))
        print('Translation: {:.5f} m | Rotation: {:.5f} rads' .format(point_distance, quat_distance))

        with ClientSaver(world.client):
            bowl_beads = get_contained_beads(bowl_body, init_beads)
            fraction_bowl = float(len(bowl_beads)) / len(init_beads) if init_beads else 0
            mass_in_bowl = sum(map(get_mass, bowl_beads))
            spoon_beads = get_contained_beads(world.get_body(spoon_name), init_beads)
            fraction_spoon = float(len(spoon_beads)) / len(init_beads) if init_beads else 0
            mass_in_spoon = sum(map(get_mass, spoon_beads))
        print('In Bowl: {:.3f} | In Spoon: {:.3f}'.format(fraction_bowl, fraction_spoon))
        # TODO: measure change in roll/pitch

        # TODO: could make latent parameters field
        score = {
            # Displacements
            'bowl_translation': point_distance,
            'bowl_rotation': quat_distance,
            # Masses
            'total_mass': init_mass,
            'mass_in_bowl': mass_in_bowl,
            'mass_in_spoon': mass_in_spoon,
            'spoon_mass_capacity': (init_mass / len(init_beads)) * spoon_capacity,
            # Counts
            'num_beads': len(init_beads),
            'bowl_beads': len(bowl_beads),
            'spoon_beads': len(spoon_beads),
            'spoon_capacity': spoon_capacity,
            # Fractions
            'fraction_in_bowl': fraction_bowl,
            'fraction_in_spoon': fraction_spoon,
            # Latent
            'bead_radius': bead_radius,
            DYNAMICS: parameters_from_name
        }

        fraction_filled = float(score['spoon_beads']) / score['spoon_capacity']
        spilled_beads = score['num_beads'] - (score['bowl_beads'] + score['spoon_beads'])
        fraction_spilled = float(spilled_beads) / score['num_beads']
        print('Fraction Filled: {} | Fraction Spilled: {}'.format(fraction_filled, fraction_spilled))

        #_, args = find_unique(lambda a: a[0] == 'scoop', plan)
        #control = args[-1]
        return score

    return task, feature, score_fn

##################################################

# only inf: {'tan_bowl': '4/7 (57%)', 'purple_bowl': '4/7 (57%)', 'red_speckled_bowl': '10/10 (100%)', 'small_green_bowl': '11/11 (100%)', 'yellow_bowl': '10/10 (100%)'}
# w/o correction: {'orange_spoon': '7/15 (47%)', 'green_spoon': '3/16 (19%)', 'grey_spoon': '11/14 (79%)'}
# w/ correction: {'orange_spoon': '7/15 (47%)', 'green_spoon': '3/16 (19%)', 'grey_spoon': '14/14 (100%)'}

SCORE_PER_FEATURE = defaultdict(list)

# grey_spoon: [0.3, 0.5] oz
# orange_spoon: [0.3, 0.6] oz
# green_spoon: [0.1, 0.9] oz

# assumes radius=0.005
BEAD_CAPACITIES = {
    'grey_spoon': 6,
    'orange_spoon': 14,
    'green_spoon': 22, # 28
}

CHICKPEAS_CAPACITIES_OZ = {
    'grey_spoon': SPOON_CAPACITIES_OZ['grey_spoon', 'chickpeas'], # 0.5 oz
    'orange_spoon': SPOON_CAPACITIES_OZ['orange_spoon', 'chickpeas'], # 0.9 oz
    'green_spoon': 10.5 - 9.1, # 1.4 oz
}

CM_FROM_M = 1e-2

def scoop_bowl_difficulty(feature, threshold=0.5):
    min_height = CM_FROM_M * BOWL_PROPERTIES['red'].height
    max_height = CM_FROM_M * BOWL_PROPERTIES['large_red'].height
    interval = (min_height, max_height)
    percent = rescale(feature['bowl_height'], interval=interval, new_interval=(0, 1))
    alpha = rescale(min(percent, threshold), interval=(0, threshold), new_interval=(0.5, 0.9))
    # print(feature['bowl_type'], interval, feature['bowl_height'], alpha)
    # print('{:.3f}, {:.3f}'.format(percent, alpha))
    return alpha

def is_real(feature):
    return not feature.get('simulation', True)

def scoop_spoon_capacity(feature, score):
    # Capacity changes due to different masses
    radius = SPOON_DIAMETERS[feature['spoon_type']] / 2  # - 2*SPOON_THICKNESSES[feature['spoon_type']] / 2
    diameter = 2 * radius
    max_volume = hemisphere_volume(radius, diameter)
    height = 0.9 * feature['bowl_height']
    volume = hemisphere_volume(radius, min(diameter, height))
    fraction = volume / max_volume
    # print(feature['bowl_type'], feature['spoon_type'], fraction)
    # assert feature['spoon_type'] == 'green_spoon'
    # spoon_capacity = score['spoon_capacity']
    if is_real(feature):
        spoon_capacity = KG_PER_OZ * CHICKPEAS_CAPACITIES_OZ[feature['spoon_type']]
    else:
        # TODO: factor in the volume of particles (or just the initial height in the bowl)
        spoon_capacity = score['spoon_capacity']
        #spoon_capacity = BEAD_CAPACITIES[feature['spoon_type']]
    return fraction*spoon_capacity

def scoop_fraction_filled(feature, score):
    #real = 'mass_in_bowl' in score
    #real = 'num_beads' not in score
    real = not feature.get('simulation', True)
    correction = CORRECTION if real else 0
    #correction = 0
    spoon_mass = score['mass_in_spoon' if real else 'spoon_beads'] + correction
    spoon_capacity = scoop_spoon_capacity(feature, score)
    fraction_filled = float(spoon_mass) / spoon_capacity
    return fraction_filled

def scoop_score(feature, parameter, score):
    if score is None:
        return PLANNING_FAILURE
    #bowl_mass = score['mass_in_bowl' if real else 'bowl_beads'] # TODO: correction?
    #total_mass = score['total_mass' if real else 'num_beads'] - correction
    # return score['fraction_in_spoon'] - 0.5 # the proportion of the capacity of the scoop
    # Above half of spoon full

    #from learn_tools.statistics import dump_statistics
    #print()
    # MASSES_PER_FEATURE['{}/{}'.format(feature['spoon_type'], feature['bowl_type'])].append(spoon_mass / KG_PER_OZ) # oz
    # for pair in sorted(MASSES_PER_FEATURE):
    #     dump_statistics(pair, MASSES_PER_FEATURE[pair])
    #SCORE_PER_FEATURE[feature['spoon_type']].append(score['spoon_capacity']) # oz
    #for spoon in sorted(SCORE_PER_FEATURE):
    #    dump_statistics(spoon, SCORE_PER_FEATURE[spoon])

    #spilled_beads = total_mass - (bowl_mass + spoon_mass)
    #fraction_spilled = float(spilled_beads) / total_mass
    #return fraction_spilled

    #if MAX_SPILLED < fraction_spilled:
    #    return FAILURE
    #if MAX_TRANSLATION < score['bowl_translation']: # in meters
    #    return FAILURE
    #if INF <= score['bowl_translation']:
    #    return FAILURE
    #return SUCCESS

    fraction_filled = scoop_fraction_filled(feature, score)
    if is_real(feature):
        alpha = scoop_bowl_difficulty(feature)
    else:
        alpha = 0.7
    # [mass / capacity >= alpha] => [mass >= alpha * capacity]
    return piecewise_score(fraction_filled, alpha)

# TODO: define a function that augments the result with derived statics

SCOOP_COLLECTOR = Collector(collect_scoop, get_scoop_gen_fn, SCOOP_PARAMETER_FNS,
                            is_valid_scoop, SCOOP_FEATURES, scoop_score)
