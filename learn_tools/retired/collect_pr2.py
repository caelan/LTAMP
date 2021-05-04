#!/usr/bin/env python2

from __future__ import print_function

import sys

# NOTE(caelan): must come before other imports
sys.path.extend([
    'pddlstream',
    'ss-pybullet',
    #'pddlstream/examples',
    #'pddlstream/examples/pybullet/utils',
])

import argparse
import numpy as np
import time
import rosnode
import roslaunch
import os
import math
import datetime
import rospy
#np.set_printoptions(precision=3, threshold=3, edgeitems=1, suppress=True) #, linewidth=1000)

from std_msgs.msg import String

from itertools import count, product, cycle

from pddlstream.utils import str_from_object, get_python_version, implies, write_pickle
from pddlstream.algorithms.constraints import PlanConstraints, WILD as X

from control_tools.execution import execute_plan, get_arm_prefix
from perception_tools.common import get_type, create_name, get_body_urdf
from plan_tools.common import LEFT_ARM, COFFEE, SUGAR, MODEL_MASSES, \
    SCALE_TYPE, KG_PER_OZ, is_obj_type, SPOON_CAPACITIES, MATERIALS, SPOONS
from plan_tools.retired.pr2_problems import get_spoon_init_holding
from plan_tools.ros_world import ROSWorld
from plan_tools.planner import plan_actions, PlanningWorld, Task
from plan_tools.visualization import draw_forward_reachability, draw_names
from learn_tools.collectors.collect_push import POSE2D_RANGE, draw_push_goal
from learn_tools.learner import TRAINING, SEPARATOR, LEARNER_DIRECTORY, SKILL
from learn_tools.collect_simulation import get_parameter_fns, SKILL_COLLECTORS, test_validity, get_data_path, write_results
from learn_tools.active_learner import BEST, STRADDLE, ActiveLearner
from learn_tools.learnable_skill import load_data
from plan_tools.samplers.scoop import get_scoop_feature
from plan_tools.samplers.push import get_push_feature
from plan_tools.samplers.pour import get_pour_feature
from learn_tools.common import DATE_FORMAT
from learn_tools.select_active import optimize_feature

from dimensions.bowls.dimensions import BOWL
from dimensions.cups.dimensions import CUP
from dimensions.common import CUPS, BOWLS

# for pickling
from learn_tools.analyze_experiment import get_label

from pybullet_tools.utils import ClientSaver, elapsed_time, get_pose, point_from_pose, \
    get_distance, quat_from_pose, quat_angle_between, get_point, load_pybullet, stable_z, wait_for_user, remove_all_debug, set_camera_pose, INF, \
    draw_aabb, create_box, get_aabb_center, \
    get_aabb_extent, aabb_from_points, Pose, Point, set_pose, multiply, get_euler, set_euler, randomize, read_pickle, HideOutput, WorldSaver, ensure_dir, clip

from learn_tools.run_active import SCOOP_TEST_DATASETS, evaluate_confusions # TEST_DATASETS
from plan_tools.retired.run_pr2 import get_task_fn, add_holding, move_to_initial_config
from retired.utils.scale_reader import read_scales, ARIADNE_BACK_USB, ARIADNE_FRONT_USB


TEST_BOWLS = {'red_speckled_bowl', 'yellow_bowl', 'tan_bowl', 'purple_bowl', 'small_green_bowl'}
TEST_CUPS = {'olive3_cup', 'yellow_cup', 'red3D_cup', 'green3D_cup', 'cyan3D_cup', 'orange_cup'}
TEST_SPOONS = {'orange_spoon'}

TRAIN_BOWLS = set(BOWLS) - TEST_BOWLS
TRAIN_CUPS = set(CUPS) - TEST_CUPS
TRAIN_SPOONS = TEST_SPOONS

SCOOP_CUP = 'whitebowl'

#fraction = 1./3
#print('#bowls={}, #cups={}, #spoons={}'.format(len(BOWLS), len(CUPS), len(SPOONS)))
#print('Test bowls:', random.sample(BOWLS, int(fraction*len(BOWLS))))
#print('Test cups:', random.sample(CUPS, int(fraction*len(CUPS))))
#print('Test spoons:', random.sample(SPOONS, int(fraction*len(SPOONS))))
#quit()
# TODO: presample combinations and avoid relabeling the same ones

# Training objects
# - Must be visually different for visual detector to differentiate
# - Must not be transparent in order to show up in the pointcloud (can always cover in tape)
# - Cups must be made of plastic or something that won't break if dropped
# - Might be different types of bowl bases

POUR_TEST_PAIRS = [
    ('orange_cup', 'red_speckled_bowl'), ('cyan3D_cup', 'red_speckled_bowl'), ('yellow_cup', 'red_speckled_bowl'),
    ('orange_cup', 'yellow_bowl'), ('green3D_cup', 'tan_bowl'), ('olive3_cup', 'yellow_bowl'),
    ('cyan3D_cup', 'small_green_bowl'), ('yellow_cup', 'purple_bowl'), ('yellow_cup', 'small_green_bowl'),
    ('olive3_cup', 'small_green_bowl'), ('red3D_cup', 'purple_bowl'), ('orange_cup', 'tan_bowl'), ('orange_cup', 'purple_bowl'),
    ('olive3_cup', 'purple_bowl'), ('olive3_cup', 'red_speckled_bowl'), ('cyan3D_cup', 'purple_bowl'), ('olive3_cup', 'tan_bowl'),
    ('green3D_cup', 'small_green_bowl'), ('yellow_cup', 'yellow_bowl'), ('green3D_cup', 'red_speckled_bowl'),
    ('green3D_cup', 'purple_bowl'), ('red3D_cup', 'tan_bowl'), ('red3D_cup', 'small_green_bowl'), ('orange_cup', 'small_green_bowl'),
    ('cyan3D_cup', 'tan_bowl'), ('cyan3D_cup', 'yellow_bowl'), ('green3D_cup', 'yellow_bowl'), ('red3D_cup', 'yellow_bowl'),
    ('red3D_cup', 'red_speckled_bowl'), ('yellow_cup', 'tan_bowl')]

SCOOP_TEST = sorted(TEST_BOWLS)

##################################################

def get_pour_items(ros_world):
    items = set(info.name for info in ros_world.perception.items)
    cups = {name for name in items if is_obj_type(name, CUP)}
    bowls = {name for name in items if is_obj_type(name, BOWL)}
    if (len(cups) != 1) or (len(bowls) != 1) or (cups & bowls):
        return None
    [cup] = cups
    [bowl] = bowls
    return cup, bowl

def get_scoop_items(ros_world):
    items = set(info.name for info in ros_world.perception.items)
    cups = {name for name in items if is_obj_type(name, SCOOP_CUP)} # white_bowl is commented out
    bowls = {name for name in items if is_obj_type(name, BOWL)} - cups
    if (len(cups) != 1) or (len(bowls) != 1):
        return None
    [cup] = cups
    [bowl] = bowls
    return cup, bowl

REQUIREMENT_FNS = {
    'pour': get_pour_items,
    'scoop': get_scoop_items,
}

##################################################

# arm x open
ACTIVE_ARMS = {
    'pour': (LEFT_ARM, True),
    #'scoop': (RIGHT_ARM, False), # Left gripper has tape over it
    'scoop': (LEFT_ARM, False),  # Left gripper has tape over it

}

def collect_pour(args, ros_world):
    arm, _ = ACTIVE_ARMS['pour']
    cup_name, bowl_name = get_pour_items(ros_world)

    init = [
        ('Contains', cup_name, COFFEE),
    ]
    goal = [
        ('Contains', bowl_name, COFFEE),
    ]
    skeleton = [
        ('move-arm', [arm, X, X, X]),
        ('pick', [arm, cup_name, X, X, X, X, X]),
        ('move-arm', [arm, X, X, X]),
        ('pour', [arm, bowl_name, X, cup_name, X, COFFEE, X, X, X]),
        ('move-arm', [arm, X, X, X]),
        ('place', [arm, cup_name, X, X, X, X, X, X, X, X]),
        ('move-arm', [arm, X, X, X]),
    ]
    constraints = PlanConstraints(skeletons=[skeleton], exact=True)
    # TODO: remove required

    # TODO: ensure the robot reaches its start configuration
    task = Task(init=init, goal=goal, arms=[arm],
                required=[cup_name, bowl_name], # TODO: table?
                reset_arms=True, empty_arms=True,
                use_scales=True, constraints=constraints)
    feature = get_pour_feature(ros_world, bowl_name, cup_name)

    return task, feature

def collect_scoop(args, ros_world):
    arm, _ = ACTIVE_ARMS['scoop']
    cup_name, bowl_name = get_scoop_items(ros_world)
    spoon_name = create_name(args.spoon, 1) # grey_spoon | orange_spoon | green_spoon

    init_holding = get_spoon_init_holding(arm, spoon_name)
    init = [
        ('Contains', bowl_name, SUGAR),
    ]
    goal = [
        ('Contains', cup_name, SUGAR),
    ]
    skeleton = [
        ('move-arm', [arm, X, X, X]),
        ('scoop', [arm, bowl_name, X, spoon_name, X, SUGAR, X, X, X]),
        ('move-arm', [arm, X, X, X]),
        ('pour', [arm, cup_name, X, spoon_name, X, SUGAR, X, X, X]),
        ('move-arm', [arm, X, X, X]),
    ]
    constraints = PlanConstraints(skeletons=[skeleton], exact=True)

    task = Task(init=init, init_holding=init_holding, goal=goal, arms=[arm],
                required=[cup_name, bowl_name],
                reset_arms=True, use_scales=True, constraints=constraints)
    add_holding(task, ros_world)
    feature = get_scoop_feature(ros_world, bowl_name, spoon_name)

    return task, feature

def collect_push(args, ros_world):
    arm = LEFT_ARM
    block_name, = get_pour_items(ros_world)

    goal_pos2d = np.random.uniform(*POSE2D_RANGE)[:2]
    init = [('CanPush', block_name, goal_pos2d)]
    goal = [('InRegion', block_name, goal_pos2d)]
    skeleton = [
        ('move-arm', [arm, X, X, X]),
        ('push', [arm, block_name, X, X, X, X, X]),
        ('move-arm', [arm, X, X, X]),
    ]
    constraints = PlanConstraints(skeletons=[skeleton], exact=True)
    task = Task(init=init, goal=goal, arms=[arm], required=[block_name],
                reset_arms=True, use_scales=False, constraints=constraints)

    initial_pose = ros_world.perception.get_pose(block_name)
    feature = get_push_feature(ros_world, block_name, arm, initial_pose, goal_pos2d)

    for goal in task.goal:
        if goal[0] == 'InRegion':
            _, block_name, goal_pos2d = goal
            draw_push_goal(ros_world, block_name, goal_pos2d)

    return task, feature

##################################################

POUR = False

PROBLEMS = [
    collect_pour,
    collect_scoop,
    collect_push,
]

# TODO: front USB ports are unresponsive
SCALE_FROM_BUS = {
    ARIADNE_BACK_USB: create_name(SCALE_TYPE, 1), # Right scale
}
if POUR:
    SCALE_FROM_BUS.update({
        ARIADNE_FRONT_USB: create_name(SCALE_TYPE, 2),  # Left scale
    })

def read_raw_masses(stackings):
    if not stackings:
        return {}
    weights = {SCALE_FROM_BUS[bus]: weight
               for bus, weight in read_scales().items()}
    print('Raw weights (oz):', str_from_object(weights))
    for scale in stackings:
        if scale not in weights:
            # TODO: return which scale failed
            print('Scale {} is not turned on!'.format(scale))
            return None
    #assert all(0 < weight for weight in weights.values()) # TODO: check that objects are on
    masses = {stackings[scale]: KG_PER_OZ*weight
             for scale, weight in weights.items()}
    print('Raw masses (kg):', str_from_object(masses))
    return masses

def estimate_particles(masses, material):
    particles = {item: mass / MODEL_MASSES[material]
                 for item, mass in masses.items()}
    print('{} (num): {}'.format(material, particles))
    return particles

def estimate_masses(masses, bowl_masses, material=None):
    print('Bowl masses (kg):', bowl_masses)
    #material_masses = {bowl: max(0.0, mass - bowl_masses[bowl])
    #                   for bowl, mass in masses.items()}
    material_masses = {bowl: mass - bowl_masses[bowl] for bowl, mass in masses.items()}
    print('Material masses (kg):', str_from_object(material_masses))
    #if material is not None:
    #    estimate_particles(masses, material)
    return material_masses

##################################################

def bowls_holding_material(facts):
    names = set()
    for fact in facts:
        if fact[0] == 'Contains':
            names.add(fact[1])
    return names
    # TODO: method for creating the task once the state is parsed

def wait_until_observation(problem, ros_world, max_time=5):
    ros_world.sync_controllers()  # AKA update sim robot joint positions
    iteration = count()
    start_time = time.time()
    ros_world.perception.updated_simulation = False
    while elapsed_time(start_time) < max_time:
        ros_world.perception.wait_for_update()
        names = set(info.name for info in ros_world.perception.surfaces + ros_world.perception.items)
        items = REQUIREMENT_FNS[problem](ros_world)
        print('Iteration: {} | Bodies: {} | Time: {:.3f}'.format(next(iteration), sorted(names),
                                                                 elapsed_time(start_time)))
        if items is not None:
            break
    else:
        print('Failed to detect required objects for {}'.format(problem))
        return None
    ros_world.perception.update_simulation()
    # TODO: return the detected items
    return items

##################################################

def score_poses(problem, task, ros_world):
    cup_name, bowl_name = REQUIREMENT_FNS[problem](ros_world)
    name_from_type = {'cup': cup_name, 'bowl': bowl_name}
    initial_poses = {ty: ros_world.get_pose(name) for ty, name in name_from_type.items()}
    final_stackings = wait_until_observation(problem, ros_world)
    #final_stackings = None
    if final_stackings is None:
        # TODO: only do this for the bad bowl
        point_distances = {ty: INF for ty in name_from_type}
        quat_distances = {ty: 2*np.pi for ty in name_from_type} # Max rotation
    else:
        final_poses = {ty: ros_world.get_pose(name) for ty, name in name_from_type.items()}
        point_distances = {ty: get_distance(point_from_pose(initial_poses[ty]), point_from_pose(final_poses[ty]))
                           for ty in name_from_type}
        # TODO: wrap around distance for symmetric things
        quat_distances = {ty: quat_angle_between(quat_from_pose(initial_poses[ty]), quat_from_pose(final_poses[ty]))
                          for ty in name_from_type}
    score = {}
    for ty in sorted(name_from_type):
        print('{} | translation (m): {:.3f} | rotation (degrees): {:.3f}'.format(
            ty, point_distances[ty], math.degrees(quat_distances[ty])))
        score.update({
            '{}_translation'.format(ty): point_distances[ty],
            '{}_rotation'.format(ty): quat_distances[ty],
        })
    return score

def score_masses(args, task, ros_world, init_total_mass, bowl_masses, final_raw_masses):
    # TODO: assert that mass is not zero if object known to be above
    # TODO: unify this with the simulator scoring
    cup_name, bowl_name = REQUIREMENT_FNS[args.problem](ros_world)
    name_from_type = {
        'cup': cup_name,
    }
    if POUR:
        name_from_type.update({
            'bowl': bowl_name,
        })

    final_masses = estimate_masses(final_raw_masses, bowl_masses, args.material)
    if POUR:
        final_total_mass = sum(final_masses.values())
        print('Total mass:', final_total_mass)
        print('Spilled (%):', clip(init_total_mass - final_total_mass, min_value=0, max_value=1))
    score = {'total_mass': init_total_mass}
    score.update({'mass_in_{}'.format(ty): final_masses[name]
                  for ty, name in name_from_type.items()})
    if POUR:
        final_percentages = {name: mass / init_total_mass for name, mass in final_masses.items()}
        print('Percentages (%):', str_from_object(final_percentages))
    #score.update({'fraction_in_{}'.format(ty): final_percentages[name]
    #             for ty, name in name_from_type.items()})
    if task.holding:
        [spoon_name] = task.holding
        spoon_capacity = SPOON_CAPACITIES[get_type(spoon_name), args.material]
        fraction_filled = final_masses[cup_name] / spoon_capacity
        print('Spoon percentage filled (%):', fraction_filled)
        score.update({
            'mass_in_spoon': final_masses[cup_name],
            #'fraction_in_spoon': final_masses[cup_name] / init_total_mass,
            #'spoon_capacity': spoon_capacity,
            #'fraction_filled': fraction_filled,
        })
    return score

##################################################

def add_scales(task, ros_world):
    scale_stackings = {}
    holding = {grasp.obj_name for grasp in task.init_holding.values()}
    with ClientSaver(ros_world.client):
        perception = ros_world.perception
        items = sorted(set(perception.get_items()) - holding,
                       key=lambda n: get_point(ros_world.get_body(n))[1], reverse=False) # Right to left
        for i, item in enumerate(items):
            if not POUR and (get_type(item) != SCOOP_CUP):
                continue
            item_body = ros_world.get_body(item)
            scale = create_name(SCALE_TYPE, i + 1)
            with HideOutput():
                scale_body = load_pybullet(get_body_urdf(get_type(scale)), fixed_base=True)
            ros_world.perception.sim_bodies[scale] = scale_body
            ros_world.perception.sim_items[scale] = None
            item_z = stable_z(item_body, scale_body)
            scale_pose_item = Pose(point=Point(z=-item_z)) # TODO: relies on origin in base
            set_pose(scale_body, multiply(get_pose(item_body), scale_pose_item))
            roll, pitch, _ = get_euler(scale_body)
            set_euler(scale_body, [roll, pitch, 0])
            scale_stackings[scale] = item
        #wait_for_user()
    return scale_stackings

def add_walls(ros_world):
    thickness = 0.01  # 0.005 | 0.01 | 0.02
    height = 0.11  # 0.11 | 0.12
    [table_name] = ros_world.perception.get_surfaces()
    #table_body = ros_world.get_body(table_name)
    table_info = ros_world.perception.info_from_name[table_name]
    #draw_pose(table_info.pose, length=1)
    with ros_world:
        #aabb = aabb_from_points(apply_affine(table_info.pose, table_info.type.vertices))
        aabb = aabb_from_points(table_info.type.vertices)
        draw_aabb(aabb)
        # pose = get_pose(table_body)
        # pose = ros_world.get_pose(table_name)
        # aabb = approximate_as_prism(table_body, pose)
        x, y, z = get_aabb_center(aabb)
        l, w, h = get_aabb_extent(aabb)

        #right_wall = create_box(l, thickness, height)
        #set_pose(right_wall, multiply(table_info.pose, (Pose(point=[x, y - (w + thickness) / 2., z + (h + height) / 2.]))))

        #bottom_wall = create_box(thickness, w / 2., height)
        #set_point(bottom_wall, [x - (l + thickness) / 2., y - w / 4., z + (h + height) / 2.])

        top_wall = create_box(thickness, w, height)
        set_pose(top_wall, multiply(table_info.pose, (Pose(point=[x + (l + thickness) / 2., y, z + (h + height) / 2.]))))

        walls = [top_wall]
        #walls = [right_wall, top_wall] # , bottom_wall]
        for i, body in enumerate(walls):
            name = create_name('wall', i + 1)
            ros_world.perception.sim_bodies[name] = body
            ros_world.perception.sim_items[name] = None

    #wait_for_user()
    return walls # Return names?

##################################################

def run_loop(args, ros_world, policy):
    items = wait_until_observation(args.problem, ros_world)
    if items is None:
        ros_world.controller.speak("Observation failure")
        print('Failed to detect the required objects')
        return None

    task_fn = get_task_fn(PROBLEMS, 'collect_{}'.format(args.problem))
    task, feature = task_fn(args, ros_world)
    feature['simulation'] = False
    #print('Arms:', task.arms)
    #print('Required:', task.required)

    init_stackings = add_scales(task, ros_world) if task.use_scales else {}
    print('Scale stackings:', str_from_object(init_stackings))
    print('Surfaces:', ros_world.perception.get_surfaces())
    print('Items:', ros_world.perception.get_items())
    add_walls(ros_world)

    with ros_world:
        remove_all_debug()
    draw_names(ros_world)
    draw_forward_reachability(ros_world, task.arms)

    #########################

    init_raw_masses = read_raw_masses(init_stackings)
    if init_raw_masses is None:
        ros_world.controller.speak("Scale failure")
        return None
    # Should be robust to material starting in the target
    initial_contains = bowls_holding_material(task.init)
    print('Initial contains:', initial_contains)
    #bowl_masses = {bowl: MODEL_MASSES[get_type(bowl)] if bowl in initial_contains else init_raw_masses[bowl]
    #               for bowl in init_stackings.values()}
    bowl_masses = {bowl: MODEL_MASSES[get_type(bowl)] for bowl in init_stackings.values()}

    print('Material:', args.material)
    init_masses = estimate_masses(init_raw_masses, bowl_masses, args.material)
    init_total_mass = sum(init_masses.values())
    print('Total mass:', init_total_mass)
    if POUR and (init_total_mass < 1e-3):
       ros_world.controller.speak("Material failure")
       print('Initial total mass is {:.3f}'.format(init_total_mass))
       return None
    # for bowl, mass in init_masses.items():
    #     if bowl in initial_contains:
    #         if mass < 0:
    #             return None
    #     else:
    #         if 0 <= mass:
    #             return None

    #########################

    policy_name = policy if isinstance(policy, str) else ''
    print('Feature:', feature)
    collector = SKILL_COLLECTORS[args.problem]
    parameter_fns, parameter, prediction = get_parameter_fns(ros_world, collector, policy, feature, valid=True) # valid=not args.active)
    print('Policy:', policy_name)
    print('Parameter:', parameter)
    print('Prediction:', prediction)

    result = {
        SKILL: args.problem,
        'date': datetime.datetime.now().strftime(DATE_FORMAT),
        'simulated': False,
        'material': args.material, # TODO: pass to feature
        'dataset': 'train' if args.train else 'test',
        'feature': feature,
        'parameter': parameter,
        'score': None,
    }

    plan = None
    if test_validity(result, ros_world, collector, feature, parameter, prediction):
        planning_world = PlanningWorld(task, visualize=args.visualize_planning)
        with ClientSaver(planning_world.client):
            #set_caching(False)
            planning_world.load(ros_world)
            print(planning_world)
            start_time = time.time()
            saver = WorldSaver()
            plan = None
            while (plan is None) and (elapsed_time(start_time) < 45): # Doesn't typically timeout
               plan = plan_actions(planning_world, parameter_fns=parameter_fns, collisions=True, max_time=30)
               saver.restore()
        planning_world.stop()
        result.update({
            'success': plan is not None,
            'plan-time': elapsed_time(start_time), # TODO: should be elapsed time
        })

    if plan is None:
        print('Failed to find a plan')
        ros_world.controller.speak("Planning failure")
        return result

    #########################

    #if not review_plan(task, ros_world, plan):
    #    return
    start_time = time.time()
    ros_world.controller.speak("Executing")
    success = execute_plan(ros_world, plan, joint_speed=0.5, default_sleep=0.25)
    print('Finished! Success:', success)
    result['execution'] = success
    result['execution-time'] = elapsed_time(start_time)
    if not success:
        ros_world.controller.speak("Execution failure")
        return None

    #########################

    # TODO: could read scales while returning home
    final_raw_masses = read_raw_masses(init_stackings)
    if final_raw_masses is None:
        ros_world.controller.speak("Scale failure")
        return None
    result['score'] = {}
    result['score'].update(score_masses(args, task, ros_world, init_total_mass, bowl_masses, final_raw_masses))
    result['score'].update(score_poses(args.problem, task, ros_world)) # TODO: fail if fail to detect

    result['score'].update({'initial_{}_mass'.format(name): mass for name, mass in init_raw_masses.items()})
    result['score'].update({'final_{}_mass'.format(name): mass for name, mass in final_raw_masses.items()})

    ros_world.controller.speak("Done")
    return result

##################################################

def format_class(cls):
    return cls.replace('3D', '_3D').replace('_', ' ')

# http://wiki.ros.org/video_stream_opencv
# roslaunch lis_pr2_pkg record_video.launch
# roslaunch lis_pr2_pkg record_video.launch filename:=/home/ari/boop.avi
#    <!--node pkg="rosbag" type="record" name="rosbag_record" args="-a" if="$(arg rosbag)" /-->
#    <node pkg="rosbag" type="record" name="rosbag_record" args="-o mudfish /tf /tf_static /head_mount_kinect/rgb/image_rect_color /tensorflow_detector/image
#        /mudfish_node/markers /mudfish_node/full_tables /mudfish_node/tables" if="$(arg rosbag)" />

def launch_kinect():
    record = rosnode.rosnode_ping("/camera/camera_nodelet_manager", 1e-2)
    print('Record:', record)
    if not record:
        return None
    # TODO: set arguments
    launch_path = os.path.abspath('utils/record_video.launch')
    #video_path = os.path.abspath(os.path.join(DATA_DIRECTORY, 'video2.avi'))
    #cli_args = [launch_path, 'filename:={}'.format(video_path)]
    #roslaunch_args = roslaunch.rlutil.resolve_launch_arguments(cli_args)
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_path])
    #launch = roslaunch.parent.ROSLaunchParent(uuid, [(roslaunch_args[0], cli_args[1:])])
    launch.start()
    return launch

##################################################

def update_learner(learner, learner_path, result):
    # Discard if a planning failure?
    if learner.validity_learner is not None:
        X_online, Y_online, _ = learner.validity_learner.func.example_from_result(result, validity=True)
        learner.validity_learner.retrain(newx=np.array([X_online]), newy=np.array([Y_online]))
        learner.validity_learner.results.append(result)
    #if result['score'] is not None:
    X_online, Y_online, W_online = learner.func.example_from_result(result, validity=False, binary=False)
    print('Score: {:.3f} | Weight: {:.3f}'.format(Y_online, W_online))
    learner.retrain(newx=np.array([X_online]), newy=np.array([Y_online]), new_w=np.array([W_online]))
    learner.results.append(result)
    # policy.save(TBD)
    write_pickle(learner_path, learner)
    print('Saved', learner_path)

##################################################

def create_active_generator(args, policy):
    assert args.train
    assert isinstance(policy, ActiveLearner)
    while True:
        object_types = optimize_feature(learner=policy, bowls=TRAIN_BOWLS, cups=TRAIN_CUPS, spoons=TRAIN_SPOONS)
        if args.problem == 'pour':
            yield object_types['cup_type'], object_types['bowl_type']
        elif args.problem == 'scoop':
            yield object_types['bowl_type'], object_types['spoon_type']
        else:
            raise NotImplementedError(args.problem)
        # TODO: return the parameter that was used
        # TODO: keep generating examples until one feasible

def create_random_generator(args):
    # Left to right
    if args.problem == 'pour':
        cups, bowls = (TRAIN_CUPS, TRAIN_BOWLS) if args.train else (TEST_CUPS, TEST_BOWLS)
        print('Cups ({}): {}'.format(len(cups), cups))
        print('Bowls ({}) {}:'.format(len(bowls), bowls))
        combinations = randomize(product(cups, bowls))
        if not args.train:
            combinations = POUR_TEST_PAIRS
    elif args.problem == 'scoop':
        bowls = TRAIN_BOWLS if args.train else TEST_BOWLS
        cups = [SCOOP_CUP]
        combinations = randomize(product(bowls, cups))
        #assert args.train
        if not args.train:
            combinations = list(product(SCOOP_TEST, cups))
    else:
        raise NotImplementedError(args.problem)
    print('Combinations ({})'.format(len(combinations)))
    #print(randomize(combinations))
    generator = cycle(combinations)
    #generator = (random.choice(combinations) for _ in inf_generator())
    return generator

##################################################

# TODO: retract at some angle before moving spoon up

ACTIVE_FEATURE = False
ATTACH_TIME = 5.0

def main():
    """
    ./home/demo/catkin_percep/collect_scales.sh (includes scale offset)
    Make sure to start the scales with nothing on them (for calibration)
    """
    assert(get_python_version() == 2) # ariadne has ROS with python2
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--active', action='store_true',
                        help='Uses active learning queries.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Disables saving during debugging.')
    parser.add_argument('-f', '--fn', type=str, default=TRAINING, # DESIGNED | TRAINING
                        help='The name of or the path to the policy that generates parameters.')
    parser.add_argument('-m', '--material', required=True, choices=sorted(MATERIALS),
                        help='The name of the material being used.')
    parser.add_argument('-p', '--problem', required=True, choices=sorted(REQUIREMENT_FNS.keys()),
                        help='The name of the skill to learn.')
    parser.add_argument('-s', '--spoon', default=None, choices=SPOONS,
                        help='The name of the spoon being used.')
    parser.add_argument('-r', '--train', action='store_true',
                        help='When enabled, uses the training dataset.')
    parser.add_argument('-v', '--visualize_planning', action='store_true',
                        help='When enabled, visualizes planning rather than the world (for debugging).')
    args = parser.parse_args()
    # TODO: toggle material default based on task

    # TODO: label material on the image
    assert args.material in MODEL_MASSES
    print('Policy:', args.fn)
    assert implies(args.problem in ['scoop'], args.spoon is not None)
    assert implies(args.active, args.train)

    ros_world = ROSWorld(sim_only=False, visualize=not args.visualize_planning)
    classes_pub = rospy.Publisher('~collect_classes', String, queue_size=1)
    with ros_world:
        set_camera_pose(np.array([1.5, -0.5, 1.5]), target_point=np.array([0.75, 0, 0.75]))

    arm, is_open = ACTIVE_ARMS[args.problem]
    open_grippers = {arm: is_open}
    if args.problem == 'scoop':
        ros_world.controller.open_gripper(get_arm_prefix(arm), blocking=True)
        ros_world.controller.speak("{:.0f} seconds to attach {}".format(ATTACH_TIME, format_class(args.spoon)))
        rospy.sleep(ATTACH_TIME) # Sleep to have time to set the spoon
    move_to_initial_config(ros_world, open_grippers) #get_other_arm
    # TODO: cross validation for measuring performance across a few bowls

    launch = launch_kinect()
    video_time = time.time()

    if args.debug:
        ros_world.controller.speak("Warning! Data will not be saved.")
        time.sleep(1.0)
        data_path = None
    else:
        # TODO: only create directory if examples made
        data_path = get_data_path(args.problem, real=True)

    # TODO: log camera image after the pour
    policy = args.fn
    learner_path = None
    test_data = None
    if isinstance(policy, str) and os.path.isfile(policy):
        policy = read_pickle(policy)
        assert isinstance(policy, ActiveLearner)
        print(policy)
        print(policy.algorithm)
        #policy.transfer_weight = 0

        # print(policy.xx.shape)
        # policy.results = policy.results[:-1]
        # policy.xx = policy.xx[:-1]
        # policy.yy = policy.yy[:-1]
        # policy.weights = policy.weights[:-1]
        # print(policy.xx.shape)
        # write_pickle(args.fn, policy)
        # print('Saved', args.fn)

        if args.active:
            # policy.retrain()
            test_domain = load_data(SCOOP_TEST_DATASETS, verbose=False)
            test_data = test_domain.create_dataset(include_none=False, binary=False)
            policy.query_type = STRADDLE # VARIANCE
            #policy.weights = 0.1*np.ones(policy.yy.shape) # TODO: make this multiplicative
            #policy.retrain()
            evaluate_confusions(test_data, policy)
        else:
            policy.query_type = BEST
        ensure_dir(LEARNER_DIRECTORY)
        date_name = datetime.datetime.now().strftime(DATE_FORMAT)
        filename = '{}_{}.pk{}'.format(get_label(policy.algorithm), date_name, get_python_version())
        learner_path = os.path.join(LEARNER_DIRECTORY, filename)

    if ACTIVE_FEATURE and args.active:
        assert isinstance(policy, ActiveLearner)
        generator = create_active_generator(args, policy)
    else:
        generator = create_random_generator(args)
    pair = next(generator)
    print('Next pair:', pair)
    classes_pub.publish('{},{}'.format(*pair))
    for phrase in map(format_class, pair):
        ros_world.controller.speak(phrase)
    wait_for_user('Press enter to begin')

    # TODO: change the name of the directory after additional samples
    results = []
    num_trials = num_failures = num_scored = 0
    while True:
        start_time = elapsed_time(video_time)
        result = run_loop(args, ros_world, policy)
        print('Result:', str_from_object(result))
        print('{}\nTrials: {} | Successes: {} | Failures: {} | Time: {:.3f}'.format(
            SEPARATOR, num_trials, len(results), num_failures, elapsed_time(video_time)))
        num_trials += 1
        if result is None: # TODO: result['execution']
            num_failures += 1
            print('Error! Trial resulted in an exception')
            move_to_initial_config(ros_world, open_grippers)
            continue

        end_time = elapsed_time(video_time)
        print('Elapsed time:', end_time - start_time)
        # TODO: record the type of failure (planning, execution, etc...)
        scored = result['score'] is not None
        num_scored += scored
        # TODO: print the score

        if isinstance(policy, ActiveLearner) and args.active: # and scored:
            update_learner(policy, learner_path, result)
            evaluate_confusions(test_data, policy)
            # TODO: how to handle failures that require bad annotations?

        pair = next(generator)
        print('Next pair:', pair)
        classes_pub.publish('{},{}'.format(*pair))
        for phrase in map(format_class, pair):
            ros_world.controller.speak(phrase)

        annotation = wait_for_user('Enter annotation and press enter to continue: ')
        result.update({
            # TODO: record the query_type
            'policy': args.fn,
            'active_feature': ACTIVE_FEATURE,
            'trial': num_trials,
            'start_time': start_time,
            'end_time': end_time,
            'annotation': annotation,
        })
        results.append(result)
        if data_path is not None:
            write_results(data_path, results)
        #if annotation in ['q', 'quit']: # TODO: Ctrl-C to quit
        #    break

    ros_world.controller.speak("Finished")
    if launch is not None:
        launch.shutdown()
    print('Total time:', elapsed_time(video_time))


if __name__ == '__main__':
    main()

# (nish)demo@ariadne:~/catkin_wsd/src/lis_ltamp$ python2 -m learn_tools.collect_pr2 -p pour -f data/pour_19-04-27_13-54-07/gp_batch_mlp.pk2

# Need to use python2 because we cannot install python2 and python3 using ROS
# (nish)demo@ariadne:~/catkin_wsd/src/lis_ltamp$ python2 -m learn_tools.run_active data/pour_19-04-30_11-40-21/trials_n\=20000.json -r 1 -n 0 -s

# python2 -m learn_tools.collect_pr2 -p pour -f data/pour_19-04-27_13-54-07/gp_batch_mlp.pk2 -m red_men
# python2 -m learn_tools.collect_pr2 -p scoop -f data/scoop_19-04-24_18-18-35/gp_batch_mlp.pk2 -m red_men

# Run the following after starting up roslaunch lis_pr2_pkg pr2_launch.launch
# demo@ariadne:~/$ roslaunch openni_launch openni.launch