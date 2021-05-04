import numpy as np
import argparse
import sys
import os
import random
import scipy
import colorsys

from itertools import islice
import matplotlib.pyplot as plt

sys.path.extend([
    os.path.join(os.getcwd(), 'pddlstream'), # Important to use absolute path when doing chdir
    os.path.join(os.getcwd(), 'ss-pybullet'),
])

from plan_tools.planner import PlanningWorld
from pybullet_tools.utils import set_pose, wait_for_user, set_camera_pose, draw_pose, get_pose, load_pybullet, \
    add_data_path, remove_body, get_bodies, set_color, apply_alpha, BLACK, RED, GREEN, spaced_colors, clip, multiply, \
    Pose, Euler, invert, unit_pose, get_image, get_camera, take_picture, save_image, wait_for_duration, read_pickle, \
    clip, create_box, create_cylinder, set_point, Point, add_line, tform_point, LockRenderer, approximate_as_prism, \
    WHITE, draw_point, GREY, randomize, INF, create_plane, safe_zip, point_from_pose, remove_handles
from plan_tools.samplers.pour import get_pour_feature, pour_path_from_parameter, is_valid_pour, get_bowl_from_pivot
from learn_tools.run_active import TRAIN_DATASETS, ACTIVE_DATASETS, TEST_DATASETS, BEST_DATASETS
from learn_tools.active_learner import BEST, STRADDLE, REJECTION, tail_confidence
from learn_tools.active_gp import get_parameters
from learn_tools.learner import FAILURE, SUCCESS, DEFAULT_INTERVAL, rescale, inclusive_range
from learn_tools.learnable_skill import read_data
#from learn_tools.collect_simulation import SKILL_COLLECTORS
from learn_tools.collectors.collect_pour import compute_fraction_filled, pour_score
from dimensions.common import load_cup_bowl

#BACKGROUND_COLOR = WHITE
BACKGROUND_COLOR = [0.9, 0.9, 1.0] # 229, 229, 255
GROUND_COLOR = 0.8*np.ones(3)

def create_world():
    world = PlanningWorld(task=None, use_robot=False, visualize=True, color=BACKGROUND_COLOR, shadows=False)
    add_data_path()
    environment = [
        #load_pybullet('models/short_floor.urdf'),
        #load_pybullet('plane.urdf', fixed_base=True),
        create_plane(color=GROUND_COLOR),
    ]
    set_camera_pose(camera_point=0.4*np.array([0, -1.0, 0.5]))
    #draw_pose(unit_pose(), length=1)
    return world

##################################################

def interpolate_hue(fraction, min_hue=0./3, max_hue=2./3):
    hue = (1 - fraction) * min_hue + fraction * max_hue
    return colorsys.hsv_to_rgb(hue, s=1, v=1)

def visualize_collected_pours(paths, num=6, save=True):
    result_from_bowl = {}
    for path in randomize(paths):
        results = read_data(path)
        print(path, len(results))
        for result in results:
            result_from_bowl.setdefault(result['feature']['bowl_type'], []).append(result)

    world = create_world()
    environment = get_bodies()
    #collector = SKILL_COLLECTORS['pour']
    print(get_camera())

    for bowl_type in sorted(result_from_bowl):
        # TODO: visualize same
        for body in get_bodies():
            if body not in environment:
                remove_body(body)
        print('Bowl type:', bowl_type)
        bowl_body = load_cup_bowl(bowl_type)
        bowl_pose = get_pose(bowl_body)

        results = result_from_bowl[bowl_type]
        results = randomize(results)

        score_cup_pose = []
        for i, result in enumerate(results):
            if num <= len(score_cup_pose):
                break
            feature = result['feature']
            parameter = result['parameter']
            score = result['score']
            if (score is None) or not result['execution'] or result['annotation']:
                continue
            cup_body = load_cup_bowl(feature['cup_type'])
            world.bodies[feature['bowl_name']] = bowl_body
            world.bodies[feature['cup_name']] = cup_body
            fraction = compute_fraction_filled(score)
            #value = collector.score_fn(feature, parameter, score)
            value = pour_score(feature, parameter, score)
            print(i, feature['cup_type'], fraction, value)
            path, _ = pour_path_from_parameter(world, feature, parameter)
            sort = fraction
            #sort = parameter['pitch']
            #sort = parameter['axis_in_bowl_z']
            score_cup_pose.append((sort, fraction, value, cup_body, path[0]))

        #order = score_cup_pose
        order = randomize(score_cup_pose)
        #order = sorted(score_cup_pose)
        angles = np.linspace(0, 2*np.pi, num=len(score_cup_pose), endpoint=False) # Halton
        for angle, (_, fraction, value, cup_body, pose) in zip(angles, order):
            #fraction = clip(fraction, min_value=0, max_value=1)
            value = clip(value, *DEFAULT_INTERVAL)
            fraction = rescale(value, DEFAULT_INTERVAL, new_interval=(0, 1))
            #color = (1 - fraction) * np.array(RED) + fraction * np.array(GREEN)
            color = interpolate_hue(fraction)
            set_color(cup_body, apply_alpha(color, alpha=0.25))
            #angle = random.uniform(-np.pi, np.pi)
            #angle = 0
            rotate_bowl = Pose(euler=Euler(yaw=angle))
            cup_pose = multiply(bowl_pose, invert(rotate_bowl), pose)
            set_pose(cup_body, cup_pose)
            #wait_for_user()
            #remove_body(cup_body)

        if save:
            #filename = os.path.join('workspace', '{}.png'.format(bowl_type))
            #save_image(filename, take_picture())  # [0, 255]
            #wait_for_duration(duration=0.5)
            # TODO: get window location
            #os.system("screencapture -R {},{},{},{} {}".format(
            #    275, 275, 500, 500, filename)) # -R<x,y,w,h> capture screen rect
            wait_for_user()
        remove_body(bowl_body)

##################################################

def add_obstacles():
    ceiling = create_box(w=0.2, l=0.2, h=0.01, color=RED)
    set_point(ceiling, Point(z=0.2))
    pole = create_cylinder(radius=0.025, height=0.1, color=RED)
    set_point(pole, Point(z=0.15))

def visualize_learned_pours(learner, bowl_type='blue_bowl', cup_type='red_cup', num_steps=None):
    learner.query_type = REJECTION # BEST | REJECTION | STRADDLE
    learner.validity_learner = None

    world = create_world()
    #add_obstacles()
    #environment = get_bodies()

    world.bodies[bowl_type] = load_cup_bowl(bowl_type)
    world.bodies[cup_type] = load_cup_bowl(cup_type)
    feature = get_pour_feature(world, bowl_type, cup_type)
    draw_pose(get_pose(world.get_body(bowl_type)), length=0.2)
    # TODO: sample different sizes

    print('Feature:', feature)
    for parameter in learner.parameter_generator(world, feature, valid=False):
        handles = []
        print('Parameter:', parameter)
        valid = is_valid_pour(world, feature, parameter)
        score = learner.score(feature, parameter)
        x = learner.func.x_from_feature_parameter(feature, parameter)
        [prediction] = learner.predict_x(np.array([x]), noise=True)
        print('Valid: {} | Mean: {:.3f} | Var: {:.3f} | Score: {:.3f}'.format(
            valid, prediction['mean'], prediction['variance'], score))

        #fraction = rescale(clip(prediction['mean'], *DEFAULT_INTERVAL), DEFAULT_INTERVAL, new_interval=(0, 1))
        #color = (1 - fraction) * np.array(RED) + fraction * np.array(GREEN)
        #set_color(world.get_body(cup_type), apply_alpha(color, alpha=0.25))
        # TODO: overlay many cups at different poses
        bowl_from_pivot = get_bowl_from_pivot(world, feature, parameter)
        #handles.extend(draw_pose(bowl_from_pivot))
        handles.extend(draw_point(point_from_pose(bowl_from_pivot), color=BLACK))

        # TODO: could label bowl, cup dimensions
        path, _ = pour_path_from_parameter(world, feature, parameter, cup_yaw=0)
        for p1, p2 in safe_zip(path[:-1], path[1:]):
            handles.append(add_line(point_from_pose(p1), point_from_pose(p2), color=RED))

        if num_steps is not None:
            path = path[:num_steps]
        for i, pose in enumerate(path[::-1]):
            #step_handles = draw_pose(pose, length=0.1)
            #step_handles = draw_point(point_from_pose(pose), color=BLACK)
            set_pose(world.get_body(cup_type), pose)
            print('Step {}/{}'.format(i+1, len(path)))
            wait_for_user()
            #remove_handles(step_handles)
        remove_handles(handles)

##################################################

def visualize_vector_field(learner, bowl_type='blue_bowl', cup_type='red_cup',
                           num=500, delta=False, alpha=0.5):
    print(learner, len(learner.results))
    xx, yy, ww = learner.xx, learner.yy, learner.weights
    learner.hyperparameters = get_parameters(learner.model)
    print(learner.hyperparameters)
    learner.query_type = REJECTION  # BEST | REJECTION | STRADDLE

    world = create_world()
    world.bodies[bowl_type] = load_cup_bowl(bowl_type)
    world.bodies[cup_type] = load_cup_bowl(cup_type)
    feature = get_pour_feature(world, bowl_type, cup_type)
    set_point(world.bodies[cup_type], np.array([0.2, 0, 0]))

    # TODO: visualize training data as well
    # TODO: artificially limit training data to make a low-confidence region
    # TODO: visualize mean and var at the same time using intensity and hue
    print('Feature:', feature)
    with LockRenderer():
        #for parameter in islice(learner.parameter_generator(world, feature, valid=True), num):
        parameters = []
        while len(parameters) < num:
            parameter = learner.sample_parameter()
            # TODO: special color if invalid
            if is_valid_pour(world, feature, parameter):
                parameters.append(parameter)

    center, _ = approximate_as_prism(world.get_body(cup_type))
    bodies = []
    with LockRenderer():
        for parameter in parameters:
            path, _ = pour_path_from_parameter(world, feature, parameter)
            pose = path[0]
            body = create_cylinder(radius=0.001, height=0.02, color=apply_alpha(GREY, alpha))
            set_pose(body, multiply(pose, Pose(point=center)))
            bodies.append(body)

    #train_sizes = inclusive_range(10, 100, 1)
    #train_sizes = inclusive_range(10, 100, 10)
    #train_sizes = inclusive_range(100, 1000, 100)
    #train_sizes = [1000]
    train_sizes = [None]

    # training_poses = []
    # for result in learner.results[:train_sizes[-1]]:
    #     path, _ = pour_path_from_parameter(world, feature, result['parameter'])
    #     pose = path[0]
    #     training_poses.append(pose)

    # TODO: draw the example as well
    scores_per_size = []
    for train_size in train_sizes:
        if train_size is not None:
            learner.xx, learner.yy, learner.weights = xx[:train_size], yy[:train_size],  ww[:train_size]
            learner.retrain()
        X = np.array([learner.func.x_from_feature_parameter(feature, parameter) for parameter in parameters])
        predictions = learner.predict_x(X, noise=False) # Noise is just a constant
        scores_per_size.append([prediction['mean'] for prediction in predictions]) # mean | variance
        #scores_per_size.append([1./np.sqrt(prediction['variance']) for prediction in predictions])
        #scores_per_size.append([prediction['mean'] / np.sqrt(prediction['variance']) for prediction in predictions])
        #plt.hist(scores_per_size[-1])
        #plt.show()
        #scores_per_size.append([scipy.stats.norm.interval(alpha=0.95, loc=prediction['mean'],
        #                                                  scale=np.sqrt(prediction['variance']))[0]
        #                         for prediction in predictions]) # mean | variance
        # score = learner.score(feature, parameter)

    if delta:
        scores_per_size = [np.array(s2) - np.array(s1) for s1, s2 in zip(scores_per_size, scores_per_size[1:])]
        train_sizes = train_sizes[1:]

    all_scores = [score for scores in scores_per_size for score in scores]
    #interval = (min(all_scores), max(all_scores))
    interval = scipy.stats.norm.interval(alpha=0.95, loc=np.mean(all_scores), scale=np.std(all_scores))
    #interval = DEFAULT_INTERVAL
    print('Interval:', interval)
    print('Mean: {:.3f} | Stdev: {:.3f}'.format(np.mean(all_scores), np.std(all_scores)))

    #train_body = create_cylinder(radius=0.002, height=0.02, color=apply_alpha(BLACK, 1.0))
    for i, (train_size, scores) in enumerate(zip(train_sizes, scores_per_size)):
        print('Train size: {} | Average: {:.3f} | Min: {:.3f} | Max: {:.3f}'.format(
            train_size, np.mean(scores), min(scores), max(scores)))
        handles = []
        # TODO: visualize delta
        with LockRenderer():
            #if train_size is None:
            #    train_size = len(learner.results)
            #set_pose(train_body, multiply(training_poses[train_size-1], Pose(point=center)))
            #set_pose(world.bodies[cup_type], training_poses[train_size-1])
            for score, body in zip(scores, bodies):
                score = clip(score, *interval)
                fraction = rescale(score, interval, new_interval=(0, 1))
                color = interpolate_hue(fraction)
                #color = (1 - fraction) * np.array(RED) + fraction * np.array(GREEN) # TODO: convex combination
                set_color(body, apply_alpha(color, alpha))
                #set_pose(world.bodies[cup_type], pose)
                #draw_pose(pose, length=0.05)
                #handles.extend(draw_point(tform_point(pose, center), color=color))
                #extent = np.array([0, 0, 0.01])
                #start = tform_point(pose, center - extent/2)
                #end = tform_point(pose, center + extent/2)
                #handles.append(add_line(start, end, color=color, width=1))
        wait_for_duration(0.5)
        # TODO: test on boundaries
    wait_for_user()

##################################################

# python3 -m learn_tools.visualize_pours data/pour_19-11-22_13-08-55/gp_batch_k\=mlp_v\=true.pk3
# data/scoop_19-12-07_16-02-40/gp_batch_k=mlp_v=true.pk3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*', help='Paths to the data.')
    args = parser.parse_args()

    if not args.paths:
        visualize_collected_pours(paths=BEST_DATASETS) # TRAIN_DATASETS | ACTIVE_DATASETS | TEST_DATASETS | BEST_DATASETS
    else:
        [path] = args.paths
        learner = read_pickle(path)
        visualize_learned_pours(learner)
        #visualize_vector_field(learner)

if __name__ == '__main__':
    main()