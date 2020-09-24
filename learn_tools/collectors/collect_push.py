import numpy as np

from pddlstream.utils import find_unique, flatten
from pddlstream.algorithms.constraints import PlanConstraints, WILD as X

from perception_tools.common import create_name
from learn_tools.collectors.common import INFEASIBLE, get_contained_beads, stabilize, fill_with_beads, \
    InitialRanges, create_table_bodies, randomize_dynamics, sample_bead_parameters
from plan_tools.samplers.grasp import hold_item
from plan_tools.common import LEFT_ARM
from learn_tools.learner import Collector, PLANNING_FAILURE, SUCCESS, FAILURE, \
    DYNAMICS, SKILL
from plan_tools.planner import Task
from plan_tools.samplers.push import get_push_feature, get_push_gen_fn, PUSH_PARAMETER_FNS, PUSH_FEATURES
from pybullet_tools.utils import ClientSaver, get_distance, point_from_pose, \
    quat_angle_between, quat_from_pose, get_point, draw_point, add_segments
from plan_tools.simulated_problems import TABLE_NAME, update_world

POSE2D_RANGE = ([0.4, -0.1, -np.pi],
                [0.65, 0.3, np.pi])  # x, y, theta

def draw_push_goal(world, block_name, pos2d):
    block_z = get_point(world.get_body(block_name))[2]
    handles = draw_point(np.append(pos2d, [block_z]), size=0.05, color=(1, 0, 0))
    lower, upper = POSE2D_RANGE
    handles.extend(add_segments([
        (lower[0], lower[1], block_z),
        (upper[0], lower[1], block_z),
        (upper[0], upper[1], block_z),
        (lower[0], upper[1], block_z),
    ], closed=True))
    return handles

def collect_push(world):
    arm = LEFT_ARM
    block_name = create_name('purpleblock', 1)
    item_ranges = {
        block_name: InitialRanges(
            width_range=(0.75, 1.25),
            height_range=(0.75, 1.25),
            mass_range=(1., 1.),
            pose2d_range=POSE2D_RANGE,  # x, y, theta
        ),
    }

    ##################################################

    # TODO: check collisions/feasibility when sampling
    # TODO: grasps on the blue cup seem off for some reason...
    with ClientSaver(world.client):
        create_table_bodies(world, item_ranges)
        update_world(world, world.get_body(TABLE_NAME))
        parameters_from_name = randomize_dynamics(world)
    stabilize(world)

    lower, upper = item_ranges[block_name].pose2d_range
    goal_pos2d = np.random.uniform(lower, upper)[:2]
    draw_push_goal(world, block_name, goal_pos2d)

    initial_pose = world.perception.get_pose(block_name)
    feature = get_push_feature(world, arm, block_name, initial_pose, goal_pos2d)

    init = [('CanPush', block_name, goal_pos2d)]
    goal = [('InRegion', block_name, goal_pos2d)]
    skeleton = [
        ('move-arm', [arm, X, X, X]),
        ('push', [arm, block_name, X, X, X, X, X]),
    ]
    constraints = PlanConstraints(skeletons=[skeleton], exact=True)
    task = Task(init=init, goal=goal, arms=[arm], constraints=constraints)

    ##################################################

    def score_fn(plan):
        assert plan is not None
        initial_distance = get_distance(point_from_pose(initial_pose)[:2], goal_pos2d)
        final_pose = world.perception.get_pose(block_name)
        final_distance = get_distance(point_from_pose(final_pose)[:2], goal_pos2d)
        quat_distance = quat_angle_between(quat_from_pose(initial_pose), quat_from_pose(final_pose))
        print('Initial: {:.5f} m | Final: {:.5f} | Rotation: {:.5f} rads' .format(
            initial_distance, final_distance, quat_distance))
        # TODO: compare orientation to final predicted orientation
        # TODO: record simulation time in the event that the controller gets stuck

        score = {
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'rotation': quat_distance,
            DYNAMICS: parameters_from_name,
        }

        #_, args = find_unique(lambda a: a[0] == 'push', plan)
        #control = args[-1]
        return score

    return task, feature, score_fn

def push_score(feature, parameter, score, max_distance=0.025, max_rotation=0.1):
    if score is None:
        return PLANNING_FAILURE
    #return 2 - score['final_distance']
    if max_distance < score['final_distance']:
        return FAILURE
    if max_rotation < score['rotation']:
        return FAILURE
    return SUCCESS

is_valid_push = lambda w, f, p: True

PUSH_COLLECTOR = Collector(collect_push, get_push_gen_fn, PUSH_PARAMETER_FNS,
                           is_valid_push, PUSH_FEATURES, push_score)