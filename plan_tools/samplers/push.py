import random
import numpy as np
import math
from itertools import islice, cycle, count

from control_tools.execution import ArmTrajectory, Attach, Detach, Push
from perception_tools.common import get_type
from plan_tools.common import Control, Conf, ARMS, get_reference_pose, GRIPPER_LINKS, get_urdf_from_center
from plan_tools.samplers.collision import cartesian_path_collision, link_pairs_collision
from plan_tools.samplers.generators import TOOL_FRAMES, TOOL_POSE, solve_inverse_kinematics, \
    plan_waypoint_motion, get_pairwise_arm_links, set_gripper_position, Context, plan_waypoints_motion
from plan_tools.samplers.place import get_reachable_test
from learn_tools.learner import get_trial_parameter_fn, DESIGNED, CONSTANT, RANDOM, LEARNED, \
    sample_parameter, get_explore_parameter_fn

from pybullet_tools.pr2_utils import get_disabled_collisions, get_arm_joints, \
    get_gripper_joints
from pybullet_tools.utils import get_unit_vector, point_from_pose, quat_from_pose, get_yaw, multiply, \
    invert, set_pose, set_joint_positions, unit_quat, unit_from_theta, Pose, \
    interpolate_poses, is_center_stable, get_max_limit, BodySaver, approximate_as_prism, \
    get_length, Euler, unit_point, approximate_as_cylinder, link_from_name, get_link_pose, Point, \
    wait_for_user, get_link_subtree


# Maximum push distance from previous projects is about 0.3 meters
# TODO: push to the edge of a table to pick

PUSH_FEATURES = ['block_width', 'block_length', 'block_height', 'push_yaw', 'push_distance']

def get_end_pose(initial_pose, goal_pos2d):
    initial_z = point_from_pose(initial_pose)[2]
    orientation = quat_from_pose(initial_pose)
    goal_x, goal_y = goal_pos2d
    end_pose = ([goal_x, goal_y, initial_z], orientation)
    return end_pose

def get_push_feature(world, arm, block_name, initial_pose, goal_pos2d):
    block_body = world.get_body(block_name)
    block_reference = get_reference_pose(block_name)
    _, (block_w, block_l, block_h) = approximate_as_prism(block_body, body_pose=block_reference)
    goal_pose = get_end_pose(initial_pose, goal_pos2d)
    difference_initial =  point_from_pose(multiply(invert(initial_pose), goal_pose))

    feature = {
        'arm_name': arm,
        'block_name': block_name,
        'block_type': get_type(block_name),
        'block_width': block_w,
        'block_length': block_l,
        'block_height': block_h,

        'push_yaw': get_yaw(difference_initial),
        'push_distance': get_length(difference_initial)
    }
    return feature

PUSH_PARAMETER = {
    'gripper_z': 0.02,
    'gripper_tilt': np.pi/8,
    'delta_push': 0.0,
    'delta_yaw': 0.0,
}

PUSH_PARAMETER_RANGES = {
    #'gripper_z': (0.0, 0.1), # Relative to block bounding box base
    'gripper_z': (-0.05, 0.05),  # Relative to block bounding box center
    'gripper_tilt': (0.0, np.pi/4),
    'delta_push': (-0.05, 0.05), # Push shorter or farther
    #'delta_yaw': (-np.pi/4, np.pi/4), # Predict any change in block yaw
}

PUSH_PARAMETER_FNS = {
    DESIGNED: get_trial_parameter_fn(PUSH_PARAMETER), # Intentionally identical
    #CONSTANT: get_trial_parameter_fn(PUSH_PARAMETER),
    RANDOM: get_explore_parameter_fn(PUSH_PARAMETER_RANGES),
    #LEARNED: predict_pour_parameter,
}

##################################################

def sample_push_contact(world, feature, parameter, under=False):
    robot = world.robot
    arm = feature['arm_name']
    body = world.get_body(feature['block_name'])
    push_yaw = feature['push_yaw']

    center, (width, _, height) = approximate_as_prism(body, body_pose=Pose(euler=Euler(yaw=push_yaw)))
    max_backoff = width + 0.1 # TODO: add gripper bounding box
    tool_link = link_from_name(robot, TOOL_FRAMES[arm])
    tool_pose = get_link_pose(robot, tool_link)
    gripper_link = link_from_name(robot, GRIPPER_LINKS[arm])
    collision_links = get_link_subtree(robot, gripper_link)

    urdf_from_center = Pose(point=center)
    reverse_z = Pose(euler=Euler(pitch=math.pi))
    rotate_theta = Pose(euler=Euler(yaw=push_yaw))
    #translate_z = Pose(point=Point(z=-feature['block_height']/2. + parameter['gripper_z'])) # Relative to base
    translate_z = Pose(point=Point(z=parameter['gripper_z'])) # Relative to center
    tilt_gripper = Pose(euler=Euler(pitch=parameter['gripper_tilt']))

    grasps = []
    for i in range(1 + under):
        flip_gripper = Pose(euler=Euler(yaw=i * math.pi))
        for x in np.arange(0, max_backoff, step=0.01):
            translate_x = Pose(point=Point(x=-x))
            grasp_pose = multiply(flip_gripper, tilt_gripper, translate_x, translate_z, rotate_theta,
                                  reverse_z, invert(urdf_from_center))
            set_pose(body, multiply(tool_pose, TOOL_POSE, grasp_pose))
            if not link_pairs_collision(robot, collision_links, body, collision_buffer=0.):
                grasps.append(grasp_pose)
                break
    return grasps

def get_push_goal_gen_fn(world):
    reachable_test = get_reachable_test(world)
    def gen_fn(obj, pose1, surface, region=None):
        start_point = point_from_pose(pose1)
        distance_range = (0.15, 0.2) if region is None else (0.15, 0.3)
        obj_body = world.bodies[obj]
        while True:
            theta = random.uniform(-np.pi, np.pi)
            distance = random.uniform(*distance_range)
            end_point2d = np.array(start_point[:2]) + distance * unit_from_theta(theta)
            end_pose = (np.append(end_point2d, [start_point[2]]), quat_from_pose(pose1))
            set_pose(obj_body, end_pose)
            if not is_center_stable(obj_body, world.bodies[surface], above_epsilon=np.inf):
                continue
            if region is not None:
                assert region in ARMS
                if not reachable_test(region, obj, end_pose):
                    continue
            yield end_point2d,
    return gen_fn

def cartesian_path_unsupported(body, path, surface):
    for pose in path:
        set_pose(body, pose)
        if not is_center_stable(body, surface, above_epsilon=np.inf): # is_placement | is_center_stable # TODO: compute wrt origin
            return True
    return False

##################################################

def get_push_gen_fn(world, max_samples=25, max_attempts=10, collisions=True, parameter_fns={}, repeat=False):
    # TODO(caelan): could also simulate the predicated sample
    # TODO(caelan): make final the orientation be aligned with gripper
    parameter_fn = parameter_fns.get(get_push_gen_fn, PUSH_PARAMETER_FNS[DESIGNED])
    disabled_collisions = get_disabled_collisions(world.robot)
    disabled_collisions.update(get_pairwise_arm_links(world.robot, world.arms))
    obstacles = [world.bodies[obst] for obst in world.get_fixed()] if collisions else []
    backoff_distance = 0.03
    approach_tform = Pose(point=np.array([-0.1, 0, 0])) # Tool coordinates

    push_goal_gen_fn = get_push_goal_gen_fn(world)
    surface = world.get_table() # TODO: region?
    surface_body = world.bodies[surface]

    def gen_fn(arm, obj_name, pose1, region):
        # TODO: reachability test here
        if region is None:
            goals = push_goal_gen_fn(obj_name, pose1, surface)
        elif isinstance(region, str):
            goals = push_goal_gen_fn(obj_name, pose1, surface, region=region)
        else:
            goals = [(region,)]
        if repeat:
            goals = cycle(goals)

        arm_joints = get_arm_joints(world.robot, arm)
        open_width = get_max_limit(world.robot, get_gripper_joints(world.robot, arm)[0])
        body = world.bodies[obj_name]
        for goal_pos2d, in islice(goals, max_samples):
            pose2 = get_end_pose(pose1, goal_pos2d)
            body_path = list(interpolate_poses(pose1, pose2))
            if cartesian_path_collision(body, body_path, set(obstacles) - {surface_body}) or \
                    cartesian_path_unsupported(body, body_path, surface_body):
                continue
            set_pose(body, pose1)
            push_direction = np.array(point_from_pose(pose2)) - np.array(point_from_pose(pose1))
            backoff_tform = Pose(-backoff_distance * get_unit_vector(push_direction))  # World coordinates

            feature = get_push_feature(world, arm, obj_name, pose1, goal_pos2d)
            for parameter in parameter_fn(world, feature):
                push_contact = next(iter(sample_push_contact(world, feature, parameter, under=False)))
                gripper_path = [multiply(pose, invert(multiply(TOOL_POSE, push_contact))) for pose in body_path]
                set_gripper_position(world.robot, arm, open_width)
                for _ in range(max_attempts):
                    start_conf = solve_inverse_kinematics(world.robot, arm, gripper_path[0], obstacles=obstacles)
                    if start_conf is None:
                        continue
                    set_pose(body, pose1)
                    body_saver = BodySaver(body)
                    #attachment = create_attachment(world.robot, arm, body)
                    push_path = plan_waypoint_motion(world.robot, arm, gripper_path[-1],
                                                     obstacles=obstacles, #attachments=[attachment],
                                                     self_collisions=collisions, disabled_collisions=disabled_collisions)
                    if push_path is None:
                        continue
                    pre_backoff_pose = multiply(backoff_tform, gripper_path[0])
                    pre_approach_pose = multiply(pre_backoff_pose, approach_tform)
                    set_joint_positions(world.robot, arm_joints, push_path[0])
                    pre_path = plan_waypoints_motion(world.robot, arm, [pre_backoff_pose, pre_approach_pose],
                                                    obstacles=obstacles, attachments=[],
                                                    self_collisions=collisions, disabled_collisions=disabled_collisions)
                    if pre_path is None:
                        continue
                    pre_path = pre_path[::-1]
                    post_backoff_pose = multiply(backoff_tform, gripper_path[-1])
                    post_approach_pose = multiply(post_backoff_pose, approach_tform)
                    set_joint_positions(world.robot, arm_joints, push_path[-1])
                    post_path = plan_waypoints_motion(world.robot, arm, [post_backoff_pose, post_approach_pose],
                                                     obstacles=obstacles, attachments=[],
                                                     self_collisions=collisions, disabled_collisions=disabled_collisions)
                    if post_path is None:
                        continue
                    pre_conf = Conf(pre_path[0])
                    set_joint_positions(world.robot, arm_joints, pre_conf)
                    robot_saver = BodySaver(world.robot)
                    post_conf = Conf(post_path[-1])
                    control = Control({
                        'action': 'push',
                        'objects': [obj_name],
                        'feature': feature,
                        'parameter': None,
                        'context': Context(
                            savers=[robot_saver, body_saver],
                            attachments={}),
                        'commands': [
                            ArmTrajectory(arm, pre_path),
                            Push(arm, obj_name),
                            ArmTrajectory(arm, push_path),
                            Detach(arm, obj_name),
                            ArmTrajectory(arm, post_path),
                        ],
                    })
                    yield (pose2, pre_conf, post_conf, control)
                    break
    return gen_fn