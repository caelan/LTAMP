from itertools import combinations, product, islice

import numpy as np

from control_tools.ik.pr2_ik import get_arm_ik_generator
from control_tools.common import get_arm_prefix
from plan_tools.common import Conf, COLLISION_BUFFER, get_pr2_safety_limits, get_weights_resolutions
from plan_tools.samplers.collision import body_pair_collision, cartesian_path_collision
from pybullet_tools.ikfast.pr2.ik import BASE_FRAME, UPPER_JOINT as UPPER_JOINTS, IK_FRAME as TOOL_FRAMES
from pybullet_tools.pr2_utils import get_group_conf, get_arm_joints, learned_forward_generator, get_gripper_joints
from pybullet_tools.utils import set_joint_positions, set_pose, multiply, invert, get_pose, joint_from_name, \
    link_from_name, point_from_pose, quat_from_pose, Pose, \
    get_link_pose, plan_waypoints_joint_motion, plan_cartesian_motion, get_position_waypoints, movable_from_joints, \
    get_moving_links, get_custom_limits, CIRCULAR_LIMITS, get_collision_fn, unit_point, wait_for_user, \
    quat_from_euler, Euler, interpolate_poses, draw_pose, draw_aabb, get_aabb, remove_debug

TOOL_POSE = Pose(euler=Euler(pitch=np.pi/2)) # +x out of gripper arm

class Context(object):
    # Only need to encode information about moving bodies
    def __init__(self, savers=[], attachments={}):
        # TODO: I don't always have the robot in an appropriate configuration
        self.savers = tuple(savers)
        self.attachments = attachments
    def assign(self):
        for saver in self.savers:
            saver.restore()
        for attachment in self.attachments.values():
            attachment.assign()
        return dict(self.attachments)
    def apply_mapping(self, mapping):
        for saver in self.savers:
            saver.apply_mapping(mapping)
        for attachment in self.attachments.values():
            attachment.apply_mapping(mapping)

##################################################

def set_gripper_position(robot, arm, position):
    gripper_joints = get_gripper_joints(robot, arm)
    set_joint_positions(robot, gripper_joints, [position] * len(gripper_joints))

def compute_forward_reachability(robot, **kwargs):
    # TODO: compute wrt torso instead
    points = list(map(point_from_pose, learned_forward_generator(robot, get_pose(robot), **kwargs)))
    #for point in points:
    #    draw_point(point)
    #wait_for_interrupt()
    points2d = [point[:2] for point in points]
    #centriod = np.average(points, axis=0)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points2d)
    return [points2d[i] for i in hull.vertices]

def solve_inverse_kinematics(robot, arm, world_pose, obstacles=[], collision_buffer=COLLISION_BUFFER, max_attempts=25):
    arm_joints = get_arm_joints(robot, arm)
    custom_limits = get_pr2_safety_limits(robot) # TODO: self-collisions
    collision_fn = get_collision_fn(robot, arm_joints, obstacles=obstacles, attachments=[], max_distance=collision_buffer,
                                    self_collisions=False, disabled_collisions=set(), custom_limits=custom_limits)
    base_pose = get_link_pose(robot, link_from_name(robot, BASE_FRAME)) # != get_pose(robot)
    target_point, target_quat = multiply(invert(base_pose), world_pose)
    [torso_position] = get_group_conf(robot, 'torso')
    upper_joint = joint_from_name(robot, UPPER_JOINTS[arm])
    lower, upper = get_custom_limits(robot, [upper_joint], custom_limits, circular_limits=CIRCULAR_LIMITS)
    upper_limits = list(zip(lower, upper))[0]
    # TODO(caelan): just attempt one IK sample for each generator to restart quickly
    for i, arm_conf in enumerate(islice(get_arm_ik_generator(get_arm_prefix(arm), list(target_point), list(target_quat),
                                        torso_position, upper_limits=upper_limits), max_attempts)):
        if not collision_fn(arm_conf):
            #print('Attempts:', i)
            return Conf(arm_conf)
    return None

##################################################

def plan_workspace_motion(robot, arm, tool_path, collision_buffer=COLLISION_BUFFER, **kwargs):
    _, resolutions = get_weights_resolutions(robot, arm)
    tool_link = link_from_name(robot, TOOL_FRAMES[arm])
    arm_joints = get_arm_joints(robot, arm)
    arm_waypoints = plan_cartesian_motion(robot, arm_joints[0], tool_link, tool_path, pos_tolerance=5e-3)
    if arm_waypoints is None:
        return None
    arm_waypoints = [[conf[j] for j in movable_from_joints(robot, arm_joints)]
                     for conf in arm_waypoints]
    set_joint_positions(robot, arm_joints, arm_waypoints[0])
    return plan_waypoints_joint_motion(robot, arm_joints, arm_waypoints[1:], resolutions=resolutions,
                                       max_distance=collision_buffer, custom_limits=get_pr2_safety_limits(robot), **kwargs)

def interpolate_pose_waypoints(waypoints):
    if not waypoints:
        return waypoints
    pose_path = [waypoints[0]]
    for pose in waypoints[1:]:
        pose_path.extend(list(interpolate_poses(pose_path[-1], pose))[1:])
    return pose_path

def plan_waypoints_motion(robot, arm, tool_waypoints, **kwargs):
    tool_link = link_from_name(robot, TOOL_FRAMES[arm])
    tool_pose = get_link_pose(robot, tool_link)
    tool_path = interpolate_pose_waypoints([tool_pose] + list(tool_waypoints))
    return plan_workspace_motion(robot, arm, tool_path, **kwargs)

def plan_waypoint_motion(robot, arm, target_pose, **kwargs):
    return plan_waypoints_motion(robot, arm, [target_pose], **kwargs)

def plan_attachment_motion(robot, arm, attachment, attachment_path, obstacles, collision_buffer,
                           attachment_collisions=True, **kwargs):
    attachment_body = attachment.child
    if attachment_collisions and cartesian_path_collision(
            attachment_body, attachment_path, obstacles, collision_buffer=collision_buffer):
        return None
    tool_path = [multiply(p, invert(attachment.grasp_pose)) for p in attachment_path]
    grip_conf = solve_inverse_kinematics(robot, arm, tool_path[0], obstacles=obstacles)
    if grip_conf is None:
        return None
    attachments = [attachment] if attachment_collisions else []
    return plan_workspace_motion(robot, arm, tool_path, attachments=attachments,
                                 obstacles=obstacles, collision_buffer=collision_buffer, **kwargs)

##################################################

def get_pairwise_arm_links(robot, arms):
    disabled_collisions = set()
    arm_links = {}
    for arm in arms:
        arm_links[arm] = sorted(get_moving_links(robot, get_arm_joints(robot, arm)))
        #print(arm, [get_link_name(world.robot, link) for link in arm_links[arm]])
    for arm1, arm2 in combinations(arm_links, 2):
        disabled_collisions.update(product(arm_links[arm1], arm_links[arm2]))
    return disabled_collisions

def check_initial_collisions(world, obj_name, obst_names=[], **kwargs):
    obj_body = world.bodies[obj_name]
    for obst_name in obst_names:
        if obj_name == obst_name:
            continue
        obst_body = world.bodies[obst_name]
        set_pose(obst_body, world.initial_poses[obst_name])
        if body_pair_collision(obj_body, obst_body, **kwargs):
            #print(obj_name, obst_name)
            return True
    return False

def visualize_cartesian_path(body, pose_path):
    for i, pose in enumerate(pose_path):
        set_pose(body, pose)
        print('{}/{}) continue?'.format(i, len(pose_path)))
        wait_for_user()
    handles = draw_pose(get_pose(body))
    handles.extend(draw_aabb(get_aabb(body)))
    print('Finish?')
    wait_for_user()
    for h in handles:
        remove_debug(h)
