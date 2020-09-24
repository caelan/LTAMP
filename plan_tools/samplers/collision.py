import random
from itertools import product

from plan_tools.common import COLLISION_BUFFER
from pybullet_tools.pr2_utils import get_arm_joints
from pybullet_tools.utils import set_pose, pairwise_collision, get_moving_links, set_joint_positions, \
    pairwise_link_collision, get_all_links, any_link_pair_collision

def body_pair_collision(body1, body2, collision_buffer=COLLISION_BUFFER):
    if body1 == body2:
        return False
    return pairwise_collision(body1, body2, max_distance=collision_buffer)

def cartesian_path_collision(body, path, obstacles, **kwargs):
    for pose in path:
        set_pose(body, pose)
        if any(body_pair_collision(body, obst, **kwargs) for obst in obstacles):
            return True
    return False

def link_pairs_collision(body1, links1, body2, links2=None, collision_buffer=COLLISION_BUFFER):
    return any_link_pair_collision(body1, links1, body2, links2, max_distance=collision_buffer)

##################################################

def get_pose_pose_collision_test(world, collisions=True):
    collision_buffer = 0.0 # For stacking
    #collision_buffer = COLLISION_BUFFER

    def test(obj1, pose1, obj2, pose2):
        if not collisions or (obj1 == obj2):
            return False
        body1 = world.bodies[obj1]
        set_pose(body1, pose1)
        body2 = world.bodies[obj2]
        set_pose(body2, pose2)
        return body_pair_collision(body1, body2, collision_buffer=collision_buffer)
    return test

def get_conf_conf_collision_test(world, collisions=True):
    robot = world.robot
    def test(arm1, conf1, arm2, conf2):
        # TODO: don't let the arms get too close
        if not collisions or (arm1 == arm2):
            return False
        arm1_joints = get_arm_joints(robot, arm1)
        set_joint_positions(robot, arm1_joints, conf1)
        arm2_joints = get_arm_joints(robot, arm2)
        set_joint_positions(robot, arm2_joints, conf2)
        return link_pairs_collision(robot, get_moving_links(robot, arm1_joints),
                                    robot, get_moving_links(robot, arm2_joints))
    return test

##################################################

def get_control_pose_collision_test(world, collisions=True):
    collision_buffer = 0.0 # For stacking
    #collision_buffer = COLLISION_BUFFER
    robot = world.robot

    def test(arm, control, obj, pose):
        if not collisions or (obj in control['objects']):
            return False
        attachments = control['context'].assign()
        body = world.bodies[obj]
        set_pose(body, pose)

        arm_links = get_moving_links(robot, get_arm_joints(robot, arm))
        for command in control['commands']:
            for _ in command.iterate(world, attachments): # TODO: randomize sequence
                for attachment in attachments.values():
                    attachment.assign()
                    if body_pair_collision(attachment.child, body, collision_buffer=collision_buffer):
                        return True
                if link_pairs_collision(robot, arm_links, body):
                    return True
        return False
    return test

def get_control_conf_collision_test(world, collisions=True):
    robot = world.robot
    def test(arm1, control, arm2, conf2):
        if not collisions or (arm1 == arm2):
            return False
        attachments = control['context'].assign()
        arm1_links = get_moving_links(robot, get_arm_joints(robot, arm1))

        arm2_joints = get_arm_joints(robot, arm2)
        arm2_links = get_moving_links(robot, arm2_joints)
        set_joint_positions(robot, arm2_joints, conf2)
        for command in control['commands']:
            for _ in command.iterate(world, attachments): # TODO: randomize sequence
                for attachment in attachments.values():
                    attachment.assign()
                    if link_pairs_collision(robot, arm2_links, attachment.child):
                        return True
                if link_pairs_collision(robot, arm1_links, robot, arm2_links):
                    return True
        return False
    return test
