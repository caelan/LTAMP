import numpy as np

from control_tools.execution import ArmTrajectory
from plan_tools.common import Control, parse_fluents, get_liquid_quat, COLLISION_BUFFER, \
    get_pr2_safety_limits, get_weights_resolutions
from plan_tools.samplers.generators import set_gripper_position, Context
from plan_tools.samplers.grasp import get_grasp_attachment
from pybullet_tools.pr2_utils import get_arm_joints, get_disabled_collisions
from pybullet_tools.utils import set_pose, set_joint_positions, wait_for_user, \
    plan_joint_motion, unit_pose, get_sample_fn, get_distance_fn, get_extend_fn, get_collision_fn, get_pose, multiply, \
    euler_from_quat, quat_from_pose, get_joint_positions, check_initial_end, birrt, MAX_DISTANCE, unit_point, BodySaver, get_body_name

MAX_ROTATION = np.pi/6 # np.pi/6 | np.pi/5 | np.pi/4 | np.inf

##################################################

def plan_water_motion(body, joints, end_conf, attachment, obstacles=None, attachments=[],
                      self_collisions=True, disabled_collisions=set(), max_distance=MAX_DISTANCE,
                      weights=None, resolutions=None, reference_pose=unit_pose(), custom_limits={}, **kwargs):
    assert len(joints) == len(end_conf)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(body, joints, weights=weights)
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles, {attachment} | set(attachments),
                                    self_collisions, disabled_collisions,
                                    max_distance=max_distance, custom_limits=custom_limits)
    def water_test(q):
        if attachment is None:
            return False
        set_joint_positions(body, joints, q)
        attachment.assign()
        attachment_pose = get_pose(attachment.child)
        pose = multiply(attachment_pose, reference_pose) # TODO: confirm not inverted
        roll, pitch, _ = euler_from_quat(quat_from_pose(pose))
        violation = (MAX_ROTATION < abs(roll)) or (MAX_ROTATION < abs(pitch))
        #if violation: # TODO: check whether different confs can be waypoints for each object
        #    print(q, violation)
        #    wait_for_user()
        return violation
    invalid_fn = lambda q, **kwargs: water_test(q) or collision_fn(q, **kwargs)
    start_conf = get_joint_positions(body, joints)
    if not check_initial_end(start_conf, end_conf, invalid_fn):
        return None
    return birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, invalid_fn, **kwargs)

##################################################

def get_motion_fn(world, collisions=True, teleport=False):
    # TODO(caelan): could also also plan w/o arm_confs
    disabled_collisions = get_disabled_collisions(world.robot)
    custom_limits = get_pr2_safety_limits(world.robot)
    smooth = 100

    def gen_fn(arm, conf1, conf2, fluents=[]):
        arm_confs, object_grasps, object_poses, contains_liquid = parse_fluents(world, fluents)
        for a, q in arm_confs.items():
            #print(a, q) # TODO: some optimistic values are getting through
            set_joint_positions(world.robot, get_arm_joints(world.robot, a), q)
        for name, pose in object_poses.items():
            set_pose(world.bodies[name], pose)
        obstacle_names = list(world.get_fixed()) + list(object_poses)
        obstacles = [world.bodies[name] for name in obstacle_names]

        attachments = {}
        holding_water = None
        water_attachment = None
        for arm2, (obj, grasp) in object_grasps.items():
            set_gripper_position(world.robot, arm, grasp.grasp_width)
            attachment = get_grasp_attachment(world, arm2, grasp)
            attachment.assign()
            if arm == arm2: # The moving arm
                if obj in contains_liquid:
                    holding_water = obj
                    water_attachment = attachment
                attachments[obj] = attachment
            else: # The stationary arm
                obstacles.append(world.get_body(obj))
        if not collisions:
            obstacles = []
        # TODO: dynamically adjust step size to be more conservative near longer movements

        arm_joints = get_arm_joints(world.robot, arm)
        set_joint_positions(world.robot, arm_joints, conf1)
        weights, resolutions = get_weights_resolutions(world.robot, arm)
        #print(holding_water, attachments, [get_body_name(body) for body in obstacles])
        if teleport:
            path = [conf1, conf2]
        elif holding_water is None:
            # TODO(caelan): unify these two methods
            path = plan_joint_motion(world.robot, arm_joints, conf2,
                                     resolutions=resolutions, weights=weights, obstacles=obstacles,
                                     attachments=attachments.values(), self_collisions=collisions,
                                     disabled_collisions=disabled_collisions,
                                     max_distance=COLLISION_BUFFER, custom_limits=custom_limits,
                                     restarts=5, iterations=50, smooth=smooth)
        else:
            reference_pose = (unit_point(), get_liquid_quat(holding_water))
            path = plan_water_motion(world.robot, arm_joints, conf2, water_attachment,
                                     resolutions=resolutions, weights=weights, obstacles=obstacles,
                                     attachments=attachments.values(), self_collisions=collisions,
                                     disabled_collisions=disabled_collisions,
                                     max_distance=COLLISION_BUFFER, custom_limits=custom_limits,
                                     reference_pose=reference_pose,
                                     restarts=7, iterations=75, smooth=smooth)

        if path is None:
            return None
        control = Control({
            'action': 'move-arm',
            #'objects': [],
            'context': Context(
                savers=[BodySaver(world.robot)], # TODO: robot might be at the wrong conf
                attachments=attachments),
            'commands': [
                ArmTrajectory(arm, path),
            ],
        })
        return (control,)
    return gen_fn
