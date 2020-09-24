import math
import random
from itertools import cycle, count

import numpy as np

from control_tools.execution import ArmTrajectory
from plan_tools.common import Control, Conf, lookup_orientation, STIR_QUATS, get_urdf_from_base
from plan_tools.samplers.generators import get_pairwise_arm_links, set_gripper_position, \
    Context, plan_attachment_motion, visualize_cartesian_path
from plan_tools.samplers.grasp import get_grasp_attachment
from plan_tools.samplers.scoop import get_scoop_feature, lower_spoon, SCOOP_FEATURES
from pybullet_tools.pr2_utils import get_disabled_collisions
from pybullet_tools.utils import Euler, multiply, invert, set_pose, Pose, BodySaver

from learn_tools.learner import get_trial_parameter_fn, DESIGNED, CONSTANT, RANDOM, LEARNED, \
    sample_parameter, get_explore_parameter_fn

get_stir_feature = get_scoop_feature

STIR_FEATURES = SCOOP_FEATURES

def sample_stir_parameter(world, feature):
    # TODO: move this computation to the trajectory?
    top_offset_z = 0.02
    entry_z = feature['bowl_height'] + top_offset_z
    min_stir_z = 0
    stir_z = lower_spoon(world, feature['bowl_name'], feature['spoon_name'], min_stir_z, entry_z)
    if stir_z is None:
        return

    side_offset_y = 0.005
    spoon_diameter = np.linalg.norm(np.array([feature['spoon_width'], feature['spoon_length']]))
    stir_radius = feature['bowl_diameter']/2 - spoon_diameter/2 - side_offset_y
    if stir_radius <= 0:
        return

    # Relative to bowl/stirrer bounding box centers
    parameter = {
        'entry_z': entry_z,
        'stir_z': stir_z,
        'stir_radius': stir_radius,
        'revolutions': 2,
        #'revolutions': 1e-3,
    }
    yield parameter

STIR_PARAMETER_RANGES = {
    'entry_z': (0.0, 0.1), # Height from base of bowl entry
    'stir_z': (0.0, 0.1), # Height from base of bowl
    'stir_radius': (0.0, 0.05), # Radius of stir
    'revolutions': (0.0, 4.0), # Fractional number of revolutions
    # TODO: could define these as fractions of the diameter / height
}

STIR_PARAMETER_FNS = {
    DESIGNED: sample_stir_parameter,
    #CONSTANT: get_trial_parameter_fn(POUR_PARAMETER),
    RANDOM: get_explore_parameter_fn(STIR_PARAMETER_RANGES),
    #LEARNED: predict_pour_parameter,
}

##################################################

def sample_stir_trajectory(world, feature, parameter):
    bowl_urdf_from_center = get_urdf_from_base(world.bodies[feature['bowl_name']])
    stirrer_quat = lookup_orientation(feature['spoon_name'], STIR_QUATS)
    stirrer_urdf_from_center = get_urdf_from_base(
        world.bodies[feature['spoon_name']], reference_quat=stirrer_quat)

    stir_pos = np.array([0., 0., parameter['stir_z']])
    entry_pos = np.array([0, 0, parameter['entry_z']])

    # TODO: could rotate the wrist in place
    # TODO: could reverse the trajectory afterwards
    # TODO: could connect up the start and end
    step_size = np.pi/16
    stir_positions = [stir_pos + parameter['stir_radius']*np.array([math.cos(theta), math.sin(theta), 0])
                      for theta in np.arange(0, parameter['revolutions']*2*np.pi, step_size)] # Counter-clockwise
    entry_pos = stir_positions[0] + (entry_pos - stir_pos)
    exit_pos = stir_positions[-1] + (entry_pos - stir_pos)

    initial_yaw = random.uniform(-np.pi, np.pi)
    rotate_stirrer = Pose(euler=Euler(yaw=initial_yaw))

    # TODO(caelan): check the ordering/inversion when references are not the identity
    return [multiply(bowl_urdf_from_center, Pose(point=point),
                     rotate_stirrer, invert(stirrer_urdf_from_center))
            for point in ([entry_pos] + stir_positions + [exit_pos])]

##################################################

def get_stir_gen_fn(world, max_attempts=25, collisions=True, parameter_fns={}, revisit=False):
    parameter_fn = parameter_fns.get(get_stir_gen_fn, STIR_PARAMETER_FNS[DESIGNED])
    disabled_collisions = get_disabled_collisions(world.robot)
    disabled_collisions.update(get_pairwise_arm_links(world.robot, world.arms))
    obstacles = [world.bodies[surface] for surface in world.get_fixed()] if collisions else []
    collision_buffer = 0.0

    def gen_fn(arm, bowl_name, bowl_pose, stirrer_name, grasp):
        if bowl_name == stirrer_name:
            return
        bowl_body = world.bodies[bowl_name]
        attachment = get_grasp_attachment(world, arm, grasp)
        feature = get_stir_feature(world, bowl_name, stirrer_name)

        parameter_generator = parameter_fn(world, feature)
        if revisit:
            parameter_generator = cycle(parameter_generator)
        for parameter in parameter_generator:
            for _ in range(max_attempts):
                set_gripper_position(world.robot, arm, grasp.grasp_width)
                set_pose(bowl_body, bowl_pose)
                stirrer_path_bowl = sample_stir_trajectory(world, feature, parameter)
                rotate_bowl = Pose(euler=Euler(yaw=random.uniform(-np.pi, np.pi)))
                stirrer_path = [multiply(bowl_pose, invert(rotate_bowl), cup_pose_bowl)
                                for cup_pose_bowl in stirrer_path_bowl]
                #visualize_cartesian_path(world.get_body(stirrer_name), stirrer_path)
                arm_path = plan_attachment_motion(world.robot, arm, attachment, stirrer_path,
                                              obstacles=set(obstacles) - {bowl_body},
                                              self_collisions=collisions, disabled_collisions=disabled_collisions,
                                              collision_buffer=collision_buffer, attachment_collisions=False)
                if arm_path is None:
                    continue
                pre_conf = Conf(arm_path[0])
                post_conf = Conf(arm_path[-1])
                control = Control({
                    'action': 'stir',
                    'objects': [bowl_name, stirrer_name],
                    'feature': feature,
                    'parameter': parameter,
                    'context': Context(
                        savers=[BodySaver(world.robot)], # TODO: robot might be at the wrong conf
                        attachments={stirrer_name: attachment}),
                    'commands': [
                        ArmTrajectory(arm, arm_path),
                    ],
                })
                yield (pre_conf, post_conf, control)
                # TODO: continue exploration
    return gen_fn
