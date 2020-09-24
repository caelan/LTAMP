import random

import numpy as np

from itertools import cycle, count
from control_tools.execution import ArmTrajectory
from perception_tools.common import get_type
from plan_tools.common import Control, Conf, get_reference_pose, STIR_QUATS, \
    lookup_orientation, get_urdf_from_center, LIQUID_QUATS, compute_base_diameter, get_urdf_from_base
from plan_tools.samplers.generators import get_pairwise_arm_links, set_gripper_position, Context, \
    interpolate_pose_waypoints, plan_attachment_motion, visualize_cartesian_path
from plan_tools.samplers.grasp import get_grasp_attachment
from plan_tools.samplers.pour import bowl_path_collision, get_parameter_generator, scale_parameter
from pybullet_tools.pr2_utils import get_disabled_collisions, get_arm_joints
from pybullet_tools.utils import approximate_as_cylinder, Euler, multiply, set_joint_positions, \
    invert, set_pose, Pose, approximate_as_prism, BodySaver, unit_point, get_pose, Point, \
    pairwise_collision, wait_for_user, vertices_from_rigid, draw_pose
from learn_tools.learner import get_trial_parameter_fn, DESIGNED, CONSTANT, RANDOM, LEARNED, \
    sample_parameter, get_explore_parameter_fn

# TODO: the matrices are ill-conditioned if a feature is always constant
SCOOP_FEATURES = [
    'bowl_diameter', 'bowl_height', 'bowl_base_diameter',
    'spoon_height', # 'spoon_width', 'spoon_length'
]

RELATIVE_SCOOP = True
TOP_OFFSET_Z = 0.02

RELATIVE_SCOOP_SCALING = {
    'start_scoop_y' : 'bowl_diameter',
    'end_scoop_y': 'bowl_diameter',
    'exit_y': 'bowl_diameter',
    # 'start_scoop_y': 'bowl_base_diameter',
    # 'end_scoop_y': 'bowl_base_diameter',
}


##################################################

def get_scoop_feature(world, bowl_name, spoon_name):
    bowl_body = world.get_body(bowl_name)
    _, (bowl_d, bowl_h) = approximate_as_cylinder(bowl_body)
    bowl_vertices = vertices_from_rigid(bowl_body)
    #bowl_mesh = read_obj(load_cup_bowl_obj(get_type(bowl_name))[0])
    #print(len(bowl_vertices), len(bowl_mesh.vertices))

    spoon_c, (spoon_w, spoon_l, spoon_h) = approximate_as_prism(
        world.get_body(spoon_name), body_pose=(unit_point(), lookup_orientation(spoon_name, STIR_QUATS)))

    # TODO: compute moments/other features from the mesh
    feature = {
        'bowl_name': bowl_name,
        'bowl_type': get_type(bowl_name),
        'bowl_diameter': bowl_d,
        'bowl_height': bowl_h,
        'bowl_base_diameter': compute_base_diameter(bowl_vertices),

        # In stirring orientation
        'spoon_name': spoon_name,
        'spoon_type': get_type(spoon_name),
        'spoon_width': spoon_w,
        'spoon_length': spoon_l,
        'spoon_height': spoon_h,
    }
    return feature

def lower_spoon(world, bowl_name, spoon_name, min_spoon_z, max_spoon_z):
    assert min_spoon_z <= max_spoon_z
    bowl_body = world.get_body(bowl_name)
    bowl_urdf_from_center = get_urdf_from_base(bowl_body) # get_urdf_from_center

    spoon_body = world.get_body(spoon_name)
    spoon_quat = lookup_orientation(spoon_name, STIR_QUATS)
    spoon_urdf_from_center = get_urdf_from_base(spoon_body, reference_quat=spoon_quat) # get_urdf_from_center
    # Keeping the orientation consistent for generalization purposes

    # TODO: what happens when the base isn't flat (e.g. bowl)
    bowl_pose = get_pose(bowl_body)
    stir_z = None
    for z in np.arange(max_spoon_z, min_spoon_z, step=-0.005):
        bowl_from_stirrer = multiply(bowl_urdf_from_center, Pose(Point(z=z)),
                                     invert(spoon_urdf_from_center))
        set_pose(spoon_body, multiply(bowl_pose, bowl_from_stirrer))
        #wait_for_user()
        if pairwise_collision(bowl_body, spoon_body):
            # Want to be careful not to make contact with the base
            break
        stir_z = z
    return stir_z

def sample_scoop_parameter(world, feature):
    # TODO: adjust for RELATIVE_SCOOP
    side_offset_y = 0.02
    start_y = feature['bowl_diameter']/2 - feature['spoon_length']/2 - side_offset_y

    entry_z = feature['bowl_height'] + TOP_OFFSET_Z
    exit_z = feature['bowl_height']  + feature['spoon_length'] / 2  + TOP_OFFSET_Z

    #min_stir_z = -feature['bowl_height'] / 2 + feature['spoon_height'] / 2 # center
    min_stir_z = 0 # base
    scoop_z = lower_spoon(world, feature['bowl_name'], feature['spoon_name'], min_stir_z, entry_z)

    #spoon_c, (_, spoon_l, spoon_h) = approximate_as_prism(
    # spoon_body, body_pose=(unit_point(), lookup_orientation(feature['spoon_name'], LIQUID_QUATS)))
    #exit_y = -feature['spoon_height']/2 # Due to the spoon rotation
    exit_y = 0
    #exit_y = feature['spoon_height']/2

    # TODO: choose angle to tilt into before moving
    # TODO: manually specify points
    # TODO: choose initial scoop pitch
    # TODO: choose final scoop pitch

    parameter = {
        #'entry_z': entry_z,
        'start_scoop_y': start_y,
        'scoop_z': scoop_z,
        'end_scoop_y': -start_y,
        'exit_y': exit_y,
        'exit_z': exit_z,
        # TODO: other parameters for the interpolation
        # 3 key points of interest when scooping
        # I guess I could make it just two, this is more consistent with Box2D
    }
    if RELATIVE_SCOOP:
        parameter = scale_parameter(feature, parameter, RELATIVE_SCOOP_SCALING)

    yield parameter

##################################################

# Diameters: {0.045, 0.055, 0.065}

SCOOP_PARAMETER_RANGES = {
    'start_scoop_y': (0.0, 0.5), # % bowl_diameter
    'end_scoop_y': (-0.5, 0.0), # % bowl_diameter
    'exit_y': (-0.5, 0.5), # % bowl_diameter
    'scoop_z': (-0.02, 0.02), # meters
    'exit_z': (0.0, 0.05), # meters
} if RELATIVE_SCOOP else {
    'start_scoop_y': (-0.01, 0.05),  # meters
    'end_scoop_y': (-0.05, 0.01),  # meters
    'exit_y': (-0.025, 0.075),  # meters
    # 'entry_z': (0.05, 0.1),
    'scoop_z': (-0.02, 0.02),
    'exit_z': (0.0, 0.05),  # meters
}

SCOOP_PARAMETER_FNS = {
    DESIGNED: sample_scoop_parameter,
    #CONSTANT: get_trial_parameter_fn(SCOOP_PARAMETER),
    RANDOM: get_explore_parameter_fn(SCOOP_PARAMETER_RANGES),
    #LEARNED: predict_pour_parameter,
}

##################################################

def sample_scoop_trajectory(world, feature, parameter, collisions=True):
    bowl_body = world.get_body(feature['bowl_name'])
    bowl_urdf_from_center = get_urdf_from_base(bowl_body) # get_urdf_from_base
    spoon_body = world.get_body(feature['spoon_name'])
    spoon_quat = lookup_orientation(feature['spoon_name'], STIR_QUATS)
    spoon_urdf_from_center = get_urdf_from_base(spoon_body, reference_quat=spoon_quat) # get_urdf_from_base

    if RELATIVE_SCOOP:
        parameter = scale_parameter(feature, parameter, RELATIVE_SCOOP_SCALING, descale=True)
    #entry_z = parameter['entry_z'] # Ends up colliding quite frequently
    entry_z = feature['bowl_height'] + TOP_OFFSET_Z
    exit_z = parameter['exit_z'] + feature['bowl_height'] + feature['spoon_length'] / 2

    entry_pos = np.array([0., parameter['start_scoop_y'], entry_z])
    start_pos = np.array([0., parameter['start_scoop_y'], parameter['scoop_z']])
    end_pos = np.array([0., parameter['end_scoop_y'], parameter['scoop_z']])
    exit_pos = np.array([0., parameter['exit_y'], exit_z])

    spoon_path_in_bowl = interpolate_pose_waypoints(
        [Pose(point=point) for point in [entry_pos, start_pos, end_pos]]
        + [(Pose(exit_pos, Euler(roll=-np.pi/2)))])

    # TODO(caelan): check the ordering/inversion when references are not the identity
    spoon_path = [multiply(bowl_urdf_from_center, spoon_pose_in_bowl, invert(spoon_urdf_from_center))
                  for spoon_pose_in_bowl in spoon_path_in_bowl]

    #draw_pose(get_pose(bowl_body))
    #set_pose(spoon_body, multiply(get_pose(bowl_body), bowl_urdf_from_center,
    #                              Pose(point=start_pos), invert(spoon_urdf_from_center)))
    #set_pose(bowl_body, Pose())
    #wait_for_user()

    collision_path = [spoon_path[0], spoon_path[-1]]
    #collision_path = [spoon_path[-1]]
    if collisions and bowl_path_collision(bowl_body, spoon_body, collision_path):
        return None
    return spoon_path

def is_valid_scoop(world, feature, parameter):
    with world:
        # Assumes bowl is rotationally symmetric
        spoon_path = sample_scoop_trajectory(world, feature, parameter)
        return spoon_path is not None

##################################################

def get_scoop_gen_fn(world, max_attempts=100, collisions=True, parameter_fns={}):
    parameter_fn = parameter_fns.get(get_scoop_gen_fn, SCOOP_PARAMETER_FNS[DESIGNED])
    parameter_generator = get_parameter_generator(world, parameter_fn, is_valid_scoop)
    disabled_collisions = get_disabled_collisions(world.robot)
    disabled_collisions.update(get_pairwise_arm_links(world.robot, world.arms))
    obstacles = list(map(world.get_body, world.get_fixed())) if collisions else []
    collision_buffer = 0.0
    # TODO: sometimes the bowls are fixed but sometimes they are not

    def gen_fn(arm, bowl_name, bowl_pose, spoon_name, grasp):
        if bowl_name == spoon_name:
            return
        bowl_body = world.get_body(bowl_name)
        attachment = get_grasp_attachment(world, arm, grasp)
        feature = get_scoop_feature(world, bowl_name, spoon_name)
        for parameter in parameter_generator(feature):
            spoon_path_bowl = sample_scoop_trajectory(world, feature, parameter, collisions=collisions)
            if spoon_path_bowl is None:
                continue
            for _ in range(max_attempts):
                set_gripper_position(world.robot, arm, grasp.grasp_width)
                set_pose(bowl_body, bowl_pose)
                rotate_bowl = Pose(euler=Euler(yaw=random.uniform(-np.pi, np.pi)))
                spoon_path = [multiply(bowl_pose, invert(rotate_bowl), spoon_pose_bowl)
                              for spoon_pose_bowl in spoon_path_bowl]
                #visualize_cartesian_path(world.get_body(spoon_name), spoon_path)
                # TODO: pass in tool collision pairs here
                arm_path = plan_attachment_motion(world.robot, arm, attachment, spoon_path,
                                              obstacles=set(obstacles) - {bowl_body},
                                              self_collisions=collisions, disabled_collisions=disabled_collisions,
                                              collision_buffer=collision_buffer, attachment_collisions=False)
                if arm_path is None:
                    continue
                pre_conf = Conf(arm_path[0])
                set_joint_positions(world.robot, get_arm_joints(world.robot, arm), pre_conf)
                attachment.assign()
                if pairwise_collision(world.robot, bowl_body):
                    # TODO: ensure no robot/bowl collisions for the full path
                    continue
                robot_saver = BodySaver(world.robot)
                post_conf = Conf(arm_path[-1])
                control = Control({
                    'action': 'scoop',
                    'objects': [bowl_name, spoon_name],
                    'feature': feature,
                    'parameter': parameter,
                    'context': Context(
                        savers=[robot_saver],
                        attachments={spoon_name: attachment}),
                    'commands': [
                        ArmTrajectory(arm, arm_path, dialation=4.0),
                    ],
                })
                yield (pre_conf, post_conf, control)
                break
            else:
                yield None
    return gen_fn
