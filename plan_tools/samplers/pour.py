import random
from itertools import cycle

import numpy as np

from control_tools.execution import ArmTrajectory, Rest
from learn_tools.learner import get_trial_parameter_fn, DESIGNED, CONSTANT, RANDOM, LEARNED, SKLearnSampler, \
    get_explore_parameter_fn
from perception_tools.common import get_type
from plan_tools.common import Control, Conf, get_reference_pose, get_liquid_quat, \
    get_urdf_from_base, constant_velocity_times, get_urdf_from_center, compute_base_diameter, get_urdf_from_top
from plan_tools.samplers.collision import cartesian_path_collision, body_pair_collision
from plan_tools.samplers.generators import solve_inverse_kinematics, plan_workspace_motion, \
    get_pairwise_arm_links, Context, set_gripper_position
from plan_tools.samplers.grasp import get_grasp_attachment
from pybullet_tools.pr2_utils import get_disabled_collisions
from pybullet_tools.utils import approximate_as_cylinder, Euler, multiply, unit_quat, unit_point, invert, \
    point_from_pose, set_pose, get_pose, create_sphere, interpolate_poses, Pose, approximate_as_prism, \
    BodySaver, Point, vertices_from_rigid, ClientSaver

POUR_FEATURES = [
    'bowl_diameter', 'bowl_height', 'bowl_base_diameter',
    'cup_diameter', 'cup_height', 'cup_base_diameter',
]

RELATIVE_POUR = True

RELATIVE_POUR_SCALING = {
    'axis_in_cup_x' : 'cup_diameter',
    'axis_in_cup_z': 'cup_height',
    'axis_in_bowl_x': 'bowl_diameter',
    #'axis_in_bowl_z': 'bowl_height',
}

##################################################

# TODO: could compute bowl_base_diameter by bowl_diameter relative to bowl_type

def scale_parameter(feature, parameter, scaling={}, descale=False):
    scaled_parameter = dict(parameter)
    for param, feat in scaling.items():
        # if (feat in feature) and (param in scaled_parameter):
        if descale:
            scaled_parameter[param] *= feature[feat]
        else:
            scaled_parameter[param] /= feature[feat]
    return scaled_parameter

##################################################

def get_pour_feature(world, bowl_name, cup_name):
    bowl_body = world.get_body(bowl_name)
    bowl_reference = get_reference_pose(bowl_name)
    _, (bowl_d, bowl_h) = approximate_as_cylinder(bowl_body, body_pose=bowl_reference)
    bowl_vertices = vertices_from_rigid(bowl_body)
    #bowl_mesh = read_obj(load_cup_bowl_obj(get_type(bowl_name))[0])
    #print(len(bowl_vertices), len(bowl_mesh.vertices))

    cup_body = world.get_body(cup_name)
    cup_reference = (unit_point(), get_liquid_quat(cup_name))
    _, (cup_d, _, cup_h) = approximate_as_prism(cup_body, body_pose=cup_reference)
    cup_vertices = vertices_from_rigid(cup_body)
    #cup_mesh = read_obj(load_cup_bowl_obj(get_type(cup_name))[0])

    # TODO: compute moments/other features from the mesh
    feature = {
        'bowl_name': bowl_name,
        'bowl_type': get_type(bowl_name),
        'bowl_diameter': bowl_d,
        'bowl_height': bowl_h,
        'bowl_base_diameter': compute_base_diameter(bowl_vertices),

        'cup_name': cup_name,
        'cup_type': get_type(cup_name),
        'cup_diameter': cup_d,
        'cup_height': cup_h,
        'cup_base_diameter': compute_base_diameter(cup_vertices),
    }
    return feature

##################################################

# POUR_PARAMETER = {
#     #'axis_in_bowl_x': 0.07337437110184712,
#     #'axis_in_bowl_z': 0.18011448322568957,
#     #'axis_in_cup_z': 0.05549999836087227,
#     #'axis_in_cup_x': -0.05,
#     #'pitch': -2.35619449019,
#     'axis_in_bowl_x': 0.05,
#     'axis_in_bowl_z': 0.15,
#     'axis_in_cup_z': -0.05,
#     'axis_in_cup_x': -0.05,
#     'pitch': -np.pi/2,
# }

def sample_pour_parameter(world, feature):
    # TODO: adjust for RELATIVE_POUR
    cup_pour_pitch = -3 * np.pi / 4
    # pour_cup_pitch = -5*np.pi/6
    # pour_cup_pitch = -np.pi

    #axis_in_cup_center_x = -0.05
    axis_in_cup_center_x = 0
    #axis_in_cup_center_z = -feature['cup_height']/2.
    axis_in_cup_center_z = 0. # This is in meters (not a fraction of the high)
    #axis_in_cup_center_z = feature['cup_height']/2.

    # tl := top left | tr := top right
    cup_tl_in_center = np.array([-feature['cup_diameter']/2, 0, feature['cup_height']/2])
    cup_tl_in_axis = cup_tl_in_center - Point(x=axis_in_cup_center_x, z=axis_in_cup_center_z)
    cup_tl_angle = np.math.atan2(cup_tl_in_axis[2], cup_tl_in_axis[0])
    cup_tl_pour_pitch = cup_pour_pitch - cup_tl_angle

    cup_radius2d = np.linalg.norm([cup_tl_in_axis])
    pivot_in_bowl_tr = Point(
        x=-(cup_radius2d * np.math.cos(cup_tl_pour_pitch) + 0.01),
        z=(cup_radius2d * np.math.sin(cup_tl_pour_pitch) + 0.01))

    bowl_tr_in_bowl_center = Point(x=feature['bowl_diameter'] / 2, z=feature['bowl_height'] / 2)
    pivot_in_bowl_center = bowl_tr_in_bowl_center + pivot_in_bowl_tr

    parameter = {
        'pitch': cup_pour_pitch,
        'axis_in_cup_x': axis_in_cup_center_x,
        'axis_in_cup_z': axis_in_cup_center_z,
        'axis_in_bowl_x': pivot_in_bowl_center[0],
        'axis_in_bowl_z': pivot_in_bowl_center[2],
        #'velocity': None,
        #'bowl_yaw': None,
        #'cup_yaw': None,
        'relative': RELATIVE_POUR,
    }
    if RELATIVE_POUR:
        parameter = scale_parameter(feature, parameter, RELATIVE_POUR_SCALING)
    yield parameter

# TODO: the 0.75 parameters could be increased
POUR_PARAMETER_RANGES = {
    # Relative to bowl/cup bounding box centers
    'pitch': (-np.pi, -np.pi / 2),   # radians
    'axis_in_cup_x': (-0.75, 0.75),  # % cup_diameter
    'axis_in_cup_z': (-0.75, 0.75),  # % cup_height
    'axis_in_bowl_x': (-0.25, 0.75), # % bowl_diameter
    'axis_in_bowl_z': (0.0, 0.15),   # meters
} if RELATIVE_POUR else {
    # Relative to bowl/cup bounding box centers
    'pitch': (-np.pi, -np.pi / 2),  # radians
    'axis_in_cup_x': (-0.05, 0.05), # meters
    'axis_in_cup_z': (-0.05, 0.15), # meters
    'axis_in_bowl_x': (-0.1, 0.2),  # Top diameter [0.1, 0.25]
    'axis_in_bowl_z': (0.1, 0.3),   # Height [0.05, 0.11]
    # 'axis_in_cup_x': (0.0, 0.0),
    # 'axis_in_cup_z': (0.0, 0.0),
}

def predict_pour_parameter(world, feature):
    #dataset = 'collect_pour_linux_caelan_19-03-21_13-38-48'
    dataset = 'pour_linux_caelan_19-03-22_13-50-22'
    #learner = 'extratreesclassifier'
    learner = 'mlpclassifier'
    #learner = 'gaussianprocessclassifier'
    path = '{}/{}.pk3'.format(dataset, learner)
    sampler = SKLearnSampler.load(path)
    return sampler.predict(feature)

##################################################

def get_bowl_from_pivot(world, feature, parameter):
    bowl_body = world.get_body(feature['bowl_name'])
    bowl_urdf_from_center = get_urdf_from_top(bowl_body) # get_urdf_from_base | get_urdf_from_center
    if RELATIVE_POUR:
        parameter = scale_parameter(feature, parameter, RELATIVE_POUR_SCALING, descale=True)
    bowl_base_from_pivot = Pose(Point(x=parameter['axis_in_bowl_x'], z=parameter['axis_in_bowl_z']))
    return multiply(bowl_urdf_from_center, bowl_base_from_pivot)

def pour_path_from_parameter(world, feature, parameter, cup_yaw=None):
    cup_body = world.get_body(feature['cup_name'])
    #cup_urdf_from_center = get_urdf_from_center(cup_body, reference_quat=get_liquid_quat(feature['cup_name']))
    ref_from_urdf = (unit_point(), get_liquid_quat(feature['cup_name']))
    cup_center_in_ref, _ = approximate_as_prism(cup_body, body_pose=ref_from_urdf)
    cup_center_in_ref[:2] = 0 # Assumes the xy pour center is specified by the URDF (e.g. for spoons)
    cup_urdf_from_center = multiply(invert(ref_from_urdf), Pose(point=cup_center_in_ref))

    # TODO: allow some deviation around cup_yaw for spoons
    if cup_yaw is None:
        cup_yaw = random.choice([0, np.pi]) if 'spoon' in feature['cup_name'] \
            else random.uniform(-np.pi, np.pi)
    z_rotate_cup = Pose(euler=Euler(yaw=cup_yaw))

    bowl_from_pivot = get_bowl_from_pivot(world, feature, parameter)
    if RELATIVE_POUR:
        parameter = scale_parameter(feature, parameter, RELATIVE_POUR_SCALING, descale=True)
    base_from_pivot = Pose(Point(x=parameter['axis_in_cup_x'], z=parameter['axis_in_cup_z']))

    initial_pitch = 0
    final_pitch = parameter['pitch']
    assert -np.pi <= final_pitch <= initial_pitch
    cup_path_in_bowl = []
    for pitch in list(np.arange(final_pitch, initial_pitch, np.pi/16)) + [initial_pitch]:
        rotate_pivot = Pose(euler=Euler(pitch=pitch)) # Can also interpolate directly between start and end quat
        cup_path_in_bowl.append(multiply(bowl_from_pivot, rotate_pivot, invert(base_from_pivot),
                                         z_rotate_cup, invert(cup_urdf_from_center)))
    cup_times = constant_velocity_times(cup_path_in_bowl)
    # TODO: check for collisions here?

    return cup_path_in_bowl, cup_times

##################################################

def bowl_path_collision(bowl_body, body, body_path_bowl):
    bowl_pose = get_pose(bowl_body)
    with BodySaver(body):
        for cup_pose_bowl in body_path_bowl:
            cup_pose_world = multiply(bowl_pose, cup_pose_bowl)
            set_pose(body, cup_pose_world)
            if body_pair_collision(bowl_body, body): #, collision_buffer=0.0):
                return True
    return False

def is_valid_pour(world, feature, parameter):
    with world:
        # Assumes bowl and cup are rotationally symmetric
        bowl_body = world.get_body(feature['bowl_name'])
        cup_body = world.get_body(feature['cup_name'])
        cup_path_bowl_witness, _ = pour_path_from_parameter(world, feature, parameter)
        # TODO: check whether there exists a safe gripper grasp
        # TODO: check collisions with the ground
        return not bowl_path_collision(bowl_body, cup_body, cup_path_bowl_witness)


def get_water_path(bowl_body, bowl_pose, cup_body, pose_waypoints):
    cup_pose = pose_waypoints[0]
    bowl_origin_from_base = get_urdf_from_base(bowl_body)  # TODO: reference pose
    cup_origin_from_base = get_urdf_from_base(cup_body)
    ray_start = point_from_pose(multiply(cup_pose, cup_origin_from_base))
    ray_end = point_from_pose(multiply(bowl_pose, bowl_origin_from_base))
    water_path = interpolate_poses((ray_start, unit_quat()), (ray_end, unit_quat()), pos_step_size=0.01)
    return water_path

def water_robot_collision(world, bowl_body, bowl_pose, cup_body, pose_waypoints):
    water_path = get_water_path(bowl_body, bowl_pose, cup_body, pose_waypoints)
    return cartesian_path_collision(world.water_body, water_path, [world.robot])  # + obstacles
    # ray_result = ray_collision(ray_start, ray_end) # TODO(caelan): doesn't seem to work...
    # [ray_result] = batch_ray_collision([(ray_start, ray_end)])
    # print(ray_result)
    # wait_for_interrupt()
    # if ray_result.objectUniqueId != -1:
    #    continue

def water_obstacle_collision(world, bowl_body, bowl_pose, cup_body, pose_waypoints):
    water_path = get_water_path(bowl_body, bowl_pose, cup_body, pose_waypoints)
    obstacles = [world.get_body(name) for name in world.get_obstacles()]
    return cartesian_path_collision(world.water_body, water_path, obstacles)

def get_parameter_generator(world, parameter_fn, valid_fn=lambda *args: True, revisit=True):
    def fn(feature):
        num_valid = 0
        for i, p in enumerate(parameter_fn(world, feature)):
            if valid_fn(world, feature, p):
                num_valid += 1
                yield p
            else:
                print('Warning! The {}th parameter is invalid'.format(i+1))
        print('Enumerated {} valid parameters'.format(num_valid))
    return lambda f: cycle(fn(f)) if revisit else fn(f)

POUR_PARAMETER_FNS = {
    DESIGNED: sample_pour_parameter,
    #CONSTANT: get_trial_parameter_fn(POUR_PARAMETER),
    RANDOM: get_explore_parameter_fn(POUR_PARAMETER_RANGES),
    #RANDOM: get_parameter_generator(world, get_explore_parameter_fn(POUR_PARAMETER_RANGES)), # TODO(caelan)
    LEARNED: predict_pour_parameter,
}

##################################################

def get_pour_gen_fn(world, max_attempts=100, collisions=True, parameter_fns={}):
    # TODO(caelan): could also simulate the predicated sample
    # TODO(caelan): be careful with the collision_buffer
    parameter_fn = parameter_fns.get(get_pour_gen_fn, POUR_PARAMETER_FNS[DESIGNED])
    parameter_generator = get_parameter_generator(world, parameter_fn, is_valid_pour)
    disabled_collisions = get_disabled_collisions(world.robot)
    disabled_collisions.update(get_pairwise_arm_links(world.robot, world.arms))
    obstacles = [world.get_body(surface) for surface in world.get_fixed()] if collisions else []
    #world.water_body = load_pybullet(get_body_urdf('red_sphere')) # red_sphere, blue_sphere
    world.water_body = create_sphere(radius=0.005, color=(1, 0, 0, 1))

    def gen_fn(arm, bowl_name, bowl_pose, cup_name, grasp):
        if bowl_name == cup_name:
            return
        attachment = get_grasp_attachment(world, arm, grasp)
        bowl_body = world.get_body(bowl_name)
        cup_body = world.get_body(cup_name)
        feature = get_pour_feature(world, bowl_name, cup_name)

        # TODO: this may be called several times with different grasps
        for parameter in parameter_generator(feature):
            for i in range(max_attempts):
                set_pose(bowl_body, bowl_pose) # Reset because might have changed
                set_gripper_position(world.robot, arm, grasp.grasp_width)
                cup_path_bowl, times_from_start = pour_path_from_parameter(world, feature, parameter)
                rotate_bowl = Pose(euler=Euler(yaw=random.uniform(-np.pi, np.pi)))
                cup_path = [multiply(bowl_pose, invert(rotate_bowl), cup_pose_bowl)
                            for cup_pose_bowl in cup_path_bowl]
                #if world.visualize:
                #    visualize_cartesian_path(cup_body, cup_path)
                if cartesian_path_collision(cup_body, cup_path, obstacles + [bowl_body]):
                    print('Attempt {}: Pour path collision!'.format(i))
                    continue
                tool_waypoints = [multiply(p, invert(grasp.grasp_pose)) for p in cup_path]
                grip_conf = solve_inverse_kinematics(world.robot, arm, tool_waypoints[0], obstacles=obstacles)
                if grip_conf is None:
                    continue
                if water_robot_collision(world, bowl_body, bowl_pose, cup_body, cup_path):
                    print('Attempt {}: Water robot collision'.format(i))
                    continue
                if water_obstacle_collision(world, bowl_body, bowl_pose, cup_body, cup_path):
                    print('Attempt {}: Water obstacle collision'.format(i))
                    continue

                post_path = plan_workspace_motion(world.robot, arm, tool_waypoints,
                                                  obstacles=obstacles + [bowl_body], attachments=[attachment],
                                                  self_collisions=collisions, disabled_collisions=disabled_collisions)
                if post_path is None:
                    continue
                pre_conf = Conf(post_path[-1])
                pre_path = post_path[::-1]
                post_conf = pre_conf
                control = Control({
                    'action': 'pour',
                    'feature': feature,
                    'parameter': parameter,
                    'objects': [bowl_name, cup_name],
                    'times': times_from_start,
                    'context': Context(
                        savers=[BodySaver(world.robot)], # TODO: robot might be at the wrong conf
                        attachments={cup_name: attachment}),
                    'commands': [
                        ArmTrajectory(arm, pre_path, dialation=2.),
                        Rest(duration=2.0),
                        ArmTrajectory(arm, post_path, dialation=2.),
                    ],
                })
                yield (pre_conf, post_conf, control)
                break
            else:
                yield None
    return gen_fn
