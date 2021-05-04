import os
import random
import re

import numpy as np

from collections import namedtuple

from control_tools.common import PR2_JOINT_SAFETY_LIMITS
from dimensions.common import load_cup_bowl_obj
from dimensions.cups.dimensions import OLIVE_CUPS
from perception_tools.common import get_type, get_body_urdf, get_models_path
from pybullet_tools.ikfast.pr2.ik import IK_FRAME as IK_FRAMES
from pybullet_tools.pr2_utils import get_group_joints, gripper_from_arm, get_arm_joints
from pybullet_tools.utils import unit_point, unit_pose, set_joint_positions, link_from_name, multiply, invert, \
    get_link_pose, get_pose, Attachment, joint_from_name, quat_from_euler, Euler, unit_quat, read, \
    approximate_as_cylinder, write, create_obj, ensure_dir, quat_angle_between, movable_from_joints, transform_obj_file, \
    apply_alpha, aabb_from_points, get_aabb_extent, get_numpy_seed, set_random_seed, set_numpy_seed

VIDEOS_DIRECTORY = 'videos/'

LEFT_ARM = 'left'
RIGHT_ARM = 'right'
ARMS = [LEFT_ARM, RIGHT_ARM]
PR2_URDF = 'pr2_description/pr2.urdf'

COFFEE = 'coffee'
SUGAR = 'sugar'
SIM_MATERIALS = (COFFEE, SUGAR)

SCALE_TYPE = 'onyx_scale'
# +/- 0.05 oz scale precision error
SCALE_RESOLUTION_OZ = 0.1
TABLE = 'table'
TOP = None

SYMMETRIES = {
    # TODO: wrap around orientation
    'greenblock': np.pi / 2,
    'purpleblock': np.pi / 2,
    'bluecup': 0,
    'bowl': 0,
    'whitebowl': 0,
}

GRIPPER_LINKS = {
    'left': 'l_gripper_palm_link',
    'right': 'r_gripper_palm_link',
}

STABLE_QUATS = {
    # TODO(caelan): rename to placement orientation
    #'bluecup': quat_from_euler(Euler(roll=np.pi)),
    #'bowl': quat_from_euler(Euler(roll=np.pi)),
    #'spoon': (0.11923844791406242, -0.7234117212002504, -0.13249733589431234, 0.6670098426185677),
}

LIQUID_QUATS = {
    'spoon': quat_from_euler(Euler(pitch=np.pi)),
}

STIR_QUATS = {
    'spoon': quat_from_euler(Euler(roll=-np.pi/2)),
}

# TODO: avoid using this
COLORS = {
    'red': (1, 0, 0),
    'green': (0, 1, 0),
    'blue': (0, 0, 1),
    'black': (0, 0, 0),
    'white': (1, 1, 1),
    'purple': (1, 0, 1),
    'orange': (1, 0.6, 0),
    #'brown': None,
    'grey': (0.5, 0.5, 0.5),
    'tan': np.array([210, 180, 140]) / 255.
}

COLLISION_BUFFER = 0.0
#COLLISION_BUFFER = 0.005
#COLLISION_BUFFER = 0.01
# TODO: different collision buffer for different link pairs

##############################################################################################################

# https://stamps.custhelp.com/app/answers/detail/a_id/3967/~/how-to-install-a-usb-scale
OZ_PER_LB = 16
KG_PER_OZ = 0.0283495

# Can't scoop large bolts

MODEL_MASSES_OZ = {
    # spoons
    'grey_spoon': [0.5],
    'orange_spoon': [0.9],
    'green_spoon': [1.7],

    # tensorflow
    'bowl': [10.7],
    'whitebowl': [9.0, 9.5],
    'bluecup': [1.2, 1.3],
    'greenblock': [5.1, 8.9, 1*OZ_PER_LB + 2.3],
    'purpleblock': [2*OZ_PER_LB + 7.8],

    # bowls
    #'purple_bowl': [1.2],
    'white_bowl': [9.0, 9.5], # whitebowl
    'red_bowl': [3.8],
    'red_speckled_bowl': [4.9],
    'brown_bowl': [10.7], # bowl
    'yellow_bowl': [7.3],
    'blue_white_bowl': [8.6],
    'green_bowl': [10.5],
    'steel_bowl': [8.9],
    'blue_bowl': [14.3],
    'tan_bowl': [1*OZ_PER_LB + 1.8],
    #'stripes_bowl': [5.6, 6.0], # TODO: rename to striped

    'small_green_bowl': [1.6],
    'orange_bowl': [2.8],
    'small_blue_bowl': [5.4],
    'purple_bowl': [6.9],
    'lime_bowl': [8.6],
    'large_red_bowl': [10.5],

    # cups
    'teal_cup': [1.2],
    'orange_cup': [0.5],
    'blue_cup': [0.6],
    'green_cup': [0.7],
    'yellow_cup': [0.8],
    'red_cup': [0.9],
    'purple_cup': [1.0],
    'large_orange_cup': [1.2],

    'olive1_cup': [0.5],
    'olive2_cup': [0.6],
    'olive3_cup': [0.6],
    'olive4_cup': [0.7],

    'blue3D_cup': [0.5],
    'cyan3D_cup': [0.6],
    'green3D_cup': [0.7],
    'yellow3D_cup': [0.7],
    'orange3D_cup': [1.0],
    'red3D_cup': [1.1],

    # materials
    #'thin_bolts': [],
    'small_bolts': [(12.3 - 10.7) / 20, (1*OZ_PER_LB + 2.3 - 10.7) / 101],
    'medium_bolts': [(13.9 - 10.7) / 20],
    'large_bolts': [(1*OZ_PER_LB + 8.2 - 10.7) / 24], # 2cm
    'red_men': [(9.5 - 8.5) / 25], # 9.5oz - 8.5oz per 25 # 1.5cm
    'chickpeas': [(2.0 - 1.2) / 50], # 2.0oz - 1.2oz per 50 # 1cm
}

YCB_CUPS = ['orange_cup', 'blue_cup', 'green_cup', 'yellow_cup', 'red_cup', 'purple_cup', 'large_orange_cup', 'bluecup']
CUP_WITH_LIP = OLIVE_CUPS + YCB_CUPS

MATERIALS = ['red_men', 'chickpeas'] # TODO: two brands of chickpeas

MODEL_MASSES = {name: KG_PER_OZ*np.average(masses) for name, masses in MODEL_MASSES_OZ.items()}
#print(MODEL_MASSES)

# Material diameters
# chickpeas: 0.8 cm
# red_men: 1.5 cm

SPOONS = ['grey_spoon', 'orange_spoon', 'green_spoon']

# used whitebowl
SPOON_CAPACITIES_OZ = {
    ('grey_spoon', 'red_men'): 9.2 - 9.1, # = 0.1 oz
    ('grey_spoon', 'chickpeas'): 9.6 - 9.1, # = 0.5 oz

    ('orange_spoon', 'red_men'): 9.5 - 9.1, # = 0.4 oz
    ('orange_spoon', 'chickpeas'): 10.0 - 9.1, # = 0.9 oz

    ('green_spoon', 'red_men'): 9.8 - 9.1, # = 0.7 oz
    ('green_spoon', 'chickpeas'): 10.8 - 9.1, # = 0.17 oz
}

SPOON_CAPACITIES = {name: KG_PER_OZ*mass for name, mass in SPOON_CAPACITIES_OZ.items()}

CORRECTION = 0.5*KG_PER_OZ*SCALE_RESOLUTION_OZ # Accounts for scale rounding

##############################################################################################################

class Control(dict):
    def __repr__(self):
        #return '{}{}'.format(self.__class__.__name__, tuple(self.keys()))
        return '{}({})'.format(self.__class__.__name__, self.get('action', None))
    #__str__ = __repr__

class Pose(list):
    # TODO: indicate whether the pose is stacked on something
    def __repr__(self):
        #return 'p{}'.format(np.concatenate(self))
        return 'p{}'.format(np.concatenate(self)[:2])


class Conf(list):
    def __repr__(self):
        #return 'q{}{}'.format(len(self), str(np.array(self)))
        return 'q{}'.format(id(self)%1000) # Might cause hash collisions

##############################################################################################################

def is_obj_type(obj, obj_type):
    return obj_type in obj
    #return obj.startswith(obj_type)


def in_type_group(item, types):
    return any(is_obj_type(item, ty) for ty in types)


def get_pr2_safety_limits(pr2):
    return {joint_from_name(pr2, name): limits
            for name, limits in PR2_JOINT_SAFETY_LIMITS.items()}


def get_name(fact):
    return fact[0]


def get_args(fact):
    return fact[1:]


def parse_fluents(world, fluents):
    arm_confs = {}
    object_grasps = {}
    object_poses = {}
    contains_liquid = set()
    for fact in fluents:
        if get_name(fact) == 'atpose':
            o, p = get_args(fact)
            if o != world.world_name:
                object_poses[o] = p
        elif get_name(fact) == 'atconf':
            a, q = get_args(fact)
            arm_confs[a] = q
        elif get_name(fact) == 'holding':
            a, o = get_args(fact)
            object_grasps[a] = o
        elif get_name(fact) == 'atgrasp':
            a, o, g = get_args(fact)
            object_grasps[a] = (o, g)
        elif get_name(fact) == 'contains':
            o, m = get_args(fact)
            contains_liquid.add(o)
        else:
            raise ValueError(get_name(fact))
    return arm_confs, object_grasps, object_poses, contains_liquid


def get_color(name):
    for color in COLORS:
        if color in name:
            return COLORS[color]
    return COLORS['grey']


def get_reference_pose(name): # TODO: bluecup_rotated
    for model, quat in STABLE_QUATS.items():
        if is_obj_type(name, model):
            return (unit_point(), quat)
    return unit_pose()


def lookup_orientation(name, quat_from_model):
    for model, quat in quat_from_model.items():
        if is_obj_type(name, model):
            return quat
    return unit_quat()


def get_liquid_quat(name):
    return lookup_orientation(name, LIQUID_QUATS)


def set_gripper(robot, arm, value):
    joints = get_group_joints(robot, gripper_from_arm(arm))
    set_joint_positions(robot, joints, len(joints)*[value])


def create_attachment(robot, arm, body):
    tool_link = link_from_name(robot, IK_FRAMES[arm])
    grasp_pose = multiply(invert(get_link_pose(robot, tool_link)), get_pose(body))
    return Attachment(robot, tool_link, grasp_pose, body)


def compute_base_diameter(vertices, epsilon=0.001):
    lower = np.min(vertices, axis=0)
    threshold = lower[2] + epsilon
    base_vertices = [vertex for vertex in vertices
                     if vertex[2] <= threshold]
    base_aabb = aabb_from_points(base_vertices)
    #print(len(base_vertices), len(vertices))
    #print(base_aabb)
    return np.average(get_aabb_extent(base_aabb)[:2])

##############################################################################################################


def get_body_obj(name, visual=False):
    regex = r'mesh filename="(\w+.obj)"'
    obj_filenames = re.findall(regex, read(get_body_urdf(name)))
    assert obj_filenames
    visual_filename = obj_filenames[0]
    collision_filename = obj_filenames[-1]
    filename = visual_filename if visual else collision_filename
    return os.path.join(get_models_path(), filename)


def get_urdf_from_z_axis(body, z_fraction, reference_quat=unit_quat()):
    # AKA the pose of the body's center wrt to the body's origin
    # z_fraction=0. => bottom, z_fraction=0.5 => center, z_fraction=1. => top
    ref_from_urdf = (unit_point(), reference_quat)
    center_in_ref, (_, height) = approximate_as_cylinder(body, body_pose=ref_from_urdf)
    center_in_ref[2] += (z_fraction - 0.5)*height
    ref_from_center = (center_in_ref, unit_quat()) # Maps from center frame to origin frame
    urdf_from_center = multiply(invert(ref_from_urdf), ref_from_center)
    return urdf_from_center


def get_urdf_from_base(body, **kwargs):
    return get_urdf_from_z_axis(body, z_fraction=0.0, **kwargs)


def get_urdf_from_center(body, **kwargs):
    return get_urdf_from_z_axis(body, z_fraction=0.5, **kwargs)


def get_urdf_from_top(body, **kwargs):
    return get_urdf_from_z_axis(body, z_fraction=1.0, **kwargs)

ObjInfo = namedtuple('ObjInfo', ['width_scale', 'height_scale', 'mass_scale'])
RANDOMIZED_OBJS = {}

def load_body(name):
    average_mass = MODEL_MASSES[get_type(name)]
    obj_path, color = load_cup_bowl_obj(get_type(name))
    if obj_path is None:
        obj_path, color = get_body_obj(name), apply_alpha(get_color(name))
    return create_obj(obj_path, scale=1, mass=average_mass, color=color)

def randomize_body(name, width_range=(1., 1.), height_range=(1., 1.), mass_range=(1., 1.)):
    average_mass = MODEL_MASSES[get_type(name)]
    obj_path, color = load_cup_bowl_obj(get_type(name))
    if obj_path is None:
        obj_path, color = get_body_obj(name), apply_alpha(get_color(name))
    width_scale = np.random.uniform(*width_range)
    height_scale = np.random.uniform(*height_range)
    #if (width_scale == 1.) and (height_scale == 1.):
    #    return load_pybullet(obj_path)
    transform = np.diag([width_scale, width_scale, height_scale])
    transformed_obj_file = transform_obj_file(read(obj_path), transform)
    transformed_dir = os.path.join(os.getcwd(), 'temp_models/')
    ensure_dir(transformed_dir)
    global RANDOMIZED_OBJS
    transformed_obj_path = os.path.join(transformed_dir, 'transformed_{}_{}'.format(
        len(RANDOMIZED_OBJS), os.path.basename(obj_path)))
    #RANDOMIZED_OBJS[name].append(transformed_obj_path) # pybullet caches obj files
    write(transformed_obj_path, transformed_obj_file)
    # TODO: could scale mass proportionally
    mass_scale = np.random.uniform(*mass_range)
    mass = mass_scale*average_mass
    body = create_obj(transformed_obj_path, scale=1, mass=mass, color=color)
    RANDOMIZED_OBJS[body] = ObjInfo(width_scale, height_scale, mass_scale)
    return body

def empty_generator(generator):
    for _ in generator:
        return False
    return True

def constant_velocity_times(waypoints, velocity=2.0):
    """
    list of quaternion waypoints
    velocity in radians/s
    @return a list of times, [0,0.1,0.2,0.3,0.4] to execute each point in the trajectory
    """
    #goes over each pair of quaternions and finds the time it would take for them to move with a constant velocity
    times_from_start = [0.]
    waypoints_iter = iter(waypoints)
    _, prev_quat = next(waypoints_iter)
    while True:
        try:
            _, curr_quat = next(waypoints_iter)
            delta_angle = quat_angle_between(prev_quat, curr_quat)
            delta_time = delta_angle/velocity
            times_from_start.append(times_from_start[-1] + delta_time)
            prev_quat = curr_quat
        except StopIteration:
            return tuple(times_from_start)


JOINT_WEIGHTS = [0.00016230367961917808, 0.0, 0.0, 0.00016230367961917805, 0.0, 0.0, 0.00016230367961917805, 0.0, 0.0,
                 0.00016230367961917808, 0.0, 0.0, 0.49071007271225575, 0.0014465111641126854, 0.00021121715881044261,
                 1.1282768660252898e-07, 0.013761116604588269, 0.010683362202931692, 0.0016933751296455564,
                 0.0023107356084311205, 0.0002462347011418014, 0.00032742312455272054, 2.7639522730260056e-05,
                 7.521845941628282e-05, 0.0, 1.520752216936565e-05, 0.0, 1.8602487587397702e-05, 1.6056423606073485e-06,
                 3.760922970814143e-05, 0.014327976184012875, 0.010551687565741692, 0.0016437207311802706,
                 0.0022235065469253363, 0.00014284942756410284, 0.00018273259655198583, 2.843136600639648e-05,
                 7.521845941628282e-05, 0.0, 1.5207522169365647e-05, 0.0, 1.8610258021393747e-05, 1.689720960329995e-06,
                 3.7609229708141416e-05, 0.0]


def get_weights_resolutions(robot, arm, weights_regularize=0.005, resolution_scale=0.001):
    arm_joints = get_arm_joints(robot, arm)
    weights = np.array([JOINT_WEIGHTS[i] for i in movable_from_joints(robot, arm_joints)])
    weights += weights_regularize * np.ones(weights.shape)
    resolutions = np.divide(resolution_scale * np.ones(weights.shape), weights)
    return weights, resolutions

def get_random_seed():
    # random.getstate()[1][0]
    return get_numpy_seed()
    return np.random.get_state()[1][0]

def set_seed(seed):
    # These generators are different and independent
    set_random_seed(seed)
    set_numpy_seed(seed)
    print('Seed:', seed)
