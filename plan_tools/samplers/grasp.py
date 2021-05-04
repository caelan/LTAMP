import numpy as np
import math

from collections import namedtuple

from itertools import islice
from control_tools.common import get_arm_prefix
from plan_tools.common import get_reference_pose, is_obj_type, get_type, CUP_WITH_LIP
from plan_tools.samplers.generators import TOOL_FRAMES, TOOL_POSE, set_gripper_position
from pybullet_tools.pr2_utils import get_gripper_joints, get_top_grasps, get_side_cylinder_grasps, \
    get_top_cylinder_grasps, get_edge_cylinder_grasps, close_until_collision
from pybullet_tools.utils import get_unit_vector, multiply, Pose, link_from_name, Attachment, get_link_pose, set_pose, \
    approximate_as_prism, Euler, Point, BodySaver, INF
from dimensions.cups.dimensions import OLIVE_CUPS, THREE_D_CUPS

TOP_DIRECTION = get_unit_vector([1, 0, 0])

# TODO: unify these properties
# TODO: compute thickness using the bounding box
Spoon = namedtuple('Spoon', ['length', 'height', 'diameter', 'thickness'])

SPOON_THICKNESSES = {
    'grey_spoon': 0.003,
    'orange_spoon': 0.004,
    'green_spoon': 0.005,
}

SPOON_DIAMETERS = {
    'grey_spoon': 0.045,
    'orange_spoon': 0.055,
    'green_spoon': 0.065,
}

SPOON_LENGTHS = {
    'grey_spoon': 0.12,
    'orange_spoon': 0.15,
    'green_spoon': 0.18,
}

# Almost half the diameter
SPOON_HEIGHTS = {
    'grey_spoon': 0.02,
    'orange_spoon': 0.025,
    'green_spoon': 0.035,
}

def hemisphere_volume(radius, height):
    delta = height - radius
    return 0.5*math.pi*(math.pow(radius, 2)*delta - math.pow(delta, 3) / 3 + 2*math.pow(radius, 3) / 3 )

FINGER_LENGTH = 0.035
FINGER_WIDTH = 0.02

HAND_LENGTH = 0.07
HAND_WIDTH = 0.085

class Grasp(object):
    def __init__(self, obj_name, index, grasp_pose, pre_direction, grasp_width, effort=25):
        self.obj_name = obj_name
        self.index = index
        self.grasp_pose = grasp_pose
        self.pre_direction = pre_direction
        self.pregrasp_pose = multiply(Pose(point=pre_direction), grasp_pose)
        self.grasp_width = grasp_width
        self.effort = effort
    def get_attachment(self, world, arm):
        return get_grasp_attachment(world, arm, self)
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.index)

##################################################

def get_grasp_attachment(world, arm, grasp):
    tool_link = link_from_name(world.robot, TOOL_FRAMES[arm])
    body = world.get_body(grasp.obj_name)
    return Attachment(world.robot, tool_link, grasp.grasp_pose, body)


def compute_grasp_width(robot, arm, body, grasp_pose, **kwargs):
    gripper_joints = get_gripper_joints(robot, arm)
    tool_link = link_from_name(robot, TOOL_FRAMES[arm])
    tool_pose = get_link_pose(robot, tool_link)
    body_pose = multiply(tool_pose, grasp_pose)
    set_pose(body, body_pose)
    return close_until_collision(robot, gripper_joints, bodies=[body], max_distance=0.0, **kwargs)


def get_spoon_grasps(name, body, under=False, grasp_length=0.02):
    # TODO: scale thickness based on size
    thickness = SPOON_THICKNESSES[get_type(name)]
    # Origin is in the center of the spoon's hemisphere head
    # grasp_length depends on the grasp position
    center, extent = approximate_as_prism(body)
    for k in range(1+under):
        rotate_y = Pose(euler=Euler(pitch=-np.pi/2 + np.pi*k))
        rotate_z = Pose(euler=Euler(yaw=-np.pi / 2))
        length = (-center + extent/2)[1] - grasp_length
        translate_x = Pose(point=Point(x=length, y=-thickness/2.))
        gripper_from_spoon = multiply(translate_x, rotate_z, rotate_y)
        yield gripper_from_spoon


def compute_grasps(world, name, grasp_length=(HAND_LENGTH-0.02),
                   pre_distance=0.1, max_attempts=INF):
    body = world.get_body(name)
    reference_pose = get_reference_pose(name)
    # TODO: add z offset in the world frame
    pre_direction = pre_distance*TOP_DIRECTION
    ty = get_type(name)
    if is_obj_type(name, 'block'):
        generator = get_top_grasps(body, under=False, tool_pose=TOOL_POSE,
                                   body_pose=reference_pose, grasp_length=grasp_length, max_width=np.inf)
    elif is_obj_type(name, 'cup'):
        #pre_direction = pre_distance*get_unit_vector([1, 0, -2]) # -x, -y, -z
        pre_direction = pre_distance*get_unit_vector([3, 0, -1]) # -x, -y, -z
        # Cannot pick if top_offset is too large(0.03 <=)
        top_offset = 3*FINGER_WIDTH/4 if ty in CUP_WITH_LIP else FINGER_WIDTH/4
        generator = get_side_cylinder_grasps(body, under=False, tool_pose=TOOL_POSE, body_pose=reference_pose,
                                             grasp_length=grasp_length, top_offset=top_offset,
                                             max_width=np.inf)
    elif is_obj_type(name, 'stirrer'):
        generator = get_top_cylinder_grasps(body, tool_pose=TOOL_POSE, body_pose=reference_pose,
                                            grasp_length=grasp_length, max_width=np.inf)
    elif is_obj_type(name, 'bowl'):
        generator = get_edge_cylinder_grasps(body, tool_pose=TOOL_POSE, body_pose=reference_pose,
                                             grasp_length=0.02)
    elif is_obj_type(name, 'spoon'):
        generator = get_spoon_grasps(name, body)
    else:
        raise NotImplementedError(name)

    effort = 25 if ty in (OLIVE_CUPS + THREE_D_CUPS) else 50
    for index, grasp_pose in enumerate(islice(generator, None if max_attempts is INF else max_attempts)):
        with BodySaver(world.robot):
            grasp_width = compute_grasp_width(world.robot, 'left', body, grasp_pose)
        #print(index, grasp_width)
        if grasp_width is not None:
            yield Grasp(name, index, grasp_pose, pre_direction, grasp_width, effort)


def get_grasp_gen_fn(world, **kwargs):
    #initial_poses = {obj: get_pose(world.bodies[obj]) for obj in world.items}
    def gen_fn(obj):
        #grasp = get_pick_grasp(obj, list_from_pair_pose(initial_poses[obj]))
        #yield (grasp,)
        for grasp in compute_grasps(world, obj, **kwargs):
            yield (grasp,)
    return gen_fn

##################################################

def hold_item(world, arm, name):
    try:
        grasp, = next(get_grasp_gen_fn(world)(name))
    except StopIteration:
        return None
    set_gripper_position(world.robot, arm, grasp.grasp_width)
    attachment = get_grasp_attachment(world, arm, grasp)
    attachment.assign()
    world.controller.attach(get_arm_prefix(arm), name)
    return {arm: grasp}


def has_grasp(world, name, max_attempts=4):
    with BodySaver(world.get_body(name)):
        try:
            next(get_grasp_gen_fn(world, max_attempts=max_attempts)(name))
        except StopIteration:
            return False
    return True
