#Author: Skye Thompson

#TODO: Rename?

from pybullet_tools.utils import get_pose, ClientSaver, load_pybullet, set_pose, HideOutput
from perception_tools.common import get_body_urdf, get_type
from dimensions.common import load_cup_bowl
from collections import OrderedDict

class Perception(object):
    """ For detecting useful information about the world. Handles robot belief state."""

    def __init__(self, world):
        """Initializes the perception module"""
        self.world = world
        self.client = world.client
        self.robot = world.robot
        self.sim_items = OrderedDict()
        self.sim_surfaces = OrderedDict()
        self.sim_bodies = OrderedDict()

    @property
    def pr2(self):
        return self.robot

    def load_body(self, name, pose=None, fixed_base=False):
        assert name not in self.sim_bodies
        ty = get_type(name)
        with ClientSaver(self.client):
            with HideOutput():
                body = load_cup_bowl(ty)
                if body is None:
                    body = load_pybullet(get_body_urdf(name), fixed_base=fixed_base)
            if pose is not None:
                set_pose(body, pose)
            self.sim_bodies[name] = body
            return body

    def get_items(self):
        return list(self.sim_items)

    def get_surfaces(self):
        return list(self.sim_surfaces)

    def get_body(self, name):
        """
        :param name: The name of a PyBullet body
        :return: The index of the corresponding body
        """
        return self.sim_bodies[name]

    def get_pose(self, name):
        with ClientSaver(self.client):
            return get_pose(self.get_body(name))
        #if name in self.sim_items:
        #    return self.sim_items[name]
        #if name in self.sim_surfaces:
        #    return self.sim_surfaces[name]
        #raise ValueError(name)
