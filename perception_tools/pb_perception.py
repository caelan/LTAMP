#!/usr/bin/env python

from perception_tools.perception import Perception, set_pose
from pybullet_tools.utils import ClientSaver
from perception_tools.common import is_fixed, is_item

class PBPerception(Perception):
    def __init__(self, world, state={}):
        super(PBPerception, self).__init__(world)
        for name, pose in state.items():
            if is_fixed(name):
                self.add_surface(name, pose)
            elif is_item(name):
                self.add_item(name, pose)
            else:
                raise ValueError(name)

    def set_pose(self, name, pos, orn):
        pose = (pos, orn)
        with ClientSaver(self.client):
            set_pose(self.get_body(name), pose)

    def add_surface(self, surface, pose=None):
        # TODO: move to Perception
        self.sim_surfaces[surface] = None # pose
        return self.load_body(surface, pose=pose, fixed_base=True)

    def add_item(self, item, pose=None):
        self.sim_items[item] = None # pose
        return self.load_body(item, pose=pose, fixed_base=False)

    def remove(self, name):
        if name in self.sim_surfaces:
            del self.sim_surfaces[name]
        if name in self.sim_items:
            del self.sim_items[name]
        if name in self.sim_bodies:
            #with ClientSaver(self.client):
            #    remove_body(self.sim_bodies[name])
            del self.sim_bodies[name]
