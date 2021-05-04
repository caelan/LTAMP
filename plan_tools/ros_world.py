import os
import pybullet as p

#from contextlib import contextmanager
from control_tools.common import BASE_FRAME
from control_tools.pb_controller import PBController
from perception_tools.pb_perception import PBPerception
from perception_tools.common import get_models_path
from plan_tools.common import PR2_URDF
from pybullet_tools.utils import HideOutput, ClientSaver, get_link_pose, link_from_name, set_pose, invert, \
    load_pybullet, disconnect, get_constraints, remove_constraint, \
    remove_all_debug, reset_simulation, connect


class ROSWorld(object):
    def __init__(self, sim_only=False, visualize=False, use_robot=True, **kwargs):
        self.simulation = sim_only
        if not self.simulation:
            import rospy
            rospy.init_node(self.__class__.__name__, anonymous=False)
        self.client_saver = None
        with HideOutput():
            self.client = connect(visualize)  # TODO: causes assert (len(visual_data) == 1) error?
            #self.client = p.connect(p.GUI if visualize else p.DIRECT)

        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, False, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True, physicsClientId=self.client)
        self.pr2 = None
        if use_robot:
            self._create_robot()
        self.initial_beads = {}

        if sim_only:
            self.perception = PBPerception(self, **kwargs)
            self.controller = PBController(self)
            self.alternate_controller = None
        else:
            from perception_tools.retired.ros_perception import ROSPerception
            from control_tools.retired.ros_controller import ROSController
            self.perception = ROSPerception(self)
            self.controller = ROSController(self)
            self.alternate_controller = PBController(self)
            self.sync_controllers()

    @property
    def robot(self):
        return self.pr2

    def _create_robot(self):
        with ClientSaver(self.client):
            with HideOutput():
                pr2_path = os.path.join(get_models_path(), PR2_URDF)
                self.pr2 = load_pybullet(pr2_path, fixed_base=True)

            # Base link is the origin
            base_pose = get_link_pose(self.robot, link_from_name(self.robot, BASE_FRAME))
            set_pose(self.robot, invert(base_pose))
        return self.pr2

    def get_body(self, name):
        return self.perception.get_body(name)

    def get_pose(self, name):
        return self.perception.get_pose(name)

    def sync_controllers(self):
        if self.alternate_controller is None:
            return
        joints1 = self.controller.get_joint_names()
        joints2 = set(self.alternate_controller.get_joint_names())
        common_joints = [j for j in joints1 if j in joints2]
        positions = self.controller.get_joint_positions(common_joints)
        self.alternate_controller.set_joint_positions(common_joints, positions)

    def reset(self, keep_robot=False):
        # Avoid using remove_body(body)
        # https://github.com/bulletphysics/bullet3/issues/2086
        self.controller.reset()
        self.initial_beads = {}
        with ClientSaver(self.client):
            for name in list(self.perception.sim_bodies):
                self.perception.remove(name)
            for constraint in get_constraints():
                remove_constraint(constraint)
            remove_all_debug()
            reset_simulation()
        if keep_robot:
            self._create_robot()

    def stop(self):
        self.reset(keep_robot=False)
        with ClientSaver(self.client):
            disconnect()

    def __enter__(self):
        # TODO: be careful about the nesting of these
        self.client_saver = ClientSaver(self.client)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.client_saver is not None
        self.client_saver.restore()
        self.client_saver = None

    #@contextmanager
    #def __call__(self, *args, **kwargs):
    #    saver = ClientSaver(self.client)
    #    yield
    #    saver.restore()
