import numpy as np
import random

from control_tools.common import adjust_trajectory, get_arm_prefix, generate_times
from pybullet_tools.utils import set_joint_positions, get_extend_fn, get_joint_positions, is_circular, \
    joints_from_names, get_difference_fn
from pybullet_tools.pr2_utils import get_arm_joints, get_gripper_joints
from plan_tools.common import create_attachment


# TODO: add extra constraint on tool/gripper to exert more force
# TODO: plan path for free body (e.g. just gripper)

OBJECT_EFFORT = 25

class Command(object):
    def iterate(self, world, attachments):
        # attachments is a part of the state separate from pybullet
        # TODO: more generically maintain a state class
        raise NotImplementedError()

    def execute(self, controller, joint_speed):
        raise NotImplementedError()


class GripperCommand(Command):
    def __init__(self, arm, position, effort=OBJECT_EFFORT, timeout=1.0):
        self.arm = arm
        self.position = position
        self.effort = effort
        self.timeout = timeout
        # self.max_effort = 100 # Too strong for the cup

    def iterate(self, world, attachments):
        joints = get_gripper_joints(world.robot, self.arm)
        start_conf = get_joint_positions(world.robot, joints)
        end_conf = [self.position] * len(joints)
        extend_fn = get_extend_fn(world.robot, joints)
        path = [start_conf] + list(extend_fn(start_conf, end_conf))
        for positions in path:
            set_joint_positions(world.robot, joints, positions)
            yield positions

    def execute(self, controller, joint_speed):
        controller.command_gripper(get_arm_prefix(self.arm), self.position,
                                   max_effort=self.effort, timeout=self.timeout)
        return True


class ArmTrajectory(Command):
    def __init__(self, arm, path, dialation=1.):
        self.arm = arm
        self.path = path
        self.dialation = dialation # time dialation
        # self.teleport = teleport
        # self.times_from_start = times_from_start
        # if self.times_from_start is not None:
        #    assert len(self.path) == self.times_from_start

    def iterate(self, world, attachments):
        joints = get_arm_joints(world.robot, self.arm)
        # gripper_joints = get_gripper_joints(robot, arm)
        for positions in self.path:
            set_joint_positions(world.robot, joints, positions)
            yield positions

    def execute(self, controller, joint_speed):
        if not self.path:
            return
        # if self.teleport:
        #    for _ in self.iterate(controller.world, attachments={}):
        #        pass
        arm_prefix = get_arm_prefix(self.arm)
        robot = controller.world.robot
        arm_joints = joints_from_names(robot, controller.get_arm_joint_names(arm_prefix))
        current_positions = controller.get_arm_positions(arm_prefix)
        # For debugging
        #modification = [random.randint(-10, +10) * np.pi if is_circular(robot, joint) else 0 for joint in arm_joints]
        #current_positions = current_positions + np.array(modification)

        difference_fn = get_difference_fn(robot, arm_joints)
        differences = [difference_fn(q2, q1) for q1, q2 in zip(self.path, self.path[1:])]

        adjusted_path = [np.array(current_positions)]
        for difference in differences:
            adjusted_path.append(adjusted_path[-1] + difference)

        #adjusted_path = adjust_trajectory(self.path, current_positions)
        #for i, (q1, q2) in enumerate(zip(self.path, trajectory)):
        #    print(i, (np.array(q2) - np.array(q1)).round(3).tolist())
        times_from_start = generate_times(adjusted_path, joint_speed / self.dialation)
        controller.command_arm_trajectory(arm_prefix, adjusted_path[1:],
                                          times_from_start=times_from_start[1:])
        controller.command_arm(arm_prefix, adjusted_path[-1], timeout=1.0)
        return True


class Rest(Command):
    def __init__(self, duration):
        self.duration = duration

    def iterate(self, world, attachments):
        yield

    def execute(self, controller, joint_speed):
        controller.rest_for_duration(self.duration)
        return True

class Push(Command):
    def __init__(self, arm, obj):
        self.arm = arm
        self.obj = obj

    def iterate(self, world, attachments):
        obj_body = world.get_body(self.obj)
        attachments[self.obj] = create_attachment(world.robot, self.arm, obj_body)
        yield

    def execute(self, controller, joint_speed):
        return True

class Attach(Command):
    def __init__(self, arm, obj, effort=OBJECT_EFFORT):
        self.arm = arm
        self.obj = obj
        self.effort = effort

    def iterate(self, world, attachments):
        obj_body = world.get_body(self.obj)
        attachments[self.obj] = create_attachment(world.robot, self.arm, obj_body)
        yield

    def execute(self, controller, joint_speed):
        controller.attach(get_arm_prefix(self.arm), self.obj, max_effort=self.effort)
        return True


class Detach(Command):
    def __init__(self, arm, obj):
        self.arm = arm
        self.obj = obj

    def iterate(self, world, attachments):
        if self.obj in attachments:
            del attachments[self.obj]
        yield

    def execute(self, controller, joint_speed):
        controller.detach(get_arm_prefix(self.arm), self.obj)
        return True

# TODO: could deprecate returned boolean

####################################################################################################

def execute_plan(world, plan, joint_speed=0.05 * np.pi, default_sleep=1.0):
    # TODO: effort along trajectory
    # http://docs.ros.org/api/control_msgs/html/action/FollowJointTrajectory.html
    # TODO: singular Jacobian at pouring configurations?
    # http://wiki.ros.org/joint_trajectory_controller
    if plan is None:
        return False
    # TODO: return the number of pybullet steps taken
    for name, args in plan:
        control = args[-1]
        for command in control['commands']:
            # times = control['times']
            if not world.controller.check_state():
                return False
            success = command.execute(world.controller, joint_speed)
            world.controller.rest_for_duration(default_sleep)
            if not success:
                return False
    return world.controller.check_state()
