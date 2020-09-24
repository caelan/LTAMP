#!/usr/bin/env python

import pybullet as p
import time
import warnings
import numpy as np

from control_tools.common import PR2_TOOL_FRAMES, BASE_FRAME
from control_tools.controller import Controller  # TODO: wrong interface
from pybullet_tools.utils import get_movable_joints, control_joint, set_joint_positions, set_joint_position, BASE_LINK, \
    get_joint_positions, link_from_name, remove_fixed_constraint, ClientSaver, get_time_step, joints_from_names, \
    add_fixed_constraint, step_simulation, joint_controller_hold2, joint_from_name, enable_gravity, get_joint_name, \
    remove_constraint, get_joint_velocities, get_max_force, get_joint_torque, elapsed_time, get_constraint_info

NULL_EFFORT = -1


class PBController(Controller):

    def __init__(self, world):
        super(PBController, self).__init__()
        self.world = world
        self.execute_motor_control = False
        # Default maxAppliedForce=500.0
        self.attachment_force = 500.0  # None | 300 | 500 | 1000
        self.attachments = {}

    '''
    ===============================================================
                   Get State information
    ===============================================================
    '''

    @property
    def client(self):
        return self.world.client

    @property
    def robot(self):
        return self.world.robot

    def get_joint_names(self):
        with ClientSaver(self.client):
            return [get_joint_name(self.robot, joint)
                    for joint in get_movable_joints(self.robot)]

    def get_joint_positions(self, joint_names):
        # Returns the configuration of the specified joints
        with ClientSaver(self.client):
            joints = joints_from_names(self.robot, joint_names)
            return get_joint_positions(self.robot, joints)

    def set_joint_positions(self, joint_names, positions):
        with ClientSaver(self.client):
            joints = joints_from_names(self.robot, joint_names)
            set_joint_positions(self.robot, joints, positions)

    # return the current Cartesian pose of the gripper
    def return_cartesian_pose(self, arm, frame=BASE_FRAME):
        raise NotImplementedError()

    '''
    ===============================================================
                    Joint Control Commands
    ===============================================================
    '''

    # def command_base(self, x, y, yaw):
    #     # TODO: How to command base in pybullet?
    #     raise NotImplementedError()

    def command_torso(self, pose, timeout=2.0, blocking=True):
        # Commands Torso to a certain position
        with ClientSaver(self.client):
            torso_joint = joint_from_name(self.robot, self.TORSO)
            set_joint_position(self.robot, torso_joint, pose)
            if self.execute_motor_control:
                control_joint(self.robot, torso_joint, pose)
            else:
                set_joint_position(self.robot, torso_joint, pose)

    def rest_for_duration(self, duration):
        if not self.execute_motor_control:
            return
        sim_duration = 0.0
        body = self.robot
        position_gain = 0.02
        with ClientSaver(self.client):
            # TODO: apply to all joints
            dt = get_time_step()
            movable_joints = get_movable_joints(body)
            target_conf = get_joint_positions(body, movable_joints)
            while sim_duration < duration:
                p.setJointMotorControlArray(body, movable_joints, p.POSITION_CONTROL,
                                            targetPositions=target_conf,
                                            targetVelocities=[0.0] * len(movable_joints),
                                            positionGains=[position_gain] * len(movable_joints),
                                            # velocityGains=[velocity_gain] * len(movable_joints),
                                            physicsClientId=self.client)
                step_simulation()
                sim_duration += dt

    def follow_trajectory(self, joints, path, times_from_start=None, real_time_step=0.0,
                          waypoint_tolerance=1e-2 * np.pi, goal_tolerance=5e-3 * np.pi,
                          max_sim_duration=1.0, print_sim_frequency=1.0, **kwargs):
        # real_time_step = 1e-1 # Can optionally sleep to slow down visualization
        start_time = time.time()
        if times_from_start is not None:
            assert len(path) == len(times_from_start)
            current_positions = get_joint_positions(self.robot, joints)
            differences = [(np.array(q2) - np.array(q1)) / (t2 - t1) for q1, t1, q2, t2 in
                           zip([current_positions] + path, [0.] + times_from_start, path, times_from_start)]
            velocity_path = differences[1:] + [np.zeros(len(joints))]  # Using velocity at endpoints
        else:
            velocity_path = [None] * len(path)

        sim_duration = 0.0
        sim_steps = 0
        last_print_sim_duration = sim_duration
        with ClientSaver(self.client):
            dt = get_time_step()
            # TODO: fit using splines to get velocity info
            for num, positions in enumerate(path):
                if self.execute_motor_control:
                    sim_start = sim_duration
                    tolerance = goal_tolerance if (num == len(path) - 1) else waypoint_tolerance
                    velocities = velocity_path[num]
                    for _ in joint_controller_hold2(self.robot, joints, positions, velocities,
                                                    tolerance=tolerance, **kwargs):
                        step_simulation()
                        # print(get_joint_velocities(self.robot, joints))
                        # print([get_joint_torque(self.robot, joint) for joint in joints])
                        sim_duration += dt
                        sim_steps += 1
                        time.sleep(real_time_step)
                        if print_sim_frequency <= (sim_duration - last_print_sim_duration):
                            print(
                                'Waypoint: {} | Simulation steps: {} | Simulation seconds: {:.3f} | Steps/sec {:.3f}'.format(
                                    num, sim_steps, sim_duration, sim_steps / elapsed_time(start_time)))
                            last_print_sim_duration = sim_duration
                        if max_sim_duration <= (sim_duration - sim_start):
                            print('Timeout of {:.3f} simulation seconds exceeded!'.format(max_sim_duration))
                            break
                    # control_joints(self.robot, arm_joints, positions)
                else:
                    set_joint_positions(self.robot, joints, positions)
        print('Followed {} waypoints in {:.3f} simulation seconds and {} simulation steps'.format(
            len(path), sim_duration, sim_steps))

    def command_arm_trajectory(self, arm, path, times_from_start, blocking=True, **kwargs):
        # angles is a list of joint angles, times is a list of times from start
        # When calling joints on an arm, needs to be called with all the angles in the arm
        # times is not used because our pybullet's position controller doesn't take into account times
        assert len(path) == len(times_from_start)
        with ClientSaver(self.client):
            arm_joints = joints_from_names(self.robot, self.get_arm_joint_names(arm))
            self.follow_trajectory(arm_joints, path, times_from_start, **kwargs)
        # TODO(caelan): spline interpolation (actually scipy might only support 1D/2D
        # https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html

    def command_arm(self, arm, positions, timeout, **kwargs):
        return self.command_arm_trajectory(arm, [positions], [timeout], **kwargs)

    ##################################################

    def reset(self):
        for arm, obj in list(self.holding.items()):
            self.detach(arm, obj)

    def command_gripper(self, arm, position, max_effort=NULL_EFFORT, timeout=2.0, blocking=True):
        # position is the width of the gripper as in the physical distance between the two fingers
        with ClientSaver(self.client):
            gripper_joints = joints_from_names(self.robot, self.get_gripper_joint_names(arm))
            positions = [position] * len(gripper_joints)
            self.follow_trajectory(gripper_joints, [positions], max_sim_duration=timeout)

    def attach(self, arm, obj, **kwargs):
        self.holding[arm] = obj
        assert obj not in self.attachments
        body = self.world.get_body(obj)
        with ClientSaver(self.client):
            if arm == 'l':
                frame = "left_gripper"
            elif arm == 'r':
                frame = "right_gripper"
            else:
                raise ValueError("Arm should be l or r but was {}".format(arm))
            robot_link = link_from_name(self.robot, PR2_TOOL_FRAMES[frame])
            self.attachments[obj] = add_fixed_constraint(body, self.robot, robot_link,
                                                         max_force=self.attachment_force)
            # print(get_constraint_info(self.attachments[obj]))

    def detach(self, arm, obj):
        del self.holding[arm]
        if obj in self.attachments:
            with ClientSaver(self.client):
                remove_constraint(self.attachments[obj])
                del self.attachments[obj]

    def set_gravity(self):
        with ClientSaver(self.client):
            enable_gravity()
