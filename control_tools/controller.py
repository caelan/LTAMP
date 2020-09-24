#!/usr/bin/env python

import numpy as np

MIN_HOLDING = 0.01

class Controller(object):
    RIGHT_ARM = ['r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint', 'r_elbow_flex_joint',
                 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint']
    LEFT_ARM = ['l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_upper_arm_roll_joint', 'l_elbow_flex_joint',
                'l_forearm_roll_joint', 'l_wrist_flex_joint', 'l_wrist_roll_joint']
    RIGHT_GRIPPER = ['r_gripper_r_finger_joint', 'r_gripper_l_finger_joint',
                     'r_gripper_r_finger_tip_joint', 'r_gripper_l_finger_tip_joint']
    LEFT_GRIPPER = ['l_gripper_r_finger_joint', 'l_gripper_l_finger_joint',
                    'l_gripper_r_finger_tip_joint', 'l_gripper_l_finger_tip_joint']
    TORSO = 'torso_lift_joint'
    HEAD = ['head_pan_joint', 'head_tilt_joint']

    def __init__(self):
        self.holding = {}

    def get_joint_names(self):
        raise NotImplementedError()

    def get_arm_joint_names(self, arm):
        if arm == 'l':
            return self.LEFT_ARM
        elif arm == 'r':
            return self.RIGHT_ARM
        raise ValueError(arm)

    def get_gripper_joint_names(self, arm):
        if arm == 'l':
            return self.LEFT_GRIPPER
        elif arm == 'r':
            return self.RIGHT_GRIPPER
        raise ValueError(arm)

    def get_arm_positions(self, arm):
        return self.get_joint_positions(self.get_arm_joint_names(arm))

    def get_joint_positions(self, joint_names):
        # Works for multiple or single inputs
        # Returns list or single object matching joint_ids
        raise NotImplementedError()

    def get_joint_velocities(self, joint_names):
        # Works for multiple or single inputs
        # Returns list or single object matching joint_ids
        raise NotImplementedError()

    # There are separate commands for the arms, grippers, head, and torso because of how the ros nodes are set up.
    # Only position control available for now
    def command_arm(self, arm, positions, timeout, **kwargs):
        # arm is 'l' or 'r' for left or right, only one at a time
        # The positions must be supplied for every joint in the arm
        # The gripper is not included in the arm
        raise NotImplementedError()

    def rest_for_duration(self, duration):
        # Controller holds its current positions for a specified duration
        raise NotImplementedError()

    def command_arm_trajectory(self, arm, positions, times_from_start, blocking):
        # arm is 'l' or 'r' for left or right, only one at a time
        # The positions must be supplied for every joint in the arm
        # The gripper is not included in the arm
        raise NotImplementedError()

    def command_gripper(self, arm, position, max_effort, timeout, blocking):
        # arm is 'l' or 'r' for left or right, only one at a time
        raise NotImplementedError()

    def command_head(self, angles, timeout, blocking):
        # angles in the form [pan, tilt]
        raise NotImplementedError()

    def attach(self, arm, obj):
        raise NotImplementedError()

    def detach(self, arm, obj):
        raise NotImplementedError()

    def open_gripper(self, arm, **kwargs):
        self.command_gripper(arm, position=0.548, **kwargs)

    def close_gripper(self, arm, **kwargs):
        self.command_gripper(arm, position=0.0, **kwargs)

    def reset(self):
        raise NotImplementedError()

    def check_state(self):
        for arm in self.holding:
            joints = self.get_gripper_joint_names(arm)
            positions = self.get_joint_positions(joints)  # They should all be about the same
            position = np.average(positions)
            if position < MIN_HOLDING:
                print('Failure! {} gripper is closed'.format(arm))
                return False
        return True
