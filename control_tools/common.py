from math import pi as PI

from control_tools.ik.pr2_ik import arm_difference_fn

import numpy as np

# NOTE(caelan): these are LTAMP-specific utility functions (separate from utils.py)

BASE_FRAME = 'base_link'

PR2_JOINT_SAFETY_LIMITS = {
    # Taken from /ltamp-pr2/models/pr2_description/pr2.urdf
    # safety_controller: (soft_lower_limit, soft_upper_limit)
    'torso_lift_joint': (0.0115, 0.325),
    'head_pan_joint': (-2.857, 2.857),
    'head_tilt_joint': (-0.3712, 1.29626),
    'laser_tilt_mount_joint': (-0.7354, 1.43353),

    'r_shoulder_pan_joint': (-2.1353981634, 0.564601836603),
    'r_shoulder_lift_joint': (-0.3536, 1.2963),
    'r_upper_arm_roll_joint': (-3.75, 0.65),
    'r_elbow_flex_joint': (-2.1213, -0.15),
    'r_wrist_flex_joint': (-2.0, -0.1),
    'r_gripper_joint': (-0.01, 0.088),

    'l_shoulder_pan_joint': (-0.564601836603, 2.1353981634),
    'l_shoulder_lift_joint': (-0.3536, 1.2963),
    'l_upper_arm_roll_joint': (-0.65, 3.75),
    'l_elbow_flex_joint': (-2.1213, -0.15),
    'l_wrist_flex_joint': (-2.0, -0.1),
    'l_gripper_joint': (-0.01, 0.088),

    # TODO: custom base rotation limit to prevent power cord strangulation
    # TODO: could also just change joints in the URDF
    # 'theta': (-2*np.pi, 2*np.pi),
}

PR2_TOOL_FRAMES = {
    'left_gripper': 'l_gripper_palm_link',
    'right_gripper': 'r_gripper_palm_link',
}


def get_arm_prefix(arm):
    letter = arm[0].lower()
    assert letter in ['l', 'r'], 'Arm not found'
    return letter


def get_arm_joint_names(arm):
    # The names of the movable joints in an arm
    arm_prefix = get_arm_prefix(arm)
    return [
        arm_prefix + '_shoulder_pan_joint',
        arm_prefix + '_shoulder_lift_joint',
        arm_prefix + '_upper_arm_roll_joint',
        arm_prefix + '_elbow_flex_joint',
        arm_prefix + '_forearm_roll_joint',
        arm_prefix + '_wrist_flex_joint',
        arm_prefix + '_wrist_roll_joint',
    ]


##################################################

def adjust_config(next_config, current_config):
    # Removes extra rotations of the wrist and forearm roll joints
    # Adjusts the forearm and wrist roll joints in the next_config to be close to the ones in current_config
    # Guaranteed to be within [-PI,+PI]
    # TODO(kevin): this actually is not guaranteed to be within [-PI,+PI]
    next_config = list(next_config)  # avoid modifying original, just in case
    for i in [4, 6]:
        k = (current_config[i] - next_config[i]) // (2 * PI)
        if (current_config[i] - next_config[i]) % (2 * PI) > PI:
            k += 1
        next_config[i] += 2 * PI * k
    return next_config


def adjust_trajectory(trajectory, current_config):
    # Adjusts the configs in trajectory to match the current one
    # Adjusts the first config first and then adjusts each config to the one preceding it
    # Removes extra rotations of the wrist and forearm.
    # Note: If the trajectory already includes rotations they probably won't be removed
    # Sparse trajectories (where joints may be more than one rotation apart for wrist and forearm roll) will lose rotations
    trajectory[0] = adjust_config(trajectory[0], current_config)
    for i in range(1, len(trajectory)):
        trajectory[i] = adjust_config(trajectory[i], trajectory[i - 1])
    return trajectory


def generate_times(trajectory, speed):
    # given a trajectory, generate the times so that it has constant velocity
    # the first point in the path gets time=0
    # total_time is the total time for the trajectory. speed is the maximum speed for the trajectory
    # if you supply just one of them, it goes with that one
    # if you supply both, it goes with the slower one. if you provide neither, default to min(self.JOINT_SPEED, self.CARTESIAN_SPEED)
    # dist is a distance function. If none is provided, length 7 lists get arm_distance, poses get pose_distance, and everything else gets euclidean distance
    # poses need to be in the form [position, quaternion]
    # TODO: write a version to take a function for converting distance to time. t(dist, total_dist)
    # TODO: add a buffer to the initial time
    # distance_fn = lambda q1, q2: np.linalg.norm(np.array(q2) - np.array(q1))
    distance_fn = lambda q1, q2: np.max(np.abs(arm_difference_fn(q1, q2)))
    distances = [distance_fn(trajectory[i], trajectory[i + 1])
                 for i in range(len(trajectory) - 1)]
    cumulative_distances = np.cumsum([0.] + distances)
    return [partial_dist / speed for partial_dist in cumulative_distances]
