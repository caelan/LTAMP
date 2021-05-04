#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import rospy
import tf
import time
import os
from collections import namedtuple

from pr2_controllers_msgs.msg import JointTrajectoryAction, JointTrajectoryGoal, SingleJointPositionAction, \
    SingleJointPositionGoal, Pr2GripperCommandAction, Pr2GripperCommandGoal
from pr2_gripper_sensor_msgs.msg import PR2GripperEventDetectorAction

from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import Twist
from actionlib import SimpleActionClient
from sensor_msgs.msg import JointState
from actionlib_msgs.msg import GoalStatus
from moveit_msgs.msg import DisplayRobotState, DisplayTrajectory, RobotState, RobotTrajectory
#from soundplay_msgs.msg import SoundRequest

from control_tools.common import get_arm_joint_names, BASE_FRAME, get_arm_prefix
from control_tools.controller import Controller


def make_twist(x, y, yaw):
    twist = Twist()
    twist.linear.x = x
    twist.linear.y = y
    twist.linear.z = 0
    twist.angular.x = 0
    twist.angular.y = 0
    twist.angular.z = yaw
    return twist


# from actionlib_msgs/GoalStatus
# http://docs.ros.org/melodic/api/actionlib_msgs/html/msg/GoalStatus.html
# PENDING = 0
# ACTIVE = 1
# PREEMPTED = 2
# SUCCEEDED = 3
# ABORTED = 4
# REJECTED = 5
# PREEMPTING = 6
# RECALLING = 7
# RECALLED = 8
# LOST = 9

# http://wiki.ros.org/pr2_controllers/Tutorials/Moving%20the%20arm%20using%20the%20Joint%20Trajectory%20Action
# http://wiki.ros.org/joint_trajectory_action
# TODO: constraints/goal_time
# /l_arm_controller/joint_trajectory_action_node/constraints/goal_time
# /opt/ros/indigo/share/pr2_controller_configuration/pr2_arm_controllers.yaml
# http://wiki.ros.org/robot_mechanism_controllers/JointTrajectoryActionController
# http://wiki.ros.org/pr2_controller_configuration
# vim /opt/ros/indigo/share/pr2_controller_configuration/pr2_arm_controllers.yaml
# rostopic echo /l_arm_controller/state
# http://docs.ros.org/api/actionlib/html/simple__action__client_8py_source.html
# http://docs.ros.org/api/actionlib/html/classactionlib_1_1simple__action__client_1_1SimpleActionClient.html
# The controller will interpolate between these points using cubic splines.
# rosservice info /l_arm_controller/query_state
# http://wiki.ros.org/pr2_controller_manager/safety_limits
# https://www.clearpathrobotics.com/wp-content/uploads/2014/08/pr2_manual_r321.pdf
# http://wiki.ros.org/robot_mechanism_controllers/JointSplineTrajectoryController
# rosnode info /realtime_loop

Client = namedtuple('Client', ['topic', 'action'])
MAX_EFFORT = 100.0
INFINITE_EFFORT = -1


##################################################

class ROSController(Controller):
    simple_clients = {
        'torso': Client('torso_controller/position_joint_action', SingleJointPositionAction),
        'head': Client('head_traj_controller/joint_trajectory_action', JointTrajectoryAction),
        'l_joint': Client('l_arm_controller/joint_trajectory_action', JointTrajectoryAction),
        'l_gripper_event': Client('l_gripper_sensor_controller/event_detector', PR2GripperEventDetectorAction),
        'l_gripper': Client('l_gripper_controller/gripper_action', Pr2GripperCommandAction),
        # Can comment out
        'r_joint': Client('r_arm_controller/joint_trajectory_action', JointTrajectoryAction),
        'r_gripper': Client('r_gripper_controller/gripper_action', Pr2GripperCommandAction),
        'r_gripper_event': Client('r_gripper_sensor_controller/event_detector', PR2GripperEventDetectorAction),
    }

    INIT_LOGS = True
    ERROR_LOGS = True
    COMMAND_LOGS = True
    ARMS = ['right', 'left']
    #ARMS = ['left']

    # XXX do subscriber callback nicely

    '''
    ============================================================================
                    Initializing all controllers    
    ============================================================================
    '''

    def __init__(self, world, verbose=True):
        super(ROSController, self).__init__()
        self.world = world
        self.timeout = 2.0
        self.clients = {}
        self.tf_listener = tf.TransformListener()

        # Not convinced that this is actually working
        for arm in self.ARMS:
            client_name = '{}_joint'.format(get_arm_prefix(arm))
            client = self.simple_clients[client_name]
            #goal_param = '{}_node/constraints/goal_time'.format(client.topic)
            #rospy.set_param(goal_param, 1.0)
            for joint_name in get_arm_joint_names(arm):
                joint_param = '{}_node/constraints/{}/goal'.format(client.topic, joint_name)
                rospy.set_param(joint_param, 1e-3)

        if self.INIT_LOGS and verbose:
            rospy.loginfo("starting simple action clients")
        for client_name in self.simple_clients:
            self.clients[client_name] = SimpleActionClient(*self.simple_clients[client_name])
            if self.INIT_LOGS and verbose:
                rospy.loginfo("%s client started" % client_name)

        for client_name in self.clients:
            result = self.clients[client_name].wait_for_server(rospy.Duration(0.1))
            self.clients[client_name].cancel_all_goals()
            if result:
                if self.INIT_LOGS and verbose:
                    rospy.loginfo("%s done initializing" % client_name)
            else:
                if self.ERROR_LOGS:
                    rospy.loginfo("Failed to start %s" % client_name)

        self.rate = rospy.Rate(10)
        if self.INIT_LOGS:
            rospy.loginfo("Subscribing to state messages")

        self.joint_state = None
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_sub = rospy.Subscriber("joint_states", JointState, self._jointCB)

        # Base control copied from teleop_base_head.py
        # Consider switching this to a simple action client to match the others
        self.base_pub = rospy.Publisher('base_controller/command', Twist, queue_size=1)
        self.base_speed = rospy.get_param("~speed", 0.5)  # Run: 1.0
        self.base_turn = rospy.get_param("~turn", 1.0)  # Run: 1.5

        self.display_trajectory_pub = rospy.Publisher('ros_controller/display_trajectory', DisplayTrajectory,
                                                      queue_size=1)
        self.robot_state_pub = rospy.Publisher('ros_controller/robot_state', DisplayRobotState, queue_size=1)
        #self.sound_pub = rospy.Publisher('robotsound', SoundRequest, queue_size=1)

        '''
        TODO: Move to ros perception section
        self.gripper_events_sub= {}
        self.gripper_event = {}
        self.gripper_events_sub['l'] = rospy.Subscriber(\
                "/l_gripper_sensor_controller/event_detector_state",\
                PR2GripperEventDetectorData, self.l_gripper_eventCB)
        self.gripper_events_sub['r'] = rospy.Subscriber(\
                "/r_gripper_sensor_controller/event_detector_state",\
                PR2GripperEventDetectorData, self.r_gripper_eventCB)
        '''

        # Contains [[time, joints], ...]
        # The first element is the start time and the joint names
        # All the later elements are the times relative to the start time and the joint values
        self.joint_logs = []
        self.wait_until_ready()
        if self.INIT_LOGS and verbose:
            rospy.loginfo("Done initializing PR2 Controller!")
        #for arm in ['l', 'r']:
        #    self.open_gripper(arm)

    def get_joint_names(self):
        return list(self.joint_state.name)

    def reset(self):
        pass

    def speak(self, phrase):
        os.system('rosrun sound_play say.py "{}"'.format(phrase))
        #request = SoundRequest()
        #self.sound_pub.publish(request)


    '''
    =============================================================== #XXX make these all nice :)
                    State subscriber callbacks    
    ===============================================================
    '''

    def get_robot_state(self):
        # pose = pose_from_trans(self.get_world_pose(BASE_FRAME)) # TODO: could get this transform directly
        # transform = Transform(Vector3(*point_from_pose(pose)), Quaternion(*quat_from_pose(pose)))
        # transform = self.get_transform()
        # if transform is None:
        #    return None
        state = RobotState()
        state.joint_state = self.joint_state
        # state.multi_dof_joint_state.header.frame_id = '/base_footprint'
        # state.multi_dof_joint_state.header.stamp = rospy.Time(0)
        # state.multi_dof_joint_state.joints = ['world_joint']
        # state.multi_dof_joint_state.transforms = [transform]
        # 'world_joint'
        # http://cram-system.org/tutorials/intermediate/moveit
        state.attached_collision_objects = []
        state.is_diff = False
        # rostopic info /joint_states
        return state

    def get_display_trajectory(self, *joint_trajectories):
        display_trajectory = DisplayTrajectory()
        display_trajectory.model_id = 'pr2'
        for joint_trajectory in joint_trajectories:
            robot_trajectory = RobotTrajectory()
            robot_trajectory.joint_trajectory = joint_trajectory
            # robot_trajectory.multi_dof_joint_trajectory = ...
            display_trajectory.trajectory.append(robot_trajectory)
        display_trajectory.trajectory_start = self.get_robot_state()
        return display_trajectory

    def publish_joint_trajectories(self, *joint_trajectories):
        display_trajectory = self.get_display_trajectory(*joint_trajectories)
        display_state = DisplayRobotState()
        display_state.state = display_trajectory.trajectory_start
        # self.robot_state_pub.publish(display_state)
        self.display_trajectory_pub.publish(display_trajectory)
        # raw_input('Continue?')

        last_trajectory = joint_trajectories[-1]
        last_conf = last_trajectory.points[-1].positions
        joint_state = display_state.state.joint_state
        joint_state.position = list(joint_state.position)
        for joint_name, position in zip(last_trajectory.joint_names, last_conf):
            joint_index = joint_state.name.index(joint_name)
            joint_state.position[joint_index] = position
        self.robot_state_pub.publish(display_state)
        # TODO: record executed trajectory and overlay them
        return display_trajectory

    ##################################################

    def not_ready(self):
        return self.joint_state is None

    def wait_until_ready(self, timeout=5.0):
        end_time = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and (rospy.Time.now() < end_time) and self.not_ready():
            self.rate.sleep()
        if self.not_ready():
            if self.ERROR_LOGS:
                rospy.loginfo("Warning! Did not complete subscribing")
        else:
            if self.INIT_LOGS:
                rospy.loginfo("Robot ready")

    def _jointCB(self, data):
        self.joint_state = data
        self.joint_positions = dict(zip(data.name, data.position))
        self.joint_velocities = dict(zip(data.name, data.velocity))
        self._loggingCB()

    def _loggingCB(self):
        pass

    # joints is a list of joint names for get_joint_positions
    def start_joint_logging(self, joints):
        start_time = rospy.Time.now().to_sec()
        self.joint_logs = [[start_time] + joints]

        def _logging_func():
            elapsed_time = rospy.Time.now().to_sec() - start_time
            self.joint_logs.append([elapsed_time] + self.get_joint_positions(joints))

        self._loggingCB = _logging_func

    def get_joint_log(self):
        return self.joint_logs

    def stop_joint_logging(self):
        self._loggingCB = lambda: None

    '''
    ===============================================================
                   Get State information    
    ===============================================================
    '''

    def get_joint_positions(self, joint_names):
        if isinstance(joint_names, str):
            return self.joint_positions[joint_names] if joint_names in self.joint_positions else None
        return [self.joint_positions[joint] if joint in self.joint_positions else None
                for joint in joint_names]

    def get_joint_velocities(self, joint_names):
        if isinstance(joint_names, str):
            return self.joint_velocities[joint_names] if joint_names in self.joint_positions else None
        return [self.joint_velocities[joint] if joint in self.joint_positions else None
                for joint in joint_names]

    # return the current Cartesian pose of the gripper
    def return_cartesian_pose(self, arm, frame=BASE_FRAME):
        end_time = rospy.Time.now() + rospy.Duration(5)
        link = arm + "_gripper_tool_frame"
        while not rospy.is_shutdown() and (rospy.Time.now() < end_time):
            try:
                t = self.tf_listener.getLatestCommonTime(frame, link)
                (trans, rot) = self.tf_listener.lookupTransform(frame, link, t)
                # if frame == 'base_link' and self.COMMAND_LOGS:
                #    expected_trans, expected_rot = arm_fk(arm, self.get_arm_positions(arm), self.get_torso_position())
                #    error_threshold = 0.05 # 5 cm position difference
                #    if any([abs(t - e) > error_threshold for t, e in zip(trans, expected_trans)]):
                #        rospy.loginfo("TF position does not match FK position")
                #        rospy.loginfo("TF Pose: " + str([trans, rot]))
                #        rospy.loginfo("FK Pose: " + str([expected_trans, expected_rot]))
                return list(trans), list(rot)
            except (tf.Exception, tf.ExtrapolationException):
                rospy.sleep(0.5)
                # current_time = rospy.get_rostime()
                if self.COMMAND_LOGS:
                    rospy.logerr("Waiting for a tf transform between %s and %s" % (frame, link))
        if self.ERROR_LOGS:
            rospy.logerr("Return_cartesian_pose waited 10 seconds tf transform!  Returning None")
        return None, None

    '''
    ===============================================================
                Send Commands for Action Clients                
    ===============================================================
    '''

    def _send_command(self, client, goal, blocking=False, timeout=None):
        if client not in self.clients:
            return False
        self.clients[client].send_goal(goal)
        start_time = rospy.Time.now()
        # rospy.loginfo(goal)
        rospy.sleep(0.1)
        if self.COMMAND_LOGS:
            rospy.loginfo("Command sent to %s client" % client)
        if not blocking:  # XXX why isn't this perfect?
            return None

        status = 0
        end_time = rospy.Time.now() + rospy.Duration(timeout + 0.1)
        while (not rospy.is_shutdown()) and \
                (rospy.Time.now() < end_time) and \
                (status < GoalStatus.SUCCEEDED) and \
                (type(self.clients[client].action_client.last_status_msg) != type(None)):
            status = self.clients[client].action_client.last_status_msg.status_list[-1].status  # XXX get to 80
            self.rate.sleep()

        # It's reporting time outs that are too early
        # FIXED: See comments about rospy.Time(0)
        text = GoalStatus.to_string(status)
        if GoalStatus.SUCCEEDED <= status:
            if self.COMMAND_LOGS:
                rospy.loginfo("Goal status {} achieved. Exiting".format(text))
        else:
            if self.ERROR_LOGS:
                rospy.loginfo("Ending due to timeout {}".format(text))

        result = self.clients[client].get_result()
        elapsed_time = (rospy.Time.now() - start_time).to_sec()
        print('Executed in {:.3f} seconds. Predicted to take {:.3f} seconds.'.format(elapsed_time, timeout))
        #state = self.clients[client].get_state()
        #print('Goal state {}'.format(GoalStatus.to_string(state)))
        # get_goal_status_text
        return result

    def command_base(self, x, y, yaw):
        # This doesn't use _send_command because the base uses a different publisher. Probably could be switched later.
        # Don't forget that x, y, and yaw are multiplied by self.base_speed and self.base_turn.
        # Recommended values for x, y, and yaw are {-1, 0, 1}. Consider cutting out the speed multipliers.
        # The motion is pretty jerky. Consider sending a smoother acceleration trajectory.
        # We can also just re-register using perception after it stops.
        # +x is forward, -x is backward, +y is left, -y is right, +yaw is ccw looking down from above, -yaw is cw
        # Once the robot gets sent a base command, it continues on that velocity. Remember to send a stop command.
        twist = make_twist(x * self.base_speed, y * self.base_speed, yaw * self.base_turn)
        self.base_pub.publish(twist)

    def command_torso(self, pose, timeout, blocking=True):
        goal = SingleJointPositionGoal(
            position=pose,
            min_duration=rospy.Duration(timeout),
            max_velocity=1.0)
        return self._send_command('torso', goal, blocking=blocking, timeout=timeout)

    def command_head(self, angles, timeout, blocking=True):
        goal = JointTrajectoryGoal()
        goal.trajectory.joint_names = ['head_pan_joint', 'head_tilt_joint']
        point = JointTrajectoryPoint()
        point.positions = angles
        point.time_from_start = rospy.Duration(timeout)
        goal.trajectory.points.append(point)
        if blocking:
            self.clients['head'].send_goal_and_wait(goal)
        else:
            self.clients['head'].send_goal(goal)
        return None

    # Sending a negative max_effort means no limit for maximum effort.
    def command_gripper(self, arm, position, max_effort=MAX_EFFORT, timeout=2.0, blocking=True):
        goal = Pr2GripperCommandGoal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        client = "%s_gripper" % arm
        return self._send_command(client, goal, blocking=blocking, timeout=timeout)

    '''
    ===============================================================
                    Joint Control Commands 
    ===============================================================
    '''

    def rest_for_duration(self, duration):
        time.sleep(duration)

    def command_arm_trajectory(self, arm, angles, times_from_start, blocking=True, logging=False, time_buffer=5.0):
        # angles is a list of joint angles, times is a list of times from start
        # When calling joints on an arm, needs to be called with all the angles in the arm
        # rospy.Duration is fine with taking floats, so the times can be floats
        assert len(angles) == len(times_from_start)

        goal = JointTrajectoryGoal()
        goal.trajectory.joint_names = get_arm_joint_names(arm)
        for positions, time_from_start in zip(angles, times_from_start):
            point = JointTrajectoryPoint()
            point.positions = positions
            # point.velocities = [(ang[i] - last_position[i])/(t - last_time) for i in range(7)]
            # point.velocities = [0.0 for _ in range(len(positions))]
            point.time_from_start = rospy.Duration(time_from_start)
            goal.trajectory.points.append(point)
        # goal.trajectory.header.stamp = rospy.Time.now()
        # Using rospy.Time.now() is bad because the PR2 might be a few seconds ahead.
        # In that case, it clips off the first few points in the trajectory.
        # The clipping causes a jerking motion which can ruin the motion.
        goal.trajectory.header.stamp = rospy.Time(0)
        self.publish_joint_trajectories(goal.trajectory)
        if logging:
            self.start_joint_logging(get_arm_joint_names(arm))
        # TODO(caelan): multiplicative time_buffer
        timeout = times_from_start[-1] + time_buffer
        result = self._send_command("%s_joint" % arm, goal, blocking=blocking, timeout=timeout)
        if logging:
            self.stop_joint_logging()

        actual = np.array([self.joint_positions[joint_name] for joint_name in goal.trajectory.joint_names])
        desired = np.array(goal.trajectory.points[-1].positions)
        print('Error:', zip(goal.trajectory.joint_names, np.round(actual - desired, 5)))
        return result

    def command_arm(self, arm, angles, timeout, **kwargs):
        return self.command_arm_trajectory(arm, [angles], [timeout], **kwargs)

    def stop_arm(self, arm):
        goal = JointTrajectoryGoal()
        goal.trajectory.joint_names = get_arm_joint_names(arm)
        # goal.trajectory.header.stamp = rospy.Time.now()
        goal.trajectory.header.stamp = rospy.Time(0)
        return self._send_command("%s_joint" % arm, goal, blocking=False)

    def stop_base(self):
        self.command_base(0, 0, 0)

    def attach(self, arm, obj, **kwargs):
        # Closing all the way to grasp
        self.close_gripper(arm, **kwargs)
        self.holding[arm] = obj

    def detach(self, arm, obj):
        if arm in self.holding:
            del self.holding[arm]
            # An error showed up when executing test_push: 'l' is not in self.holding. There might be a place that should have
            # used attach but didn't

    ##################################################

    '''
    TODO: Move to ros_perception
    def r_gripper_eventCB(self, data):
        self.gripper_event['r'] = data

    def l_gripper_eventCB(self, data):
        self.gripper_event['l'] = data

    TODO: Move to ros_perception
    def get_gripper_event(self, arm):
        #This may not work until you subscribe to the gripper event 
        if arm in self.gripper_event:

            msg = self.gripper_event[arm]
            event = msg.trigger_conditions_met or msg.acceleration_event
            return event
        else:
            print "No gripper event found... did you launch gripper sensor action?"
            return None

    TODO: Move this to ros_perception
    def command_event_detector(self, arm, trigger, magnitude, blocking,timeout):
        goal = PR2GripperEventDetectorGoal()
        goal.command.trigger_conditions =  trigger
        goal.command.acceleration_trigger_magnitude=magnitude
        client = "%s_gripper_event"% arm
        return self._send_command(client, goal, blocking, timeout)
    '''
