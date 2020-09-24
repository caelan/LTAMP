from itertools import cycle

from control_tools.execution import ArmTrajectory, GripperCommand
from plan_tools.common import Control, Conf, Pose
from plan_tools.samplers.generators import TOOL_POSE, Context, \
    solve_inverse_kinematics, plan_waypoint_motion, get_pairwise_arm_links, set_gripper_position, plan_attachment_motion
from pybullet_tools.pr2_utils import get_disabled_collisions, get_top_presses, get_gripper_joints
from pybullet_tools.utils import multiply, invert, get_joint_limits, \
    get_unit_vector, Pose, BodySaver


def get_press_gen_fn(world, max_attempts=25, collisions=True):
    disabled_collisions = get_disabled_collisions(world.robot)
    disabled_collisions.update(get_pairwise_arm_links(world.robot, world.arms))
    obstacle_bodies = [world.bodies[surface] for surface in world.get_fixed()] if collisions else []
    pre_direction = 0.15 * get_unit_vector([-1, 0, 0])
    collision_buffer = 0.0 # Because of contact with the table

    def gen_fn(arm, button):
        gripper_joint = get_gripper_joints(world.robot, arm)[0]
        closed_width, open_width = get_joint_limits(world.robot, gripper_joint)
        pose = world.initial_poses[button]
        body = world.bodies[button]
        presses = cycle(get_top_presses(body, tool_pose=TOOL_POSE))
        for attempt in range(max_attempts):
            try:
                press = next(presses)
            except StopIteration:
                break
            set_gripper_position(world.robot, arm, closed_width)
            tool_pose = multiply(pose, invert(press))
            grip_conf = solve_inverse_kinematics(world.robot, arm, tool_pose, obstacles=obstacle_bodies,
                                                 collision_buffer=collision_buffer)
            if grip_conf is None:
                continue
            pretool_pose = multiply(tool_pose, Pose(point=pre_direction))
            post_path = plan_waypoint_motion(world.robot, arm, pretool_pose,
                                             obstacles=obstacle_bodies, collision_buffer=collision_buffer,
                                             self_collisions=collisions, disabled_collisions=disabled_collisions)
            if post_path is None:
                continue
            post_conf = Conf(post_path[-1])
            pre_conf = post_conf
            pre_path = post_path[::-1]
            control = Control({
                'action': 'press',
                'objects': [],
                'context': Context( # TODO: robot might be at the wrong conf
                    savers=[BodySaver(world.robot)], # TODO: start with open instead
                    attachments={}),
                'commands': [
                    GripperCommand(arm, closed_width),
                    ArmTrajectory(arm, pre_path),
                    ArmTrajectory(arm, post_path),
                    GripperCommand(arm, open_width),
                ],
            })
            yield (pre_conf, post_conf, control)
            # TODO: continue exploration
    return gen_fn
