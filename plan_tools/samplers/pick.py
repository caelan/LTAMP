from itertools import cycle

from control_tools.execution import ArmTrajectory, Attach, GripperCommand
from plan_tools.common import Control, Conf
from plan_tools.samplers.generators import solve_inverse_kinematics, plan_waypoint_motion, \
    get_pairwise_arm_links, set_gripper_position, Context
from plan_tools.samplers.grasp import get_grasp_gen_fn, get_grasp_attachment
from pybullet_tools.pr2_utils import get_disabled_collisions, get_gripper_joints
from pybullet_tools.utils import set_pose, multiply, invert, \
    get_max_limit, BodySaver

def get_pick_gen_fn(world, max_attempts=25, collisions=True):
    # TODO(caelan): check object/end-effector path collisions
    disabled_collisions = get_disabled_collisions(world.robot)
    disabled_collisions.update(get_pairwise_arm_links(world.robot, world.arms))

    obstacles = [world.bodies[surface] for surface in world.get_fixed()] if collisions else []
    grasp_gen_fn = get_grasp_gen_fn(world)
    #def gen_fn(arm, obj_name, pose, grasp):
    def gen_fn(arm, obj_name, pose):
        open_width = get_max_limit(world.robot, get_gripper_joints(world.robot, arm)[0])
        obj_body = world.bodies[obj_name]

        #grasps = cycle([(grasp,)])
        grasps = cycle(grasp_gen_fn(obj_name))
        for attempt in range(max_attempts):
            try:
                grasp, = next(grasps)
            except StopIteration:
                break
            # TODO: if already successful for a grasp, continue
            set_pose(obj_body, pose)
            set_gripper_position(world.robot, arm, open_width)
            robot_saver = BodySaver(world.robot) # TODO: robot might be at the wrong conf
            body_saver = BodySaver(obj_body)
            pretool_pose = multiply(pose, invert(grasp.pregrasp_pose))
            tool_pose = multiply(pose, invert(grasp.grasp_pose))
            grip_conf = solve_inverse_kinematics(world.robot, arm, tool_pose, obstacles=obstacles)
            if grip_conf is None:
                continue
            #attachment = Attachment(world.robot, tool_link, get_grasp_pose(grasp), world.bodies[obj])
            # Attachments cause table collisions
            post_path = plan_waypoint_motion(world.robot, arm, pretool_pose,
                                             obstacles=obstacles, attachments=[],  #attachments=[attachment],
                                             self_collisions=collisions, disabled_collisions=disabled_collisions)
            if post_path is None:
                continue
            pre_conf = Conf(post_path[-1])
            pre_path = post_path[::-1]
            post_conf = pre_conf
            control = Control({
                'action': 'pick',
                'objects': [obj_name],
                'context': Context(
                    savers=[robot_saver, body_saver],
                    attachments={}),
                'commands': [
                    ArmTrajectory(arm, pre_path, dialation=2.),
                    GripperCommand(arm, grasp.grasp_width, effort=grasp.effort),
                    Attach(arm, obj_name, effort=grasp.effort),
                    ArmTrajectory(arm, post_path, dialation=2.),
                ],
            })
            yield (grasp, pre_conf, post_conf, control)
            # TODO: continue exploration
    return gen_fn
