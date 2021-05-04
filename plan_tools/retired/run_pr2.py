from __future__ import print_function

import sys

sys.path.extend([
    'pddlstream',
    'ss-pybullet',
    #'pddlstream/examples',
    #'pddlstream/examples/pybullet/utils',
])

import argparse
import numpy as np
#np.set_printoptions(precision=3, threshold=3, edgeitems=1, suppress=True) #, linewidth=1000)

from control_tools.execution import execute_plan, get_arm_prefix
from perception_tools.common import get_body_urdf, get_type
from plan_tools.retired.pr2_problems import HEAD_CONF, LEFT_ARM_CONF, STOVE_NAME, PLACEMAT_NAME, BUTTON_NAME, PROBLEMS
from plan_tools.simulated_problems import STOVE_POSITION, BUTTON_POSITION, PLACEMAT_POSITION, TORSO_POSITION
from plan_tools.ros_world import ROSWorld
from plan_tools.common import TABLE
from plan_tools.planner import plan_actions, PlanningWorld
from plan_tools.samplers.grasp import get_grasp_attachment
from plan_tools.visualization import draw_forward_reachability, step_plan, draw_names

from pddlstream.utils import get_python_version
from pybullet_tools.pr2_utils import rightarm_from_leftarm
from pybullet_tools.utils import ClientSaver, set_camera_pose, multiply, unit_quat, user_input, load_pybullet

def move_to_initial_config(ros_world, open_grippers={}, timeout=5.0, blocking=True):
    for arm, is_open in open_grippers.items():
        if is_open:
            ros_world.controller.open_gripper(get_arm_prefix(arm), blocking=False)
        else:
            ros_world.controller.close_gripper(get_arm_prefix(arm), blocking=False)
    ros_world.controller.command_torso(TORSO_POSITION, timeout=timeout, blocking=False)
    ros_world.controller.command_head(angles=HEAD_CONF, timeout=timeout, blocking=False)
    ros_world.controller.command_arm('l', LEFT_ARM_CONF, timeout=timeout, blocking=False)
    ros_world.controller.command_arm('r', rightarm_from_leftarm(LEFT_ARM_CONF), timeout=timeout, blocking=blocking)

def reset_robot(task, ros_world, **kwargs):
    with ClientSaver(ros_world.perception.client):
        set_camera_pose(np.array([1.5, -0.5, 1.5]), target_point=np.array([0.75, 0, 0.75]))
    open_grippers = {arm: arm in task.init_holding for arm in task.arms}
    move_to_initial_config(ros_world, open_grippers, **kwargs)

def get_input(message, options):
    full_message = '{} [{}]: '.format(message, ','.join(options))
    response = user_input(full_message)
    while response not in options:
        response = user_input(full_message)
    return response

def add_table_surfaces(world):
    table_pose = None
    for body_info in world.perception.surfaces:
        if body_info.type == TABLE:
            table_pose = body_info.pose
    assert table_pose is not None
    initial_poses = {
        STOVE_NAME: (STOVE_POSITION, unit_quat()),
        PLACEMAT_NAME: (PLACEMAT_POSITION, unit_quat()),
        BUTTON_NAME: (BUTTON_POSITION, unit_quat()),
    }
    with ClientSaver(world.perception.client):
        for name, local_pose in initial_poses.items():
            world_pose = multiply(table_pose, local_pose)
            from perception_tools.retired.ros_perception import BodyInfo
            world.perception.surfaces.append(BodyInfo(name, None, world_pose, name))

##################################################

# Right wrist calibration
# https://projects.csail.mit.edu/pr2/wiki/index.php?title=Troubleshooting_PR2
# TODO: trouble detecting bowl when filled with chickpeas

def review_plan(task, ros_world, plan):
    while True:
        step_plan(ros_world, plan, task.get_attachments(ros_world), time_step=0.04)
        response = get_input('Execute plan?', ('y', 'n', 'r'))
        if response == 'y':
            return True
        if response == 'n':
            return False

def add_holding(task, ros_world):
    with ClientSaver(ros_world.client):
        for arm, grasp in task.init_holding.items():
            name = grasp.obj_name
            body = load_pybullet(get_body_urdf(get_type(name)), fixed_base=False)
            ros_world.perception.sim_bodies[name] = body
            ros_world.perception.sim_items[name] = None
            attachment = get_grasp_attachment(ros_world, arm, grasp)
            attachment.assign()
            ros_world.controller.attach(get_arm_prefix(arm), name)
        #wait_for_user()


# TODO: some of the objects a little higher up than others

def get_task_fn(problems, name):
    problem_fn_from_name = {fn.__name__: fn for fn in problems}
    if name not in problem_fn_from_name:
        raise ValueError(name)
    print('Problem:', name)
    return problem_fn_from_name[name]

##################################################

def main():
    assert(get_python_version() == 2) # ariadne has ROS with python2
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem', default='test_pick',
                        help='The name of the problem to solve.')
    parser.add_argument('-e', '--execute', action='store_true',
                        help='When enabled, executes the plan using physics simulation.')
    parser.add_argument('-c', '--cfree', action='store_true',
                        help='When enabled, disables collision checking (for debugging).')
    parser.add_argument('-l', '--learning', action='store_true',
                        help='When enabled, uses learned generators when applicable.')
    parser.add_argument('-v', '--visualize_planning', action='store_true',
                        help='When enabled, visualizes planning rather than the world (for debugging).')
    args = parser.parse_args()

    task_fn = get_task_fn(PROBLEMS, args.problem)
    task = task_fn()
    print('Task:', task.arms)

    ros_world = ROSWorld(sim_only=False, visualize=not args.visualize_planning)
    reset_robot(task, ros_world)
    ros_world.perception.wait_for_update()
    if task.use_kitchen:
        add_table_surfaces(ros_world)
    ros_world.sync_controllers() # AKA update sim robot joint positions
    ros_world.perception.update_simulation()
    add_holding(task, ros_world)

    print('Surfaces:', ros_world.perception.get_surfaces())
    print('Items:', ros_world.perception.get_items())
    draw_names(ros_world)
    draw_forward_reachability(ros_world, task.arms)

    planning_world = PlanningWorld(task, visualize=args.visualize_planning)
    planning_world.load(ros_world)
    print(planning_world)
    plan = plan_actions(planning_world, collisions=not args.cfree)
    if plan is None:
        print('Failed to find a plan')
        user_input('Finish?')
        return

    if not args.execute and not review_plan(task, ros_world, plan):
        return
    if args.cfree:
        print('Aborting execution')
        return

    #sparsify_plan(plan)
    simulator = ros_world.alternate_controller if args.execute else None
    if simulator is not None:
        simulator.set_gravity()
    finished = execute_plan(ros_world, plan, joint_speed=0.25)
    print('Finished:', finished)


if __name__ == '__main__':
    main()
