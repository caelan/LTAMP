from __future__ import print_function

# NOTE(caelan): must come before other imports
import sys

sys.path.extend([
    'pddlstream',
    'ss-pybullet',
    #'pddlstream/examples',
    #'pddlstream/examples/pybullet/utils',
])

import argparse
import numpy as np
import datetime
import os
np.set_printoptions(precision=3, threshold=3, edgeitems=1, suppress=True) #, linewidth=1000)

from plan_tools.visualization import draw_forward_reachability, step_plan, draw_names
from plan_tools.planner import plan_actions, PlanningWorld
from plan_tools.common import set_seed, VIDEOS_DIRECTORY
from plan_tools.simulated_problems import PROBLEMS, test_pour
from control_tools.execution import execute_plan
from pybullet_tools.utils import wait_for_user, has_gui, VideoSaver
from learn_tools.common import DATE_FORMAT


def main():
    # TODO: link_from_name is slow (see python -m plan_tools.run_simulation -p test_cook)
    problem_fn_from_name = {fn.__name__: fn for fn in PROBLEMS}

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfree', action='store_true',
                        help='When enabled, disables collision checking (for debugging).')
    parser.add_argument('-e', '--execute', action='store_true',
                        help='When enabled, executes the plan using physics simulation.')
    #parser.add_argument('-l', '--learning', action='store_true',
    #                    help='When enabled, uses learned generators when applicable.')
    parser.add_argument('-p', '--problem', default=test_pour.__name__,
                        choices=sorted(problem_fn_from_name),
                        help='The name of the problem to solve.')
    parser.add_argument('-s', '--seed', default=None,
                        help='The random seed to use.')
    parser.add_argument('-v', '--visualize_planning', action='store_true',
                        help='When enabled, visualizes planning rather than the world (for debugging).')
    parser.add_argument('-d', '--disable_drawing', action='store_true',
                        help='When enabled, disables drawing names and forward reachability.')
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
    print('Problem:', args.problem)
    problem_fn = problem_fn_from_name[args.problem]
    sim_world, task = problem_fn(visualize=not args.visualize_planning)
    #sim_world, task = problem_fn(visualize=False)
    if not args.disable_drawing:
        draw_names(sim_world)
        draw_forward_reachability(sim_world, task.arms)

    planning_world = PlanningWorld(task, visualize=args.visualize_planning)
    planning_world.load(sim_world)
    print(planning_world)
    plan = plan_actions(planning_world, collisions=not args.cfree)
    planning_world.stop()
    if (plan is None) or not has_gui(sim_world.client):
        #print('Failed to find a plan')
        wait_for_user('Finished!')
        sim_world.stop()
        return

    # TODO: see-through cup underneath the table
    wait_for_user('Execute?')
    video_path = os.path.join(VIDEOS_DIRECTORY, '{}_{}.mp4'.format(
        args.problem, datetime.datetime.now().strftime(DATE_FORMAT)))
    video_path = None
    with VideoSaver(video_path): # requires ffmpeg
        if args.execute:
            # Only used by pb_controller, for ros_controller it's assumed always True
            sim_world.controller.execute_motor_control = True
            sim_world.controller.set_gravity()
            execute_plan(sim_world, plan, default_sleep=0.)
        else:
            step_plan(sim_world, plan, task.get_attachments(sim_world),
                      #time_step=None)
                      time_step=0.04)
            #step_plan(world, plan, end_only=True, time_step=None)
    if args.execute:
        wait_for_user('Finished!')
    sim_world.stop()

# http://www.science.smith.edu/dftwiki/index.php/Color_Charts_for_TKinter
# https://www.tcl.tk/man/tcl8.4/TkCmd/colors.htm
# https://docs.python.org/2/library/tkinter.html

# python -m plan_tools.run_simulation -p test_pour

if __name__ == '__main__':
    main()
