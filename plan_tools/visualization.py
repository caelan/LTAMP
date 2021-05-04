import numpy as np

from plan_tools.common import TABLE
from plan_tools.samplers.generators import compute_forward_reachability
from pybullet_tools.utils import ClientSaver, get_aabb, add_segments, wait_for_duration, \
    WorldSaver, user_input, add_text, PoseSaver, set_pose, unit_pose, draw_base_limits, \
    draw_circle, get_point, draw_pose, get_pose, wait_for_user

DEBUG = True # TODO: make an argument

MAX_VIS_DISTANCE = 2.5 # 5.
MAX_REG_DISTANCE = 1.5
assert MAX_REG_DISTANCE <= MAX_VIS_DISTANCE

def draw_forward_reachability(world, arms, grasp_type='top', color=(1, 0, 0)):
    for name, body in world.perception.sim_bodies.items():
        if name.startswith(TABLE):
            table = body
            break
    else:
        return False
    if not DEBUG:
        return True
    with ClientSaver(world.perception.client):
        lower, upper = get_aabb(table)
        for arm in arms:
            vertices = [np.append(vertex, upper[2]+1e-3) for vertex in compute_forward_reachability(
                world.perception.pr2, arm=arm, grasp_type=grasp_type)]
            add_segments(vertices, closed=True, color=color)
    return True

def draw_names(world, **kwargs):
    # TODO: adjust the colors?
    handles = []
    if not DEBUG:
        return handles
    with ClientSaver(world.perception.client):
        for name, body in world.perception.sim_bodies.items():
            #add_body_name(body, **kwargs)
            with PoseSaver(body):
                set_pose(body, unit_pose())
                lower, upper = get_aabb(body) # TODO: multi-link bodies doesn't seem to update when moved
                handles.append(add_text(name, position=upper, parent=body, **kwargs)) # parent_link=0,
            #handles.append(draw_pose(get_pose(body)))
        #handles.extend(draw_base_limits(get_base_limits(world.pr2), color=(1, 0, 0)))
        #handles.extend(draw_circle(get_point(world.pr2), MAX_VIS_DISTANCE, color=(0, 0, 1)))
        #handles.extend(draw_circle(get_point(world.pr2), MAX_REG_DISTANCE, color=(0, 0, 1)))
        #from plan_tools.debug import test_voxels
        #test_voxels(world)
    return handles

##################################################

def step_command(world, command, attachments, time_step=None):
    # TODO: end_only
    # More generally downsample
    for i, _ in enumerate(command.iterate(world, attachments)):
        for attachment in attachments.values():
            attachment.assign()
        if i == 0:
            continue
        if time_step is None:
            wait_for_duration(1e-2)
            wait_for_user('Step {}) Next?'.format(i))
        else:
            wait_for_duration(time_step)

def step_plan(world, plan, attachments={}, **kwargs):
    if plan is None:
        return False
    attachments = dict(attachments)
    with ClientSaver(world.perception.client):
        with WorldSaver():
            for name, args in plan:
                if name in ['cook']:
                    continue
                control = args[-1]
                for command in control['commands']:
                    step_command(world, command, attachments, **kwargs)
        wait_for_user('Finished!')
    return True
