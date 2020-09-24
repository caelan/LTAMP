from itertools import cycle

from control_tools.execution import ArmTrajectory, Detach, GripperCommand
from plan_tools.common import Control, Conf, Pose, get_reference_pose, is_obj_type
from plan_tools.samplers.generators import solve_inverse_kinematics, plan_waypoint_motion, \
    compute_forward_reachability, check_initial_collisions, get_pairwise_arm_links, set_gripper_position, Context
from plan_tools.samplers.grasp import get_grasp_attachment
from plan_tools.samplers.collision import body_pair_collision
from pybullet_tools.pr2_utils import get_disabled_collisions, get_gripper_joints
from pybullet_tools.utils import link_from_name, set_pose, multiply, invert, sample_placement, \
    point_from_pose, is_point_in_polygon, PoseSaver, get_max_limit, BodySaver


def get_reachable_test(world, grasp_type='top'):
    # TODO: convex hull to grow the reachable region
    # TODO: visualize the side grasp region as well
    vertices2d_from_arm = {arm: compute_forward_reachability(
        world.robot, arm=arm, grasp_type=grasp_type) for arm in world.arms}
    def test(arm, obj, pose):
        point2d = point_from_pose(pose)[:2]
        return is_point_in_polygon(point2d, vertices2d_from_arm[arm])
    return test

def get_stable_pose_gen_fn(world, max_failures=100, collisions=True):
    obstacle_bodies = [world.bodies[fixed] for fixed in world.get_fixed()] if collisions else []
    #table_body = world.bodies[world.get_table()]
    collision_buffer = 0.0 # Needed if placing on surface near the table w/ collision threshold

    # TODO: stable context to bias predictions
    # TODO: only sample for objects that have collided before
    reachable_test = get_reachable_test(world)
    shrink_extents = 0.5
    def gen_fn(obj_name, surface, surface_pose, link_name):
        if is_obj_type(obj_name, 'spoon'):
            return
        # TODO: make a drop action?
        obj_body = world.bodies[obj_name]
        surface_reference_pose = get_reference_pose(surface) # unit_pose()
        #surface_reference_pose = pose2
        surface_body = world.bodies[surface]
        collision_bodies = set(obstacle_bodies) - {obj_body, surface_body} #, table_body}
        link = None if link_name is None else link_from_name(surface_body, link_name)
        attempt = 0
        last_success = 0
        while (attempt - last_success) < max_failures:
            attempt += 1
            # TODO: use the halton sequence here instead
            # TODO: sample within the convex hull of the surface
            with PoseSaver(surface_body):
                set_pose(surface_body, surface_reference_pose)
                reference_from_obj = sample_placement(obj_body, surface_body,
                                                      top_pose=get_reference_pose(obj_name),
                                                      bottom_link=link, percent=shrink_extents)
            if reference_from_obj is None:
                break
            obj_pose = multiply(surface_pose, invert(surface_reference_pose), reference_from_obj)
            if not any(reachable_test(arm, obj_name, obj_pose) for arm in world.arms):
                continue
            set_pose(obj_body, obj_pose)
            if any(body_pair_collision(obj_body, obst_body, collision_buffer=collision_buffer) for obst_body in collision_bodies):
                continue
            if check_initial_collisions(world, obj_name, set(world.initial_poses) - {obj_name, surface}, collision_buffer=collision_buffer):
                continue
            yield (Pose(obj_pose),)
            last_success = attempt
    return gen_fn

##################################################

def get_place_fn(world, max_attempts=10, collisions=True):
    # TODO(caelan): check object/end-effector path collisions
    # TODO(caelan): make a corresponding pick for each place
    # TODO(caelan): check collisions with placed object
    disabled_collisions = get_disabled_collisions(world.robot)
    disabled_collisions.update(get_pairwise_arm_links(world.robot, world.arms))

    obstacles = [world.bodies[surface] for surface in world.get_fixed()] if collisions else []
    #pose_gen_fn = get_stable_pose_gen_fn(world, collisions=collisions)
    def gen_fn(arm, obj_name, target_pose, grasp):
        gripper_joint = get_gripper_joints(world.robot, arm)[0]
        open_width = get_max_limit(world.robot, gripper_joint)
        attachment = get_grasp_attachment(world, arm, grasp)
        obj_body = world.bodies[obj_name]

        poses = cycle([(target_pose,)])
        #poses = cycle(pose_gen_fn(obj, surface))
        for attempt in range(max_attempts):
            try:
                pose, = next(poses)
            except StopIteration:
                break
            body_saver = BodySaver(obj_body)
            set_pose(obj_body, pose)
            set_gripper_position(world.robot, arm, open_width)
            tool_pose = multiply(pose, invert(grasp.grasp_pose))
            pretool_pose = multiply(pose, invert(grasp.pregrasp_pose))
            grip_conf = solve_inverse_kinematics(world.robot, arm, tool_pose, obstacles=obstacles)
            if grip_conf is None:
                continue
            post_path = plan_waypoint_motion(world.robot, arm, pretool_pose,
                                             obstacles=obstacles, attachments=[],  # attachments=[attachment],
                                             self_collisions=collisions, disabled_collisions=disabled_collisions)
            if post_path is None:
                continue
            set_gripper_position(world.robot, arm, grasp.grasp_width)
            robot_saver = BodySaver(world.robot) # TODO: robot might be at the wrong conf

            post_conf = Conf(post_path[-1])
            pre_conf = post_conf
            pre_path = post_path[::-1]
            control = Control({
                'action': 'place',
                'objects': [obj_name],
                'context': Context(
                    savers=[robot_saver, body_saver],
                    attachments={obj_name: attachment}),
                'commands': [
                    ArmTrajectory(arm, pre_path, dialation=2.),
                    Detach(arm, obj_name),
                    GripperCommand(arm, open_width),
                    ArmTrajectory(arm, post_path, dialation=2.),
                ],
            })
            return (pre_conf, post_conf, control)
            # TODO: continue exploration
        return None
    return gen_fn
