import pybullet as p

import numpy as np

from perception_tools.common import create_name
from plan_tools.common import ARMS, LEFT_ARM, RIGHT_ARM, TABLE, TOP, COFFEE, SUGAR, COLORS
from plan_tools.planner import Task
from plan_tools.ros_world import ROSWorld
from plan_tools.samplers.grasp import hold_item
from pybullet_tools.pr2_utils import inverse_visibility, set_group_conf, arm_from_arm, \
    WIDE_LEFT_ARM, rightarm_from_leftarm, open_arm
from pybullet_tools.pr2_problems import create_floor
from pybullet_tools.utils import ClientSaver, get_point, unit_quat, unit_pose, link_from_name, draw_point, \
    HideOutput, stable_z, quat_from_euler, Euler, set_camera_pose, z_rotation, multiply, \
    get_quat, create_mesh, set_point, set_quat, set_pose, wait_for_user, \
    get_visual_data, read_obj, BASE_LINK, \
    approximate_as_prism, Pose, draw_mesh, rectangular_mesh, apply_alpha, Point

TABLE_NAME = create_name(TABLE, 1)

# TODO: randomly sample from interval
TABLE_POSE = ([0.55, 0.0, 0.735], unit_quat())

Z_EPSILON = 1e-3

STOVE_POSITION = np.array([0.0, 0.4, 0.0])
PLACEMAT_POSITION = np.array([0.1, 0.0, 0.0])
BUTTON_POSITION = STOVE_POSITION - np.array([0.15, 0.0, 0.0])
TORSO_POSITION = 0.325 # TODO: maintain average distance betweeen pr2 and table

CAMERA_OPTICAL_FRAME = 'head_mount_kinect_rgb_optical_frame'
CAMERA_FRAME = 'head_mount_kinect_rgb_link'
#CAMERA_FRAME = 'high_def_frame'
#CAMERA_OPTICAL_FRAME = 'high_def_optical_frame'

#VIEWER_POINT = np.array([1.25, -0.5, 1.25]) # Angled
VIEWER_POINT = np.array([1.25, 0., 1.25]) # Wide
#VIEWER_POINT = np.array([1., 0., 1.]) # Closeup

WIDTH, HEIGHT = 640, 480
FX, FY, CX, CY = 525., 525., 319.5, 239.5
KINECT_INTRINSICS = np.array([[FX, 0., CX],
                              [0., FY, CY],
                              [0., 0., 1.]])

##################################################

def get_placement_surface(body):
    # TODO: check that only has one link
    visual_data = get_visual_data(body, link=BASE_LINK)
    if (len(visual_data) != 1) or (visual_data[0].visualGeometryType != p.GEOM_MESH):
        center, (w, l, h) = approximate_as_prism(body)
        mesh = rectangular_mesh(w, l)
        local_pose = Pose(center + np.array([0, 0, h/2.]))
        #handles = draw_mesh(mesh)
        #wait_for_user()
        return mesh, local_pose
    mesh = read_obj(visual_data[0].meshAssetFileName)
    local_pose = (visual_data[0].localVisualFrame_position,
                  visual_data[0].localVisualFrame_orientation)
    return mesh, local_pose

def add_table(world, use_surface=False):
    # Use the table in simulation to ensure no penetration
    if use_surface:
        # TODO: note whether the convex hull image was clipped (only partial)
        table_mesh = rectangular_mesh(width=0.6, length=1.2)
        color = apply_alpha(COLORS['tan'])
        table_body = create_mesh(table_mesh, under=True, color=color)
        set_pose(table_body, TABLE_POSE)
        world.perception.sim_bodies[TABLE_NAME] = table_body
        world.perception.sim_surfaces[TABLE_NAME] = None
    else:
        table_body = world.perception.add_surface(TABLE_NAME, TABLE_POSE)
    return table_body

def create_world(items=[], **kwargs):
    state = {name: unit_pose() for name in items}
    with HideOutput():
        world = ROSWorld(sim_only=True, state=state, **kwargs) # state=[]
        table_body = add_table(world)
        create_floor()
    return world, table_body

def update_world(world, target_body, camera_point=VIEWER_POINT):
    pr2 = world.perception.pr2
    with ClientSaver(world.perception.client):
        open_arm(pr2, LEFT_ARM)
        open_arm(pr2, RIGHT_ARM)
        set_group_conf(pr2, 'torso', [TORSO_POSITION])
        set_group_conf(pr2, arm_from_arm('left'), WIDE_LEFT_ARM)
        set_group_conf(pr2, arm_from_arm('right'), rightarm_from_leftarm(WIDE_LEFT_ARM))
        target_point = get_point(target_body)
        head_conf = inverse_visibility(pr2, target_point, head_name=CAMERA_OPTICAL_FRAME) # Must come after torso
        set_group_conf(pr2, 'head', head_conf)
        set_camera_pose(camera_point, target_point=target_point)
        #camera_pose = get_link_pose(world.robot, link_from_name(world.robot, CAMERA_OPTICAL_FRAME))
        #draw_pose(camera_pose)
        #set_camera_pose(point_from_pose(camera_pose), target_point=target_point)
        #add_line(point_from_pose(camera_pose), target_point)
        #attach_viewcone(world.robot, CAMERA_FRAME, depth=2, camera_matrix=KINECT_INTRINSICS)
    #world.perception.set_pose('short_floor', get_point(pr2), unit_quat())

##################################################

def test_block(visualize):
    #arms = [LEFT_ARM]
    arms = ARMS
    block_name = create_name('greenblock', 1)
    tray_name = create_name('tray', 1)

    world, table_body = create_world([tray_name, block_name], visualize=visualize)
    with ClientSaver(world.perception.client):
        block_z = stable_z(world.perception.sim_bodies[block_name], table_body) + Z_EPSILON
        tray_z = stable_z(world.perception.sim_bodies[tray_name], table_body) + Z_EPSILON
        #attach_viewcone(world.perception.pr2, depth=1, head_name='high_def_frame')
        #dump_body(world.perception.pr2)
    #block_y = 0.0
    block_y = -0.4
    world.perception.set_pose(block_name, Point(0.6, block_y, block_z), unit_quat())
    world.perception.set_pose(tray_name, Point(0.6, 0.4, tray_z), unit_quat())
    update_world(world, table_body)

    init = [
        ('Stackable', block_name, tray_name, TOP),
    ]
    goal = [
        ('On', block_name, tray_name, TOP),
    ]
    #goal = [
    #    ('Grasped', arms[0], block_name),
    #]
    task = Task(init=init, goal=goal, arms=arms, empty_arms=True)

    return world, task


def test_cup(visualize):
    arms = [LEFT_ARM]
    cup_name = create_name('greenblock', 1)
    block_name = create_name('purpleblock', 1) # greenblock
    tray_name = create_name('tray', 1)

    world, table_body = create_world([tray_name, cup_name, block_name], visualize=visualize)
    #cup_x = 0.6
    cup_x = 0.8
    block_x = cup_x - 0.15
    initial_poses = {
        cup_name: ([cup_x, 0.0, 0.0], unit_quat()),
        block_name: ([block_x, 0.0, 0.0], unit_quat()), # z_rotation(np.pi/2)
        tray_name: ([0.6, 0.4, 0.0], unit_quat()),
    }
    with ClientSaver(world.perception.client):
        for name, pose in initial_poses.items():
            point, quat = pose
            point[2] += stable_z(world.perception.sim_bodies[name], table_body) + Z_EPSILON
            world.perception.set_pose(name, point, quat)
    update_world(world, table_body)

    #with ClientSaver(world.perception.client):
    #    draw_pose(get_pose(world.perception.sim_bodies['bluecup']))
    #    draw_aabb(get_aabb(world.perception.sim_bodies['bluecup']))
    #    wait_for_interrupt()

    init = [
        #('Contains', cup_name, WATER),
        ('Stackable', cup_name, tray_name, TOP),
    ]
    goal = [
        #('Grasped', goal_arm, block_name),
        #('Grasped', goal_arm, cup_name),
        ('On', cup_name, tray_name, TOP),
        #('On', cup_name, TABLE_NAME, TOP),
        #('Empty', goal_arm),
    ]
    task = Task(init=init, goal=goal, arms=arms)

    return world, task


def test_pour(visualize):
    arms = [LEFT_ARM]
    #cup_name, bowl_name = 'bluecup', 'bluebowl'
    #cup_name, bowl_name = 'cup_7', 'cup_8'
    #cup_name, bowl_name = 'cup_7-1', 'cup_7-2'
    #cup_name, bowl_name = get_name('bluecup', 1), get_name('bluecup', 2)
    cup_name = create_name('bluecup', 1) # bluecup | purple_cup | orange_cup | blue_cup | olive1_cup | blue3D_cup
    bowl_name = create_name('bowl', 1) # bowl | steel_bowl

    world, table_body = create_world([bowl_name, cup_name, bowl_name], visualize=visualize)
    with ClientSaver(world.perception.client):
        cup_z = stable_z(world.perception.sim_bodies[cup_name], table_body) + Z_EPSILON
        bowl_z = stable_z(world.perception.sim_bodies[bowl_name], table_body) + Z_EPSILON
    world.perception.set_pose(cup_name, Point(0.6, 0.2, cup_z), unit_quat())
    world.perception.set_pose(bowl_name, Point(0.6, 0.0, bowl_z), unit_quat())
    update_world(world, table_body)

    init = [
        ('Contains', cup_name, COFFEE),
        #('Graspable', bowl_name),
    ]
    goal = [
        ('Contains', bowl_name, COFFEE),
    ]
    task = Task(init=init, goal=goal, arms=arms,
                empty_arms=True, reset_arms=True, reset_items=True)

    return world, task

def test_shelves(visualize):
    arms = [LEFT_ARM]
    # TODO: smaller (2) shelves
    # TODO: sample with respect to the link it will supported by
    # shelves.off | shelves_points.off | tableShelves.off
    # import os
    # data_directory = '/Users/caelan/Programs/LIS/git/lis-data/meshes/'
    # mesh = read_mesh_off(os.path.join(data_directory, 'tableShelves.off'))
    # draw_mesh(mesh)
    # wait_for_interrupt()
    start_link = 'shelf2' # shelf1 | shelf2 | shelf3 | shelf4
    end_link = 'shelf1'

    shelf_name = 'two_shelves'
    #shelf_name = 'three_shelves'
    cup_name = create_name('bluecup', 1)

    world, table_body = create_world([shelf_name, cup_name], visualize=visualize)
    cup_x = 0.65
    #shelf_x = 0.8
    shelf_x = 0.75

    initial_poses = {
        shelf_name: ([shelf_x, 0.3, 0.0], quat_from_euler(Euler(yaw=-np.pi/2))),
        #cup_name: ([cup_x, 0.0, 0.0], unit_quat()),
    }
    with ClientSaver(world.perception.client):
        for name, pose in initial_poses.items():
            point, quat = pose
            point[2] += stable_z(world.perception.sim_bodies[name], table_body) + Z_EPSILON
            world.perception.set_pose(name, point, quat)
        shelf_body = world.perception.sim_bodies[shelf_name]
        #print([str(get_link_name(shelf_body, link)) for link in get_links(shelf_body)])
        #print([str(get_link_name(shelf_body, link)) for link in get_links(world.perception.sim_bodies[cup_name])])
        #shelf_link = None
        shelf_link = link_from_name(shelf_body, start_link)
        cup_z = stable_z(world.perception.sim_bodies[cup_name], shelf_body, surface_link=shelf_link) + Z_EPSILON
        world.perception.set_pose(cup_name, [cup_x, 0.1, cup_z], unit_quat())
    update_world(world, table_body, camera_point=np.array([-0.5, -1, 1.5]))

    init = [
        ('Stackable', cup_name, shelf_name, end_link),
    ]
    goal = [
        ('On', cup_name, shelf_name, end_link),
        #('On', cup_name, TABLE_NAME, TOP),
        #('Holding', cup_name),
    ]
    task = Task(init=init, goal=goal, arms=arms)

    return world, task

def test_clutter(visualize, num_blocks=5, num_beads=0):
    arms = [LEFT_ARM]
    cup_name = create_name('bluecup', 1)
    bowl_name = create_name('bowl', 1) # bowl | cup_7
    clutter = [create_name('greenblock', i) for i in range(num_blocks)]

    world, table_body = create_world([cup_name, bowl_name] + clutter, visualize=visualize)
    with ClientSaver(world.perception.client):
        cup_z = stable_z(world.perception.sim_bodies[cup_name], table_body) + Z_EPSILON
        bowl_z = stable_z(world.perception.sim_bodies[bowl_name], table_body) + Z_EPSILON
        block_z = None
        if 0 < num_blocks:
            block_z = stable_z(world.perception.sim_bodies[clutter[0]], table_body)

    # TODO(lagrassa): first pose collides with the bowl
    xy_poses = [(0.6, -0.1), (0.5, -0.2), (0.6, 0.11), (0.45, 0.12), (0.5, 0.3), (0.7, 0.3)]
    world.perception.set_pose(cup_name, Point(0.6, 0.2, cup_z), unit_quat())
    world.perception.set_pose(bowl_name, Point(0.6, -0.1, bowl_z), unit_quat())
    #if 0 < num_beads:
    #    world.perception.add_beads(cup_name, num_beads, bead_radius=0.007, bead_mass=0.005)
    if block_z is not None:
        clutter_poses = [np.append(xy, block_z) for xy in reversed(xy_poses)]
        for obj, pose in zip(clutter, clutter_poses):
            world.perception.set_pose(obj, pose, unit_quat())
    update_world(world, table_body)

    #init = [('Graspable', name) for name in [cup_name, bowl_name] + clutter]
    init = [
        ('Contains', cup_name, COFFEE),
    ]
    goal = [
        ('Contains', bowl_name, COFFEE),
    ]
    task = Task(init=init, goal=goal, arms=arms)

    return world, task

def test_push(visualize):
    arms = [LEFT_ARM]
    block_name = create_name('purpleblock', 1) # greenblock | purpleblock

    world, table_body = create_world([block_name], visualize=visualize)
    with ClientSaver(world.perception.client):
        block_z = stable_z(world.perception.sim_bodies[block_name], table_body) + Z_EPSILON
    #pos = (0.6, 0, block_z)
    pos = (0.5, 0.2, block_z)
    world.perception.set_pose(block_name, pos, quat_from_euler(Euler(yaw=-np.pi/6)))
    update_world(world, table_body)

    # TODO: push into reachable region
    goal_pos2d = np.array(pos[:2]) + np.array([0.025, 0.1])
    #goal_pos2d = np.array(pos[:2]) + np.array([0.025, -0.1])
    #goal_pos2d = np.array(pos[:2]) + np.array([-0.1, -0.05])
    #goal_pos2d = np.array(pos[:2]) + np.array([0.15, -0.05])
    #goal_pos2d = np.array(pos[:2]) + np.array([0, 0.7]) # Intentionally not feasible
    draw_point(np.append(goal_pos2d, [block_z]), size=0.05, color=(1, 0, 0))

    init = [
        ('CanPush', block_name, goal_pos2d),
    ]
    goal = [
        ('InRegion', block_name, goal_pos2d),
    ]
    task = Task(init=init, goal=goal, arms=arms)

    return world, task

def test_holding(visualize, num_blocks=2):
    arms = [LEFT_ARM]
    #item_type = 'greenblock'
    #item_type = 'bowl'
    item_type = 'purple_cup'

    item_poses = [
        ([0.6, +0.4, 0.0], z_rotation(np.pi / 2)),
        ([0.6, -0.4, 0.0], unit_quat()),
    ]
    items = [create_name(item_type, i) for i in range(min(len(item_poses), num_blocks))]

    world, table_body = create_world(items, visualize=visualize)
    with ClientSaver(world.perception.client):
        for name, pose in zip(items, item_poses):
            point, quat = pose
            point[2] += stable_z(world.perception.sim_bodies[name], table_body) + Z_EPSILON
            world.perception.set_pose(name, point, quat)
    update_world(world, table_body)

    init = [
        ('Graspable', item) for item in items
    ]
    goal = [
        #('Holding', items[0]),
        ('HoldingType', item_type),
    ]
    task = Task(init=init, goal=goal, arms=arms)

    return world, task

def test_cook(visualize):
    # TODO: can also disable collision bodies for stove & placemat
    arms = [LEFT_ARM]
    broccoli_name = 'greenblock'
    stove_name = 'stove'
    placemat_name = 'placemat'
    button_name = 'button'

    items = [stove_name, button_name, placemat_name, broccoli_name]
    world, table_body = create_world(items,  visualize=visualize)
    update_world(world, table_body)

    initial_poses = {
        broccoli_name: ([-0.1, 0.0, 0.0], z_rotation(np.pi/2)),
        stove_name: (STOVE_POSITION, unit_quat()),
        placemat_name: (PLACEMAT_POSITION, unit_quat()),
        button_name: (BUTTON_POSITION, unit_quat()),
    }
    with ClientSaver(world.perception.client):
        for name, local_pose in initial_poses.items():
            world_pose = multiply(TABLE_POSE, local_pose)
            #z = stable_z(world.perception.sim_bodies[name], table_body) # Slightly above world_pose
            world.perception.set_pose(name, *world_pose)

    init = [
        ('IsButton', button_name, stove_name),
        ('Stackable', broccoli_name, stove_name, TOP),
        ('Stackable', broccoli_name, placemat_name, TOP),
    ]
    goal = [
        ('Cooked', broccoli_name),
        ('On', broccoli_name, placemat_name, TOP),
    ]
    task = Task(init=init, goal=goal, arms=arms)

    return world, task

def test_stacking(visualize):
    # TODO: move & stack
    arms = [LEFT_ARM]
    #arms = ARMS

    cup_name = create_name('bluecup', 1) # bluecup | spoon
    green_name = create_name('greenblock', 1)
    purple_name = create_name('purpleblock', 1)

    items = [cup_name, green_name, purple_name]
    world, table_body = create_world(items,  visualize=visualize)
    cup_x = 0.5
    initial_poses = {
        cup_name: ([cup_x, -0.2, 0.0], unit_quat()),
        #cup_name: ([cup_x, -0.2, 0.3], unit_quat()),
        green_name: ([cup_x, 0.1, 0.0], unit_quat()),
        #green_name: ([cup_x, 0.1, 0.0], z_rotation(np.pi / 4)), # TODO: be careful about the orientation when stacking
        purple_name: ([cup_x, 0.4, 0.0], unit_quat()),
    }
    with ClientSaver(world.perception.client):
        for name, pose in initial_poses.items():
            point, quat = pose
            point[2] += stable_z(world.perception.sim_bodies[name], table_body) + Z_EPSILON
            world.perception.set_pose(name, point, quat)
    update_world(world, table_body)


    init = [
        #('Stackable', cup_name, purple_name, TOP),
        ('Stackable', cup_name, green_name, TOP),
        ('Stackable', green_name, purple_name, TOP),
    ]
    goal = [
        #('On', cup_name, purple_name, TOP),
        ('On', cup_name, green_name, TOP),
        ('On', green_name, purple_name, TOP),
    ]
    task = Task(init=init, goal=goal, arms=arms)

    return world, task

def test_stack_pour(visualize):
    arms = [LEFT_ARM]
    bowl_name = create_name('whitebowl', 1)
    cup_name = create_name('bluecup', 1)
    purple_name = create_name('purpleblock', 1)

    items = [bowl_name, cup_name, purple_name]
    world, table_body = create_world(items,  visualize=visualize)
    cup_x = 0.5
    initial_poses = {
        bowl_name: ([cup_x, -0.2, 0.0], unit_quat()),
        cup_name: ([cup_x, 0.1, 0.0], unit_quat()),
        purple_name: ([cup_x, 0.4, 0.0], unit_quat()),
    }
    with ClientSaver(world.perception.client):
        for name, pose in initial_poses.items():
            point, quat = pose
            point[2] += stable_z(world.perception.sim_bodies[name], table_body) + Z_EPSILON
            world.perception.set_pose(name, point, quat)
    update_world(world, table_body)

    init = [
        ('Contains', cup_name, COFFEE),
        ('Stackable', bowl_name, purple_name, TOP),
    ]
    goal = [
        ('Contains', bowl_name, COFFEE),
        ('On', bowl_name, purple_name, TOP),
    ]
    task = Task(init=init, goal=goal, arms=arms, graspable=['whitebowl'])
    return world, task

def test_stir(visualize):
    # TODO: change the stirrer coordinate frame to be on its side
    arms = [LEFT_ARM]
    #spoon_name = 'spoon' # spoon.urdf is very messy and has a bad bounding box
    #spoon_quat = multiply_quats(quat_from_euler(Euler(pitch=-np.pi/2 - np.pi/16)), quat_from_euler(Euler(yaw=-np.pi/8)))
    #spoon_name, spoon_quat = 'stirrer', quat_from_euler(Euler(roll=-np.pi/2, yaw=np.pi/2))
    spoon_name, spoon_quat = 'grey_spoon', quat_from_euler(Euler(yaw=-np.pi/2)) # grey_spoon | orange_spoon | green_spoon
    # *_spoon points in +y
    #bowl_name = 'bowl'
    bowl_name = 'whitebowl'

    items = [spoon_name, bowl_name]
    world, table_body = create_world(items, visualize=visualize)
    set_quat(world.get_body(spoon_name), spoon_quat) # get_reference_pose(spoon_name)

    initial_positions = {
        #spoon_name: [0.5, -0.2, 0],
        spoon_name: [0.3, 0.5, 0],
        bowl_name: [0.6, 0.1, 0],
    }
    with ClientSaver(world.perception.client):
        for name, point in initial_positions.items():
            body = world.perception.sim_bodies[name]
            point[2] += stable_z(body, table_body) + Z_EPSILON
            world.perception.set_pose(name, point, get_quat(body))
            #draw_aabb(get_aabb(body))
        #wait_for_interrupt()
        #enable_gravity()
        #for i in range(10):
        #    simulate_for_sim_duration(sim_duration=0.05, frequency=0.1)
        #    print(get_quat(world.perception.sim_bodies[spoon_name]))
        #    raw_input('Continue?')
        update_world(world, table_body)
        init_holding = hold_item(world, arms[0], spoon_name)
        assert init_holding is not None

    # TODO: add the concept of a recipe
    init = [
        ('Contains', bowl_name, COFFEE),
        ('Contains', bowl_name, SUGAR),
    ]
    goal = [
        #('Holding', spoon_name),
        ('Mixed', bowl_name),
    ]
    task = Task(init=init, init_holding=init_holding, goal=goal,
                arms=arms, reset_arms=False)

    return world, task

def test_scoop(visualize):
    # TODO: start with the spoon in a hand
    arms = [LEFT_ARM]
    spoon_name = 'grey_spoon' # green_spoon | grey_spoon | orange_spoon
    spoon_quat = quat_from_euler(Euler(yaw=-np.pi/2))
    # *_spoon points in +y
    bowl1_name = 'whitebowl'
    bowl2_name = 'bowl'

    items = [spoon_name, bowl1_name, bowl2_name]
    world, table_body = create_world(items, visualize=visualize)
    set_quat(world.get_body(spoon_name), spoon_quat)

    initial_positions = {
        spoon_name: [0.3, 0.5, 0],
        bowl1_name: [0.5, 0.1, 0],
        bowl2_name: [0.6, 0.5, 0],
    }
    with ClientSaver(world.perception.client):
        for name, point in initial_positions.items():
            body = world.perception.sim_bodies[name]
            point[2] += stable_z(body, table_body) + Z_EPSILON
            world.perception.set_pose(name, point, get_quat(body))
    update_world(world, table_body)

    init = [
        ('Contains', bowl1_name, COFFEE),
        #('Contains', spoon_name, WATER),
    ]
    goal = [
        #('Contains', spoon_name, WATER),
        ('Contains', bowl2_name, COFFEE),
    ]
    task = Task(init=init, goal=goal, arms=arms, reset_arms=True)
    # TODO: plan while not spilling the spoon

    return world, task

def test_coffee(visualize):
    arms = [LEFT_ARM, RIGHT_ARM]
    spoon_name = create_name('orange_spoon', 1) # grey_spoon | orange_spoon | green_spoon
    coffee_name = create_name('bluecup', 1)
    sugar_name = create_name('bowl', 1) # bowl | tan_bowl
    bowl_name = create_name('whitebowl', 1)

    items = [spoon_name, coffee_name, sugar_name, bowl_name]
    world, table_body = create_world(items, visualize=visualize)

    initial_positions = {
        coffee_name: [0.5, 0.3, 0],
        sugar_name: [0.5, -0.3, 0],
        bowl_name: [0.5, 0.0, 0],
    }
    with ClientSaver(world.perception.client):
        for name, point in initial_positions.items():
            body = world.perception.sim_bodies[name]
            point[2] += stable_z(body, table_body) + Z_EPSILON
            world.perception.set_pose(name, point, get_quat(body))
        update_world(world, table_body)
        init_holding = hold_item(world, RIGHT_ARM, spoon_name)
        assert init_holding is not None
        [grasp] = list(init_holding.values())
        #print(grasp.grasp_pose)
        #print(grasp.pre_direction)
        #print(grasp.grasp_width)

    init = [
        ('Contains', coffee_name, COFFEE),
        ('Contains', sugar_name, SUGAR),
    ]
    goal = [
        #('Contains', spoon_name, SUGAR),
        #('Contains', bowl_name, COFFEE),
        #('Contains', bowl_name, SUGAR),
        ('Mixed', bowl_name),
    ]
    task = Task(init=init, init_holding=init_holding, goal=goal, arms=arms,
                reset_arms=True, empty_arms=[LEFT_ARM])

    return world, task

def test_push_pour(visualize):
    arms = ARMS
    cup_name = create_name('bluecup', 1)
    bowl_name = create_name('bowl', 1)

    items = [bowl_name, cup_name, bowl_name]
    world, table_body = create_world(items, visualize=visualize)
    set_point(table_body, np.array(TABLE_POSE[0]) + np.array([0, -0.1, 0]))

    with ClientSaver(world.perception.client):
        cup_z = stable_z(world.perception.sim_bodies[cup_name], table_body) + Z_EPSILON
        bowl_z = stable_z(world.perception.sim_bodies[bowl_name], table_body) + Z_EPSILON
    world.perception.set_pose(cup_name, Point(0.75, 0.4, cup_z), unit_quat())
    world.perception.set_pose(bowl_name, Point(0.5, -0.6, bowl_z), unit_quat())
    update_world(world, table_body)

    # TODO: can prevent the right arm from being able to pick
    init = [
        ('Contains', cup_name, COFFEE),
        #('Graspable', bowl_name),
        ('CanPush', bowl_name, LEFT_ARM), # TODO: intersection of these regions
    ]
    goal = [
        ('Contains', bowl_name, COFFEE),
        ('InRegion', bowl_name, LEFT_ARM),
    ]
    task = Task(init=init, goal=goal, arms=arms, pushable=[bowl_name])
    # Most of the time places the cup to exchange arms

    return world, task

##################################################

STABLE_PROBLEMS = [
    test_block,
    test_pour,
    test_shelves,  # TODO(caelan): currently broken due to collision_buffer (maybe not?)
    test_push,
    test_holding,
    test_stacking,
    test_stack_pour,
    test_stir,
    test_coffee,
]

BROKEN_PROBLEMS = [
    # TODO(caelan): debug legacy formulation
    test_cup,
    test_clutter,  # TODO(caelan): currently broken due to collision_buffer
    test_cook,  # TODO(caelan): RuntimeError: Preimage fact ('cooked', greenblock) is not achievable!
    test_scoop,
    test_push_pour,
]

PROBLEMS = STABLE_PROBLEMS # + BROKEN_PROBLEMS
