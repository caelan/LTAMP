from itertools import combinations

import numpy as np
import pybullet as p
import random

from pddlstream.utils import find_unique, flatten
from pddlstream.algorithms.constraints import PlanConstraints, WILD as X

from perception_tools.common import create_name
from learn_tools.collectors.common import INFEASIBLE, get_contained_beads, read_mass, stabilize, fill_with_beads, \
    InitialRanges, create_table_bodies, randomize_dynamics, sample_norm, sample_bead_parameters, create_beads
from plan_tools.samplers.grasp import hold_item
from plan_tools.common import LEFT_ARM, GRIPPER_LINKS, COFFEE
from learn_tools.learner import Collector, PLANNING_FAILURE, DYNAMICS, SKILL
from plan_tools.planner import Task
from plan_tools.simulated_problems import update_world, CAMERA_OPTICAL_FRAME, KINECT_INTRINSICS, WIDTH, HEIGHT, \
    TABLE_NAME, TABLE_POSE
from plan_tools.samplers.stir import get_stir_feature, get_stir_gen_fn, STIR_PARAMETER_FNS, STIR_FEATURES
from pybullet_tools.pr2_utils import get_pr2_field_of_view, get_view_aabb, pixel_from_ray
from pybullet_tools.utils import ClientSaver, get_distance, point_from_pose, quat_angle_between, quat_from_pose, \
    get_point, get_image, get_link_pose, link_from_name, approximate_as_cylinder, \
    INF, save_image, get_link_subtree, clone_body, image_from_segmented, spaced_colors, get_bodies, \
    tform_point, unit_pose, unit_quat, wait_for_user, get_joint_reaction_force, dump_body, GRAVITY, set_point, clip_pixel


def create_gripper(robot, arm):
    # gripper = load_pybullet(os.path.join(get_data_path(), 'pr2_gripper.urdf'))
    # gripper = load_pybullet(os.path.join(get_models_path(), 'pr2_description/pr2_l_gripper.urdf'), fixed_base=False)
    links = get_link_subtree(robot, link_from_name(robot, GRIPPER_LINKS[arm]))
    #print([get_link_name(robot, link) for link in links])
    gripper = clone_body(robot, links=links, visual=True, collision=True)  # TODO: joint limits
    return gripper

def compute_dispersion(bowl_body, beads_per_color):
    # homogeneous mixture, dispersion
    # Mean absolute difference
    # Distance correlation
    distances = []
    for beads in beads_per_color:
        bowl_beads = get_contained_beads(bowl_body, beads)
        points = list(map(get_point, bowl_beads))
        distances.extend([get_distance(p1, p2, norm=2)
                          for p1, p2 in combinations(points, r=2)])
    return np.mean(distances)

def draw_box(img_array, box, **kwargs):
    from PIL import Image, ImageDraw
    source_img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(source_img)
    draw.rectangle(box, outline="red", **kwargs)
    return np.array(source_img)

def save_segmented(image, beads_per_color):
    if image is None:
        return
    all_beads = set(flatten(beads_per_color))
    non_beads = sorted(set(get_bodies()) - all_beads)
    color_from_body = dict(zip(non_beads, spaced_colors(len(non_beads))))
    segmented_image = image_from_segmented(image, color_from_body=color_from_body)
    save_image('segmented.png', segmented_image)
    # TODO: could produce bounding boxes of detections

def extract_box(img_array, box):
    (x1, y1), (x2, y2) = box
    return img_array[y1:y2+1, x1:x2+1, ...]

def take_image(world, bowl_body, beads_per_color, extension=np.array([0.1, 0.1, 0])):
    # TODO: either detect bounding or extend the bowl
    kinect_pose = get_link_pose(world.robot, link_from_name(world.robot, CAMERA_OPTICAL_FRAME))
    #draw_pose(kinect_pose, length=0.2)
    #set_camera_pose2(kinect_pose)

    distance = 5.
    target_pos = tform_point(kinect_pose, np.array([0, 0, distance]))
    _, vertical_fov = get_pr2_field_of_view(camera_matrix=KINECT_INTRINSICS)
    image = get_image(point_from_pose(kinect_pose), target_pos, width=WIDTH, height=HEIGHT,
                      vertical_fov=vertical_fov, segment=False)

    bowl_aabb = get_view_aabb(bowl_body, kinect_pose)
    bowl_depth = bowl_aabb[0][2]
    sandbox_aabb = [bowl_aabb[0] - extension, bowl_aabb[1] + extension]
    box = [clip_pixel(pixel_from_ray(KINECT_INTRINSICS, np.append(extreme[:2], [bowl_depth]))
                             .astype(int), WIDTH, HEIGHT) for extreme in sandbox_aabb]

    rgb_image = extract_box(image.rgbPixels, box)
    depth_image = extract_box(image.depthPixels, box)
    save_image('rgb.png', rgb_image) # [0, 255]
    save_image('depth.png', depth_image) # [0, 1]
    save_segmented(image.segmentationMaskBuffer, beads_per_color)
    # TODO: could draw detections on
    return rgb_image

def score_image(rgb_image, bead_colors, beads_per_color, max_distance=0.1):
    # TODO: could floodfill to identify bead clusters (and reward more clusters)
    # TODO: ensure the appropriate ratios are visible on top
    # TODO: penalize marbles that have left the bowl
    # TODO: estimate the number of escaped marbles using the size at that distance
    bead_pixels = [[] for _ in bead_colors]
    for r in range(rgb_image.shape[0]): # height
        for c in range(rgb_image.shape[1]): # width
            pixel = rgb_image[r, c]
            assert pixel[3] == 255
            rgb = pixel[:3] / 255.
            best_index, best_distance = None, INF
            for index, color in enumerate(bead_colors):
                distance = np.linalg.norm(rgb - color[:3])
                if distance < best_distance:
                    best_index, best_distance = index, distance
            if best_distance <= max_distance:
                bead_pixels[best_index].append((r, c))
    # TODO: discount beads outside
    all_beads = list(flatten(beads_per_color))
    bead_frequencies = np.array([len(beads) for beads in beads_per_color], dtype=float) / len(all_beads)
    all_pixels = list(flatten(bead_pixels))
    image_frequencies = np.array([len(beads) for beads in bead_pixels], dtype=float) / len(all_pixels)
    print(bead_frequencies, image_frequencies, image_frequencies - bead_frequencies)

    distances = []
    for pixels in bead_pixels:
        distances.extend([get_distance(p1, p2, norm=2)
                          for p1, p2 in combinations(pixels, r=2)])
    dispersion = np.mean(distances) # TODO: translate into meters?
    print(dispersion)
    # TODO: area of a pixel

def collect_stir(world, num_beads=100):
    arm = LEFT_ARM
    # TODO: randomize geometries for stirrer
    spoon_name = create_name('grey_spoon', 1) # green_spoon | grey_spoon | orange_spoon | stirrer
    bowl_name = create_name('whitebowl', 1)
    scale_name = create_name('onyx_scale', 1)
    item_ranges = {
        spoon_name: InitialRanges(
            width_range=(1., 1.),
            height_range=(1., 1.),
            mass_range=(1., 1.),
            pose2d_range=([0.3, 0.5, -np.pi/2], [0.3, 0.5, -np.pi/2]),
            surface=TABLE_NAME,
        ),
        bowl_name: InitialRanges(
            width_range=(0.75, 1.25),
            height_range=(0.75, 1.25),
            mass_range=(1., 1.),
            pose2d_range=([0.5, -0.05, -np.pi], [0.6, 0.05, np.pi]),  # x, y, theta
            surface=scale_name,
        ),
    }
    # TODO: make a sandbox on the table to contain the beads

    ##################################################

    #alpha = 0.75
    alpha = 1
    bead_colors = [
        (1, 0, 0, alpha),
        (0, 0, 1, alpha),
    ]

    beads_fraction = random.uniform(0.75, 1.25)
    print('Beads fraction:', beads_fraction)
    bead_radius = sample_norm(mu=0.006, sigma=0.001, lower=0.004) # 0.007
    print('Bead radius:', bead_radius)

    # TODO: check collisions/feasibility when sampling
    # TODO: grasps on the blue cup seem off for some reason...
    with ClientSaver(world.client):
        #dump_body(world.robot)
        world.perception.add_surface('sandbox', TABLE_POSE)
        world.perception.add_surface(scale_name, TABLE_POSE)
        create_table_bodies(world, item_ranges)
        #gripper = create_gripper(world.robot, LEFT_ARM)
        #set_point(gripper, (1, 0, 1))
        #wait_for_user()
        bowl_body = world.get_body(bowl_name)
        update_world(world, target_body=bowl_body)
        init_holding = hold_item(world, arm, spoon_name)
        if init_holding is None:
            return INFEASIBLE
        parameters_from_name = randomize_dynamics(world) # TODO: parameters_from_name['bead']

        _, (d, h) = approximate_as_cylinder(bowl_body)
        bowl_area = np.pi*(d/2.)**2
        print('Bowl area:', bowl_area)
        bead_area = np.pi*bead_radius**2
        print('Bead area:', bead_area)
        num_beads = int(np.ceil(beads_fraction*bowl_area / bead_area))
        print('Num beads:', num_beads)
        num_per_color = int(num_beads / len(bead_colors))

        # TODO: randomize bead physics
        beads_per_color = [fill_with_beads(world, bowl_name, create_beads(
            num_beads, bead_radius, uniform_color=color, parameters={})) for color in bead_colors]
        world.initial_beads.update({bead: bowl_body for beads in beads_per_color for bead in beads})
        if any(len(beads) != num_per_color for beads in beads_per_color):
            return INFEASIBLE
        #wait_for_user()

    init = [('Contains', bowl_name, COFFEE)]
    goal = [('Mixed', bowl_name)]
    skeleton = [
        ('move-arm', [arm, X, X, X]),
        ('stir', [arm, bowl_name, X, spoon_name, X, X, X, X]),
        #('move-arm', [arm, X, X, X]),
    ]
    constraints = PlanConstraints(skeletons=[skeleton], exact=True)
    task = Task(init=init, goal=goal, arms=[arm],
                init_holding=init_holding, reset_arms=False, constraints=constraints) # Reset arm to clear the scene
    # TODO: constrain the plan skeleton within the task

    feature = get_stir_feature(world, bowl_name, spoon_name)

    ##################################################

    # table_body = world.get_body(TABLE_NAME)
    # dump_body(table_body)
    # joint = 0
    # p.enableJointForceTorqueSensor(table_body, joint, enableSensor=1, physicsClientId=world.client)
    # stabilize(world)
    # reaction_force = get_joint_reaction_force(table_body, joint)
    # print(np.array(reaction_force[:3])/ GRAVITY)

    perception = world.perception
    initial_pose = perception.get_pose(bowl_name)
    bowl_body = perception.get_body(bowl_name)
    scale_body = perception.get_body(scale_name)
    with ClientSaver(world.client):
        initial_distance = compute_dispersion(bowl_body, beads_per_color)
        initial_mass = read_mass(scale_body)
        print(initial_mass)

    def score_fn(plan):
        assert plan is not None
        with ClientSaver(world.client):
            rgb_image = take_image(world, bowl_body, beads_per_color)
            values = score_image(rgb_image, bead_colors, beads_per_color)

        final_pose = perception.get_pose(bowl_name)
        point_distance = get_distance(point_from_pose(initial_pose), point_from_pose(final_pose)) #, norm=2)
        quat_distance = quat_angle_between(quat_from_pose(initial_pose), quat_from_pose(final_pose))
        print('Translation: {:.5f} m | Rotation: {:.5f} rads'.format(point_distance, quat_distance))

        with ClientSaver(world.client):
            all_beads = list(flatten(beads_per_color))
            bowl_beads = get_contained_beads(bowl_body, all_beads)
            fraction_bowl = float(len(bowl_beads)) / len(all_beads) if all_beads else 0
        print('In Bowl: {}'.format(fraction_bowl))

        with ClientSaver(world.client):
            final_dispersion = compute_dispersion(bowl_body, beads_per_color)
        print('Initial Dispersion: {:.3f} | Final Dispersion {:.3f}'.format(
            initial_distance, final_dispersion))

        score = {
            'bowl_translation': point_distance,
            'bowl_rotation': quat_distance,
            'fraction_in_bowl': fraction_bowl,
            'initial_dispersion': initial_distance,
            'final_dispersion': final_dispersion,
            'num_beads': len(all_beads), # Beads per color
            DYNAMICS: parameters_from_name,
        }
        # TODO: include time required for stirring
        # TODO: change in dispersion

        #wait_for_user()
        #_, args = find_unique(lambda a: a[0] == 'stir', plan)
        #control = args[-1]
        return score

    return task, feature, score_fn

def stir_score(feature, parameter, score):
    # TODO: control the angle of the stir
    if score is None:
        return PLANNING_FAILURE
    raise NotImplementedError()

is_valid_stir = lambda w, f, p: True

STIR_COLLECTOR = Collector(collect_stir, get_stir_gen_fn, STIR_PARAMETER_FNS,
                           is_valid_stir, STIR_FEATURES, stir_score)