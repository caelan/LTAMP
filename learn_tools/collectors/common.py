import pybullet as p
import random
import time

import numpy as np

from dimensions.common import get_properties
from plan_tools.common import randomize_body, get_type, get_liquid_quat, load_body
from plan_tools.samplers.grasp import SPOON_DIAMETERS, SPOON_LENGTHS
from plan_tools.simulated_problems import TABLE_NAME, Z_EPSILON, add_table
from pybullet_tools.utils import get_aabb, create_sphere, BASE_LINK, get_client, set_point, elapsed_time, AABB, \
    aabb_contains_aabb, get_bodies_in_region, get_mass, ClientSaver, BodySaver, LockRenderer, HideOutput, add_data_path, \
    load_pybullet, set_pose, get_pose, stable_z, Pose, Euler, INF, apply_alpha, safe_zip, \
    approximate_as_cylinder, get_point, unit_from_theta, remove_body, \
    Point, wait_for_user, set_mass, halton_generator, STATIC_MASS, \
    set_quat, get_aabb_extent, draw_aabb, approximate_as_prism, set_dynamics
from learn_tools.learner import SKILL
from pddlstream.language.statistics import safe_ratio

INFEASIBLE = None, None, None

BEADS_REST = 0.5
MAX_BEADS = 250

#MAX_SPILLED = 0.05 # %
MAX_SPILLED = 0.1 # %
MAX_TRANSLATION = 0.05 # m

#MAX_SPILLED = np.inf # %
#MAX_TRANSLATION = np.inf # m

# MAX_SPILLED is more constraining than MAX_TRANSLATION

##################################################

def sample_norm(mu, sigma, lower=-INF, upper=INF):
    # scipy.stats.truncnorm
    assert lower <= upper
    if lower == upper:
        return lower
    if sigma == 0:
        assert lower <= mu <= upper
        return mu
    while True:
        x = random.gauss(mu=mu, sigma=sigma)
        if lower <= x <= upper:
            return x

def get_contained_beads(body, beads, height_fraction=1.0, top_threshold=0.0):
    #aabb = get_aabb(body)
    center, extent = approximate_as_prism(body, body_pose=get_pose(body))
    lower = center - extent/2.
    upper = center + extent/2.
    _, _, z1 = lower
    x2, y2, z2 = upper
    z_upper = z1 + height_fraction * (z2 - z1) + top_threshold
    new_aabb = AABB(lower, np.array([x2, y2, z_upper]))
    #handles = draw_aabb(new_aabb)
    return {bead for bead in beads if aabb_contains_aabb(get_aabb(bead), new_aabb)}

##################################################

def sample_bead_parameters(randomize=True):
    # p.changeDynamics(self.target, -1, restitution=0.93, linearDamping=1e-5, angularDamping=1e-5, lateralFriction=0.4,
    #                 physicsClientId=self.client)
    # p.changeDynamics(self.source, -1, mass=1, lateralFriction=0.01, spinningFriction=0.01,
    #                 rollingFriction=0.01, physicsClientId=self.client)
    # p.changeDynamics(cup, BASE_LINK, mass=10, lateralFriction=0.99, spinningFriction=0.99,
    #                     rollingFriction=0.99)
    # p.changeDynamics(body, BASE_LINK, mass=mass, restitution=0.93, lateralFriction=0.5,
    #                 spinningFriction=0.99, rollingFriction=0.99)

    # <rolling_friction value="0.001"/>
    # <spinning_friction value="0.001"/>

    # https://github.com/RobotLocomotion/drake/search?q=sphere&type=
    # https://en.wikipedia.org/wiki/Coefficient_of_restitution
    # http://wiki.ros.org/urdf/Tutorials/Adding%20Physical%20and%20Collision%20Properties%20to%20a%20URDF%20Model#Contact_Coefficients

    # Default parameters
    # DynamicsInfo(mass=0.0005, lateral_friction=0.5,
    #             local_inertia_diagonal=(5.000000000000001e-09, 5.000000000000001e-09, 5.000000000000001e-09),
    #             local_inertial_pos=(0.0, 0.0, 0.0), local_inertial_orn=(0.0, 0.0, 0.0, 1.0), restitution=0.0,
    #             rolling_friction=0.0, spinning_friction=0.0, contact_damping=-1.0, contact_stiffness=-1.0)
    # mass, lateralFriction, spinningFriction, rollingFriction, restitution, linearDamping, angularDamping, \
    # contactStiffness, contactDamping, frictionAnchor, localInertiaDiagnoal

    # Could also randomize these parameters individually
    scale = int(randomize)
    parameters = {
        'mass': sample_norm(mu=0.001, sigma=0.0002*scale, lower=0.0005),  # 0.0002 | 0.005 # kg
        # lateral (linear) contact friction
        'lateralFriction': sample_norm(mu=0.5, sigma=0.1*scale, lower=0.1),
        # torsional friction around the contact normal
        'spinningFriction': sample_norm(mu=0.001, sigma=0.0005*scale, lower=0.0),
        # torsional friction orthogonal to contact normal
        'rollingFriction': sample_norm(mu=0.001, sigma=0.0005*scale, lower=0.0),
        # bouncyness of contact. Keep it a bit less than 1.
        # restitution: 0 => inelastic collision, 1 => elastic collision
        'restitution': sample_norm(mu=0.8, sigma=0.1*scale, lower=0.0, upper=1.0),
        # linear damping of the link (0.04 by default)
        'linearDamping': sample_norm(mu=0.04, sigma=0.01*scale, lower=0.01),
        # angular damping of the link (0.04 by default)
        'angularDamping': sample_norm(mu=0.04, sigma=0.01*scale, lower=0.01),
        # 'contactStiffness': None,
        # 'contactDamping': None,
        'localInertiaDiagonal': (sample_norm(mu=5e-9, sigma=1e-9*scale, lower=1e-9) * np.ones(3)).tolist(),
    }
    return parameters

def create_beads(num, radius, parameters={}, uniform_color=None, init_x=10.0):
    beads = []
    if num <= 0:
        return beads
    #timestep = 1/1200. # 1/350.
    #p.setTimeStep(timestep, physicsClientId=get_client())
    for i in range(num):
        color = apply_alpha(np.random.random(3)) if uniform_color is None else uniform_color
        body = create_sphere(radius, color=color) # TODO: other shapes
        #set_color(droplet, color)
        #print(get_dynamics_info(droplet))
        set_dynamics(body, **parameters)
        x = init_x + i * (2 * radius + 1e-2)
        set_point(body, Point(x=x))
        beads.append(body)
    return beads

def pour_beads(world, cup_name, beads, reset_contained=False, fix_outside=True,
               cup_thickness=0.01, bead_offset=0.01, drop_rate=0.02, **kwargs):
    if not beads:
        return set()
    start_time = time.time()
    # TODO: compute the radius of each bead
    bead_radius = np.average(approximate_as_prism(beads[0])) / 2.

    masses = list(map(get_mass, beads))
    savers = list(map(BodySaver, beads))
    for body in beads:
        set_mass(body, 0)

    cup = world.get_body(cup_name)
    local_center, (diameter, height) = approximate_as_cylinder(cup)
    center = get_point(cup) + local_center
    interior_radius = max(0.0, diameter / 2. - bead_radius - cup_thickness)
    # TODO: fill up to a certain threshold

    ty = get_type(cup_name)
    if ty in SPOON_DIAMETERS:
        # TODO: do this in a more principled way
        interior_radius = 0
        center[1] += (SPOON_LENGTHS[ty] - SPOON_DIAMETERS[ty]) / 2.

    # TODO: some sneak out through the bottom
    # TODO: could reduce gravity while filling
    world.controller.set_gravity()
    for i, bead in enumerate(beads):
        # TODO: halton sequence
        x, y = center[:2] + np.random.uniform(0, interior_radius)*unit_from_theta(
            np.random.uniform(-np.pi, np.pi))
        new_z = center[2] + height/2. + bead_radius + bead_offset
        set_point(bead, [x, y, new_z])
        set_mass(bead, masses[i])
        world.controller.rest_for_duration(drop_rate)
    world.controller.rest_for_duration(BEADS_REST)
    print('Simulated {} beads in {:3f} seconds'.format(
        len(beads), elapsed_time(start_time)))
    contained_beads = get_contained_beads(cup, beads, **kwargs)
    #wait_for_user()

    for body in beads:
        if fix_outside and (body not in contained_beads):
            set_mass(body, 0)
    for saver in savers:
        if reset_contained or (saver.body not in contained_beads):
            saver.restore()
    #wait_for_user()
    return contained_beads

##################################################

def fill_with_beads(world, bowl_name, beads, **kwargs):
    with LockRenderer(lock=True):
        with BodySaver(world.robot):
            contained_beads = pour_beads(world, bowl_name, list(beads), **kwargs)
    print('Contained: {} out of {} beads ({:.3f}%)'.format(len(contained_beads), len(beads),
        100*safe_ratio(len(contained_beads), len(beads), undefined=0)))
    return contained_beads

def estimate_spoon_capacity(world, spoon_name, beads, max_beads=100):
    beads = beads[:max_beads]
    if not beads:
        return 0
    bead_radius = np.average(approximate_as_prism(beads[0])) / 2.
    spoon_body = world.get_body(spoon_name)
    spoon_mass = get_mass(spoon_body)
    set_mass(spoon_body, STATIC_MASS)
    set_point(spoon_body, (1, 0, 1))
    set_quat(spoon_body, get_liquid_quat(spoon_name))
    capacity_beads = fill_with_beads(world, spoon_name, beads,
                                     reset_contained=True, fix_outside=False,
                                     height_fraction=1.0, top_threshold=bead_radius)
    #wait_for_user()
    set_mass(spoon_body, spoon_mass)
    return len(capacity_beads)

def stabilize(world, duration=0.1):
    # TODO: apply to simulated_problems
    with ClientSaver(world.client):
        world.controller.set_gravity()
        with BodySaver(world.robot):  # Otherwise the robot starts in self-collision
            world.controller.rest_for_duration(duration)  # Let's the objects stabilize

##################################################

def read_mass(scale_body, max_height=0.5, tolerance=1e-2):
    # Approximation: ignores connectivity outside box and bodies not resting on the body
    # TODO: could also add a force sensor and estimate the force
    scale_aabb = get_aabb(scale_body)
    extent = scale_aabb.upper - scale_aabb.lower
    lower = scale_aabb.lower + np.array([0, 0, extent[2]]) - tolerance*np.ones(3)
    upper = scale_aabb.upper + np.array([0, 0, max_height]) + tolerance*np.ones(3)
    above_aabb = AABB(lower, upper)
    total_mass = 0.
    #handles = draw_aabb(above_aabb)
    for body, link in get_bodies_in_region(above_aabb):
        if scale_body == body:
            continue
        link_aabb = get_aabb(body, link)
        if aabb_contains_aabb(link_aabb, above_aabb):
            total_mass += get_mass(body, link)
        else:
            #print(get_name(body), get_link_name(body, link))
            #handles.extend(draw_aabb(link_aabb))
            pass
    return total_mass

##################################################

class InitialRanges(object):
    def __init__(self, width_range, height_range, mass_range, pose2d_range, surface=TABLE_NAME):
        self.width_range = width_range
        self.height_range = height_range
        self.mass_range = mass_range
        self.pose2d_range = pose2d_range # Make relative to the surface origin
        self.surface = surface
    def __repr__(self):
        return repr(self.__dict__)


def create_table_bodies(world, item_ranges, randomize=True):
    perception = world.perception
    with HideOutput():
        add_data_path()
        floor_body = load_pybullet("plane.urdf")
        set_pose(floor_body, get_pose(world.robot))
        add_table(world)
        for name, limits in sorted(item_ranges.items()):
            perception.sim_items[name] = None
            if randomize:
                body = randomize_body(name, width_range=limits.width_range,
                                      height_range=limits.height_range,
                                      mass_range=limits.mass_range)
            else:
                body = load_body(name)
            perception.sim_bodies[name] = body
            # perception.add_item(name, unit_pose())
            x, y, yaw = np.random.uniform(*limits.pose2d_range)
            surface_body = perception.get_body(limits.surface)
            z = stable_z(body, surface_body) + Z_EPSILON
            pose = Pose((x, y, z), Euler(yaw=yaw))
            perception.set_pose(name, *pose)


def sample_parameters(randomize=True):
    # TODO: fix these values per object
    scale = int(randomize)
    return {
        # TODO: randomize masses here
        # lateral (linear) contact friction
        'lateralFriction': sample_norm(mu=0.5, sigma=0.1*scale, lower=0.0),
        # torsional friction around the contact normal
        'spinningFriction': sample_norm(mu=0.001, sigma=0.0005*scale, lower=0.0),
        # torsional friction orthogonal to contact normal
        'rollingFriction': sample_norm(mu=0.001, sigma=0.0005*scale, lower=0.0),
        # bouncyness of contact. Keep it a bit less than 1.
        # restitution: 0 => inelastic collision, 1 => elastic collision
        'restitution': sample_norm(mu=0.7, sigma=0.1*scale, lower=0.0, upper=1.0),
        # linear damping of the link (0.04 by default)
        'linearDamping': sample_norm(mu=0.04, sigma=0.01*scale, lower=0.01),
        # angular damping of the link (0.04 by default)
        'angularDamping': sample_norm(mu=0.04, sigma=0.01*scale, lower=0.01),
        #'contactStiffness': None,
        #'contactDamping': None,
        #'localInertiaDiagonal': (sample_norm(mu=5e-9, sigma=1e-9, lower=1e-9) * np.ones(3)).tolist(),
    }

def randomize_dynamics(world, randomize=True):
    parameters_from_name = {}
    for name, body in sorted(world.perception.sim_bodies.items()):
        print('Randomizing:', name)
        parameters_from_name[name] = sample_parameters(randomize=randomize)
        set_dynamics(body, **parameters_from_name[name])
    return parameters_from_name

##################################################

def get_default_result(skill, plan, feature=None, parameter=None):
    return {
        SKILL: skill,
        'feature': feature,
        'parameter': parameter,
        'success': (plan is not None),
        'score': None,
    }

def dump_ranges(bowls, bowl_limits):
    #print(bowls)
    print('Object ranges: base diameter, top diameter, height')
    for bowl in bowls:
        properties = get_properties(bowl)
        if properties is None:
            print(bowl, properties)
        else:
            bottom_diameter = properties.bottom_diameter * np.array(bowl_limits.width_range)
            top_diameter = properties.top_diameter * np.array(bowl_limits.width_range)
            height = properties.height * np.array(bowl_limits.height_range)
            print(bowl, bottom_diameter, top_diameter, height)
    print()
