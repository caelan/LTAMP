from __future__ import print_function

import cProfile
import os
import pstats
import numpy as np
import pddlstream.algorithms.instantiate_task

pddlstream.algorithms.instantiate_task.FD_INSTANTIATE = True

from pddlstream.algorithms.focused import solve_focused
from pddlstream.language.constants import And, PDDLProblem, print_solution
from pddlstream.language.external import DEBUG
from pddlstream.language.function import FunctionInfo
from pddlstream.language.stream import StreamInfo
from pddlstream.language.generator import from_fn, from_gen_fn, from_test
from pddlstream.utils import read, get_file_path
from pddlstream.algorithms.constraints import PlanConstraints

from perception_tools.common import is_surface, get_body_urdf, get_models_path, get_type
from plan_tools.samplers.collision import get_control_pose_collision_test, get_control_conf_collision_test, \
    get_conf_conf_collision_test, get_pose_pose_collision_test
from plan_tools.common import Conf, is_obj_type, Pose, ARMS, LEFT_ARM, TABLE, in_type_group, PR2_URDF, TOP, SIM_MATERIALS
from plan_tools.samplers.generators import BASE_FRAME
from plan_tools.samplers.move import get_motion_fn
from plan_tools.samplers.pick import get_pick_gen_fn
from plan_tools.samplers.place import get_place_fn, get_stable_pose_gen_fn, get_reachable_test
from plan_tools.samplers.pour import get_pour_gen_fn
from plan_tools.samplers.press import get_press_gen_fn
from plan_tools.samplers.push import get_push_gen_fn
from plan_tools.samplers.stir import get_stir_gen_fn
from plan_tools.samplers.scoop import get_scoop_gen_fn
from pybullet_tools.pr2_utils import get_arm_joints
from pybullet_tools.utils import get_joint_positions, get_pose, ClientSaver, connect, HideOutput, load_pybullet, \
    get_link_pose, link_from_name, set_pose, multiply, invert, set_joint_positions, \
    unit_pose, get_model_info, load_model_info, disconnect, remove_body, \
    get_configuration, joint_from_name, reset_simulation, set_configuration, \
    point_from_pose, clone_body, create_cylinder, create_box

# TODO: visualize object status (e.g. HasWater)
# TODO: dual-arm handoff

def select_values(use_values, all_values):
    if use_values is True:
        return all_values
    if use_values is False:
        return []
    return use_values

class Task(object):
    def __init__(self, init=[], goal=[], init_holding={},
                 arms=[LEFT_ARM], required=[],
                 stackable=[], graspable=[], pushable=[],
                 stirrable=[], can_scoop=[], scoopable=[],
                 pressable=[], pourable=[], can_contain=[],
                 use_kitchen=False, use_scales=False,
                 reset_arms=False, empty_arms=False,
                 reset_items=False, constraints=PlanConstraints()):
        self.init = init
        self.goal = goal
        self.arms = arms
        self.init_holding = init_holding
        self.required = set(required)
        self.stackable = [TABLE] + list(stackable)
        self.graspable = ['greenblock', 'cup'] + list(graspable) # 'spoon', 'stirrer'
        self.pushable = ['purpleblock'] + list(pushable)
        self.stirrable = ['spoon', 'stirrer'] + list(stirrable)
        self.can_scoop = ['spoon'] + list(can_scoop)
        self.scoopable = ['bowl'] + list(scoopable)
        self.pressable = ['button'] + list(pressable)
        self.can_contain = ['bowl'] + list(can_contain) # 'cup'
        self.pourable = ['cup', 'spoon'] + list(pourable) # 'bowl'
        self.use_kitchen = use_kitchen
        self.use_scales = use_scales
        self.reset_arms = reset_arms
        self.empty_arms = select_values(empty_arms, self.arms)
        self.reset_items = select_values(reset_items, self.graspable)
        self.constraints = constraints
        # TODO: init holding liquid
    @property
    def holding(self):
        return [grasp.obj_name for grasp in self.init_holding.values()]
    @property
    def movable(self):
        return self.graspable + self.pushable + self.holding
    def get_attachments(self, world):
        attachments = {}
        for arm, grasp in self.init_holding.items():
            attachment = grasp.get_attachment(world, arm)
            attachments[attachment.child] = attachment
        return attachments
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.arms)

##################################################

class PlanningWorld(object):
    def __init__(self, task, use_robot=True, visualize=False, **kwargs):
        self.task = task
        self.real_world = None
        self.visualize = visualize
        self.client_saver = None
        self.client = connect(use_gui=visualize, **kwargs)
        print('Planner connected to client {}.'.format(self.client))
        self.robot = None
        with ClientSaver(self.client):
            with HideOutput():
                if use_robot:
                    self.robot = load_pybullet(os.path.join(
                        get_models_path(), PR2_URDF), fixed_base=True)
                #dump_body(self.robot)
        #compute_joint_weights(self.robot)
        self.world_name = 'world'
        self.world_pose = Pose(unit_pose())
        self.bodies = {}
        self.fixed = []
        self.surfaces = []
        self.items = []
        #self.holding_liquid = []
        self.initial_config = None
        self.initial_poses = {}
        self.body_mapping = {}
    @property
    def arms(self):
        return self.task.arms
    @property
    def initial_grasps(self):
        return self.task.init_holding
    def get_body(self, name):
        return self.bodies[name]
    def get_table(self):
        for name in self.bodies:
            if is_obj_type(name, TABLE):
                return name
        return None
    def get_obstacles(self):
        return [name for name in self.bodies if 'obstacle' in name]
    def get_fixed(self): # TODO: make any non-movable object fixed
        movable = self.get_movable()
        return sorted(name for name in self.bodies if name not in movable)
    def get_movable(self):
        return sorted(filter(lambda item: in_type_group(item, self.task.movable), self.bodies))
    def get_held(self):
        return {grasp.obj_name for grasp in self.initial_grasps.values()}
    def set_initial(self):
        with ClientSaver(self.client):
            set_configuration(self.robot, self.initial_config)
            for name, pose in self.initial_poses.items():
                set_pose(self.get_body(name), pose)
        # TODO: set the initial grasps?
    def _load_robot(self, real_world):
        with ClientSaver(self.client):
            # TODO: set the x,y,theta joints using the base pose
            pose = get_pose(self.robot)  # base_link is origin
            base_pose = get_link_pose(self.robot, link_from_name(self.robot, BASE_FRAME))
            set_pose(self.robot, multiply(invert(base_pose), pose))
            # base_pose = real_world.controller.return_cartesian_pose(arm='l')
            # Because the robot might have an xyz
            movable_names = real_world.controller.get_joint_names()
            movable_joints = [joint_from_name(self.robot, name) for name in movable_names]
            movable_positions = real_world.controller.get_joint_positions(movable_names)
            set_joint_positions(self.robot, movable_joints, movable_positions)
            self.initial_config = get_configuration(self.robot)
            self.body_mapping[self.robot] = real_world.robot
            # TODO(caelan): can also directly access available joints
    def add_body(self, name, fixed=True):
        self.bodies[name] = load_pybullet(get_body_urdf(name), fixed_base=fixed)
        return self.bodies[name]
    def _load_bodies(self, real_world):
        with ClientSaver(self.client):
            assert not self.bodies
            #for real_body in self.bodies.values():
            #    remove_body(real_body)
            self.bodies = {}
            self.initial_poses = {}
            with HideOutput():
                for name, real_body in real_world.perception.sim_bodies.items():
                    if name in real_world.perception.get_surfaces():
                        self.add_body(name, fixed=True)
                    elif name in real_world.perception.get_items():
                        with real_world:
                            model_info = get_model_info(real_body)
                            # self.bodies[item] = clone_body(real_world.perception.get_body(item), client=self.client)
                        if 'wall' in name:
                            with real_world:
                                self.bodies[name] = clone_body(real_body, client=self.client)
                        elif 'cylinder' in name:
                            self.bodies[name] = create_cylinder(*model_info.path)
                        elif 'box' in name:
                            self.bodies[name] = create_box(*model_info.path)
                        elif model_info is None:
                            self.add_body(name, fixed=False)
                        else:
                            self.bodies[name] = load_model_info(model_info)
                    else:
                        raise ValueError(name)
                    # TODO: the floor might not be added which causes the indices to misalign
                    self.body_mapping[self.bodies[name]] = real_body
                    if name not in self.get_held():
                        self.initial_poses[name] = Pose(real_world.perception.get_pose(name))
        self.set_initial()
    def load(self, real_world): # Avoids reloading the robot
        self.fixed = tuple(real_world.perception.sim_surfaces.keys())
        self.surfaces = tuple(filter(is_surface, self.fixed))
        self.items = tuple(real_world.perception.sim_items.keys())
        self.body_mapping = {}
        #self.holding_liquid = tuple(real_world.perception.holding_liquid)
        #config = get_configuration(real_world.pr2)
        self._load_robot(real_world)
        self._load_bodies(real_world)
        self.real_world = real_world
    def stop(self):
        with ClientSaver(self.client):
            reset_simulation()
            disconnect()
    def __enter__(self):
        self.client_saver = ClientSaver(self.client)
    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.client_saver is not None
        self.client_saver.restore()
        self.client_saver = None
    def __repr__(self):
        return '{}(fixed={},movable={})'.format(
            self.__class__.__name__, self.get_fixed(), self.get_movable())

##################################################

def get_item_facts(world, item):
    task = world.task
    initial_atoms = []
    if in_type_group(item, task.graspable):
        initial_atoms.append(('Graspable', item))
        for surface in world.surfaces:  # world.bodies
            if in_type_group(surface, task.stackable):
                initial_atoms.append(('Stackable', item, surface, TOP))

    if in_type_group(item, task.pushable):
        # initial_atoms.append(('Pushable', item))
        initial_atoms.append(('CanPush', item, None))
        # for surface in world.surfaces: # world.bodies
        #    if in_type_group(surface, task.stackable):
        #        initial_atoms.append(('CanPush', item, surface))

    if in_type_group(item, task.movable):
        initial_atoms.append(('Movable', item))
    if in_type_group(item, task.pourable):
        initial_atoms.append(('Pourable', item))
    if in_type_group(item, task.can_contain):
        initial_atoms.append(('CanHold', item))
    if in_type_group(item, task.stirrable):
        initial_atoms.append(('Stirrable', item))
    if in_type_group(item, task.can_scoop):
        initial_atoms.append(('CanScoop', item))
        # if item in world.holding_liquid:
        #    initial_atoms.append(('Contains', item, ...))
    if in_type_group(item, task.scoopable):
        initial_atoms.append(('Scoopable', item))
    return initial_atoms

def get_initial_and_goal(world):
    # TODO: add an object representing the world
    task = world.task
    constant_map = {'@{}'.format(material): material for material in SIM_MATERIALS}
    initial_atoms = list(task.init) + [
        ('IsPose', world.world_name, world.world_pose),
        ('AtPose', world.world_name, world.world_pose),
    ] + [('Material', material) for material in SIM_MATERIALS]
    goal_literals = list(task.goal)
    for arm in ARMS:
        arm_joints = get_arm_joints(world.robot, arm)
        arm_conf = Conf(get_joint_positions(world.robot, arm_joints))
        constant_map['@{}'.format(arm)] = arm
        constant_map['@{}-conf'.format(arm)] = arm_conf
        initial_atoms.extend([
            ('IsConf', arm, arm_conf),
            ('AtConf', arm, arm_conf),
        ])
        if arm not in world.arms:
            continue
        initial_atoms.extend([
            ('IsArm', arm),
            ('CanMove', arm),  # Prevents double moves
        ])
        if arm not in world.initial_grasps:
            initial_atoms.append(('Empty', arm))
        if task.reset_arms:
            goal_literals.append(('AtConf', arm, arm_conf)) # Be careful of end orientation constraints
        if arm in task.empty_arms:
            goal_literals.append(('Empty', arm))
    for name, pose in world.initial_poses.items():
        initial_atoms.extend([
            ('IsPose', name, pose),
            ('AtPose', name, pose),
            # TODO: detect which frame is supporting
            ('IsSupported', name, pose, world.world_name, world.world_pose, TOP),
        ])
        if in_type_group(name, task.reset_items):
            goal_literals.append(('AtPose', name, pose))
    for arm, grasp in world.initial_grasps.items():
        name = grasp.obj_name
        initial_atoms.extend([
            ('IsGrasp', name, grasp),
            ('AtGrasp', arm, name, grasp),
        ])

    for name in world.bodies:
        initial_atoms.append(('IsType', name, get_type(name)))
    for item in world.items:
        initial_atoms.extend(get_item_facts(world, item))
    for surface in world.bodies: # world.surfaces
        if in_type_group(surface, task.pressable):
            initial_atoms.append(('Pressable', surface))
        if is_obj_type(surface, 'stove'):
            initial_atoms.append(('Stove', surface, TOP))

    print('Constants:', constant_map)
    print('Initial:', initial_atoms)
    print('Goal:', goal_literals)
    return constant_map, initial_atoms, And(*goal_literals)


def get_pddlstream(world, debug=False, collisions=True, teleport=False, parameter_fns={}):
    domain_pddl = read(get_file_path(__file__, 'pddl/domain.pddl'))
    stream_pddl = read(get_file_path(__file__, 'pddl/stream.pddl'))

    # TODO: increase number of attempts when collecting data
    constant_map, initial_atoms, goal_formula = get_initial_and_goal(world)
    stream_map = {
        'sample-motion': from_fn(get_motion_fn(world, collisions=collisions, teleport=teleport)),
        'sample-pick': from_gen_fn(get_pick_gen_fn(world, collisions=collisions)),
        'sample-place': from_fn(get_place_fn(world, collisions=collisions)),
        'sample-pose': from_gen_fn(get_stable_pose_gen_fn(world, collisions=collisions)),
        #'sample-grasp': from_gen_fn(get_grasp_gen_fn(world)),
        'sample-pour': from_gen_fn(get_pour_gen_fn(world, collisions=collisions, parameter_fns=parameter_fns)),
        'sample-push': from_gen_fn(get_push_gen_fn(world, collisions=collisions, parameter_fns=parameter_fns)),
        'sample-stir': from_gen_fn(get_stir_gen_fn(world, collisions=collisions, parameter_fns=parameter_fns)),
        'sample-scoop': from_gen_fn(get_scoop_gen_fn(world, collisions=collisions, parameter_fns=parameter_fns)),
        'sample-press': from_gen_fn(get_press_gen_fn(world, collisions=collisions)),

        'test-reachable': from_test(get_reachable_test(world)),
        'ControlPoseCollision': get_control_pose_collision_test(world, collisions=collisions),
        'ControlConfCollision': get_control_conf_collision_test(world, collisions=collisions),
        'PosePoseCollision': get_pose_pose_collision_test(world, collisions=collisions),
        'ConfConfCollision': get_conf_conf_collision_test(world, collisions=collisions),
    }
    if debug:
        # Uses an automatically constructed debug generator for each stream
        stream_map = DEBUG
    return PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, initial_atoms, goal_formula)

##################################################

def plan_actions(world, use_constraints=False, unit_costs=True, max_time=300, verbose=True, **kwargs):
    # TODO: return multiple grasps instead of one
    pr = cProfile.Profile()
    pr.enable()
    with ClientSaver(world.client):
        # TODO: be careful about the table distance
        table_pose = get_pose(world.get_body(world.get_table()))
        torso_pose = get_link_pose(world.robot, link_from_name(world.robot, 'torso_lift_link'))
        torso_in_table = multiply(invert(table_pose), torso_pose)
        # Torso wrt table: [-0.6, 0.0, 0.33]
        print('Torso wrt table:', np.array(point_from_pose(torso_in_table)).round(3).tolist())
        #wait_for_interrupt()
        problem = get_pddlstream(world, **kwargs)
        p_success = 1e-2
        eager = True
        stream_info = {
            'ControlPoseCollision': FunctionInfo(p_success=p_success, eager=eager),
            'ControlConfCollision': FunctionInfo(p_success=p_success, eager=eager),
            'PosePoseCollision': FunctionInfo(p_success=p_success, eager=eager),
            'ConfConfCollision': FunctionInfo(p_success=p_success, eager=eager),

            'test-reachable': StreamInfo(p_success=0, eager=True),
            # TODO: these should automatically be last...
            'sample-motion': StreamInfo(p_success=1, overhead=100),
        }
        # TODO: RuntimeError: Preimage fact ('order', n0, t0) is not achievable!
        constraints = world.task.constraints if use_constraints else PlanConstraints()
        solution = solve_focused(problem, planner='ff-wastar1', max_time=max_time, unit_costs=unit_costs,
                                 unit_efforts=True, effort_weight=1, stream_info=stream_info,
                                 # TODO: bug when max_skeletons=None and effort_weight != None
                                 max_skeletons=None, bind=True, max_failures=0,
                                 search_sample_ratio=0, constraints=constraints,
                                 verbose=verbose, debug=False)
        #solution = solve_incremental(problem, unit_costs=unit_costs, verbose=True)
        print_solution(solution)
        plan, cost, evaluations = solution
    pr.disable()
    if verbose:
        pstats.Stats(pr).sort_stats('tottime').print_stats(10) # cumtime | tottime
    return plan
