import numpy as np

from perception_tools.common import create_name
from plan_tools.common import LEFT_ARM, ARMS, TOP, RIGHT_ARM, COFFEE, SUGAR, is_obj_type
from plan_tools.samplers.grasp import Grasp
from plan_tools.planner import Task
from pybullet_tools.pr2_utils import WIDE_LEFT_ARM

HEAD_CONF = [0, np.pi/3]
LEFT_ARM_CONF = WIDE_LEFT_ARM # LEFT_SIDE_CONF
STOVE_NAME = 'stove'
PLACEMAT_NAME = 'placemat'
BUTTON_NAME = 'button'


##################################################

def test_pick():
    arms = [LEFT_ARM]
    #arms = ARMS
    goal_block = create_name('greenblock', 1)

    init = []
    goal = [
        ('Grasped', arms[0], goal_block),
    ]
    return Task(init=init, goal=goal, arms=arms,
                reset_arms=True)


def test_pour():
    arms = [LEFT_ARM]
    cup_name = create_name('bluecup', 1)
    bowl_name = create_name('bowl', 1)

    init = [
        ('Contains', cup_name, COFFEE),
    ]
    goal = [
        ('Contains', bowl_name, COFFEE),
    ]
    return Task(init=init, goal=goal, arms=arms,
                reset_arms=True, empty_arms=True)


def test_pour_two_cups():
    #arms = ARMS
    arms = [LEFT_ARM]

    cup_name = create_name('bluecup', 2) # bluecup2 is on the left with a higher Y value
    bowl_name = create_name('bluecup', 1)

    init = [
        ('Contains', cup_name, COFFEE),
    ]
    goal = [
        ('Contains', bowl_name, COFFEE),
    ]
    return Task(init=init, goal=goal, arms=arms,
                reset_arms=True, empty_arms=True)


def test_push():
    arms = [LEFT_ARM]
    #block_name = get_name('greenblock', 1)
    block_name = create_name('purpleblock', 1)

    goal_pos2d = np.array([0.6, 0.15])
    init = [
        ('CanPush', block_name, goal_pos2d),
    ]
    goal = [
        ('InRegion', block_name, goal_pos2d),
    ]
    return Task(init=init, goal=goal, arms=arms,
                reset_arms=True, empty_arms=True)


def test_cook():
    arms = ARMS
    #arms = [LEFT_ARM]
    broccoli_name = create_name('greenblock', 1)

    init = [
       ('IsButton', BUTTON_NAME, STOVE_NAME),
       ('Stackable', broccoli_name, STOVE_NAME, TOP),
       ('Stackable', broccoli_name, PLACEMAT_NAME, TOP),
    ]
    goal = [
        ('On', broccoli_name, PLACEMAT_NAME, TOP),
        ('Cooked', broccoli_name),
    ]
    return Task(init=init, goal=goal, arms=arms,
                use_kitchen=True, reset_arms=True, empty_arms=True)


def test_kitchen():
    arms = [LEFT_ARM]
    broccoli_name = create_name('greenblock', 1)
    cup_name = create_name('bluecup', 1)

    init = [
       ('IsButton', BUTTON_NAME, STOVE_NAME),
       ('Stackable', broccoli_name, STOVE_NAME, TOP),
       ('Stackable', broccoli_name, PLACEMAT_NAME, TOP),
       ('Stackable', cup_name, PLACEMAT_NAME, TOP),
       ('Contains', cup_name, COFFEE),
    ]
    goal = [
        ('On', cup_name, PLACEMAT_NAME, TOP),
        ('On', broccoli_name, PLACEMAT_NAME, TOP),
        ('Cooked', broccoli_name),
        ('Contains', cup_name, COFFEE),
    ]
    return Task(init=init, goal=goal, arms=arms,
                use_kitchen=True, reset_arms=True, empty_arms=True)


def test_stacking():
    arms = [LEFT_ARM]
    #arms = [RIGHT_ARM]
    #arms = ARMS
    cup_name = create_name('bluecup', 1)
    #cup_name = get_name('greenblock', 2)
    green_name = create_name('greenblock', 1)
    #green_name = get_name('bowl', 1)
    purple_name = create_name('purpleblock', 1)

    init = [
       ('Graspable', cup_name),
       ('Graspable', green_name),
       ('Stackable', cup_name, green_name, TOP),
       ('Stackable', green_name, purple_name, TOP),
    ]
    goal = [
       ('On', cup_name, green_name, TOP),
       ('On', green_name, purple_name, TOP),
    ]
    return Task(init=init, goal=goal, arms=arms,
                reset_arms=True, empty_arms=True)


def test_transfer():
    arms = [LEFT_ARM, RIGHT_ARM]
    green_name = create_name('greenblock', 1)

    init = [
       ('Stackable', green_name, STOVE_NAME, TOP),
    ]
    goal = [
       ('On', green_name, STOVE_NAME, TOP),
    ]
    return Task(init=init, goal=goal, arms=arms,
                reset_arms=True, empty_arms=True)


def test_holding():
    #arms = [LEFT_ARM, RIGHT_ARM]
    arms = [RIGHT_ARM]
    item_type = 'greenblock'
    #item_type = 'bowl'
    item_name = create_name(item_type, 1)

    init = [
        ('Graspable', item_name)
    ]
    goal = [
        #('Holding', item_name),
        ('HoldingType', item_type),
    ]
    return Task(init=init, goal=goal, arms=arms)


def test_pour_whitebowl():
    #arms = ARMS
    arms = [LEFT_ARM]
    cup_name = create_name('bluecup', 1)
    bowl_name = create_name('whitebowl', 1)

    init = [
        ('Contains', cup_name, COFFEE),
    ]

    goal = [
        ('Contains', bowl_name, COFFEE),
    ]
    return Task(init=init, goal=goal, arms=arms,
                reset_arms=True, empty_arms=True)


def test_stack_pour():
    #arms = ARMS
    arms = [LEFT_ARM]
    purple_name = create_name('purpleblock', 1)
    #purple_name = PLACEMAT_NAME
    cup_name = create_name('bluecup', 1)
    #bowl_type = 'bowl'
    bowl_type = 'whitebowl'
    bowl_name = create_name(bowl_type, 1)
    # TODO: cook & pour

    init = [
        ('Contains', cup_name, COFFEE),
        ('Stackable', bowl_name, purple_name, TOP),
    ]

    goal = [
        ('Contains', bowl_name, COFFEE),
        ('On', bowl_name, purple_name, TOP),
    ]
    return Task(init=init, goal=goal, arms=arms, graspable=[bowl_type],
                reset_arms=True, empty_arms=True)


def test_push_pour():
    arms = ARMS
    #arms = [LEFT_ARM]
    cup_name = create_name('bluecup', 1)
    bowl_type = 'bowl'
    bowl_name = create_name(bowl_type, 1)

    init = [
        ('Contains', cup_name, COFFEE),
        ('CanPush', bowl_name, LEFT_ARM),
    ]

    goal = [
        ('Contains', bowl_name, COFFEE),
        ('InRegion', bowl_name, LEFT_ARM),
    ]
    return Task(init=init, goal=goal, arms=arms, pushable=[bowl_type],
                reset_arms=True, empty_arms=True)

##################################################

def get_spoon_init_holding(arm, spoon_name):
    # TODO: compute a real grasp for the initial state
    if is_obj_type(spoon_name, 'grey_spoon'):
        grasp_point = (0.084, -0.001, 0.0)
    elif is_obj_type(spoon_name, 'orange_spoon'):
        grasp_point = (0.10854, -0.00200, 0.0) # TODO: how did I measure this previously?
    elif is_obj_type(spoon_name, 'green_spoon'):
        grasp_point = (0.135, -0.0025, 0.0) # 0.1165
    else:
        raise NotImplementedError(spoon_name)
    grasp = Grasp(spoon_name, index=0,
                  grasp_pose=(grasp_point, (0.5, 0.5, 0.5, -0.5)),
                  pre_direction=[0.1, 0., 0.], grasp_width=0.0644705882353)
    return {arm: grasp}

def test_coffee():
    # TODO: ensure you use get_name when running on the pr2
    spoon_name = create_name('orange_spoon', 1)  # grey_spoon | orange_spoon | green_spoon
    coffee_name = create_name('bluecup', 1)
    sugar_name = create_name('bowl', 1)
    bowl_name = create_name('whitebowl', 1)

    init_holding = get_spoon_init_holding(RIGHT_ARM, spoon_name)
    init = [
        ('Contains', coffee_name, COFFEE),
        ('Contains', sugar_name, SUGAR),
    ]

    goal = [
        #('Contains', bowl_name, COFFEE),
        #('Contains', bowl_name, SUGAR),
        ('Mixed', bowl_name),
    ]
    return Task(init=init, init_holding=init_holding, goal=goal, arms=ARMS, # arms=[RIGHT_ARM],
                reset_arms=True, empty_arms=[LEFT_ARM])

##################################################

# grey: 9.7 - 9.3 = 0.4 oz
# orange: 10.3 - 9.3 = 1 oz
# green: 10.9 - 9.3 = 1.6 oz

PROBLEMS = [
    test_pick,
    test_pour,
    test_pour_two_cups,
    test_push,
    test_cook,
    test_stacking,
    test_transfer,
    test_holding,
    test_pour_whitebowl,
    test_stack_pour,
    test_push_pour,
    test_kitchen,
    test_coffee,
]