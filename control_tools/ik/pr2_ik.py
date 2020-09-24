#!/usr/bin/env python

# TODO: Rewrite this using poses

# If these imports fail, check that you ran setup_ik.py
# Instructions are in the comments of that file
# Remember to move the .so files into this folder
try:
    from .ikLeft import leftIK, leftFK
    from .ikRight import rightIK, rightFK
except ImportError:
    # If these instructions aren't enough, check the named files for more details.
    print("Did you forget to run the setup script? It's called setup_ik.py.")
    print('Run it with "python setup_ik.py build" from it\'s file location.')
    print("Maybe the .so files are in the wrong location?")
    print('They\'re called "ikLeft.so" and "ikRight.so".')
    print('Put them in the same directory as pr2_ik.py.')
    assert False, 'IK Import Failed: probably not setup'

from pybullet import getMatrixFromQuaternion

import random
import numpy as np
from math import pi as PI
# only needed to convert the quaternion to a matrix for the ik solver
# quaternions are in [i,j,k,r]
# set the optional parameter real_first=True to use quaternions in [r, i, j, k]
# could also be done using pybullet

# The joint limits are taken from the pr2.urdf file. They look accurate within 0.02.
LEFT_UPPER_ARM_LIMITS = [-0.8000, 3.9000] # left upper arm roll
RIGHT_UPPER_ARM_LIMITS = [-3.9000, 0.8000] # right upper arm roll

def get_arm_ik_generator(arm, pos, quat, torso, upper_limits=None, max_attempts=10):
    if upper_limits is None:
        upper_limits = LEFT_UPPER_ARM_LIMITS if arm == 'l' else RIGHT_UPPER_ARM_LIMITS
    ik_fn = leftIK if arm == 'l' else rightIK
    for attempt in range(max_attempts):
        upper = random.uniform(*upper_limits) # TODO(caelan): Halton sequence
        solutions = _do_ik(pos, quat, torso, upper, ik_fn)
        random.shuffle(solutions)
        for arm_conf in solutions:
            # TODO(caelan): check limits here?
            #print('IK attempt:', attempt)
            yield arm_conf


def _do_ik(pos, quats, torso, upper_arm, ik_func):
    # a nicer wrapper for IK calls
    # it returns a list of IK solutions with the torso value already stripped out
    # pos is an [x, y, z] list for position
    # quat can be either [i,j,k,r] for a single pose or [[i,j,k,r],[i,j,k,r],...] for multiple poses.
    # All the parameters must be provided for the solver to work. Orientation is not optional.
    # the return format for ik_func is [solution1, solution2, ...] where solution1 = [torso_value, shoulder_pan_value, ...]
    solutions = []
    if not isinstance(quats[0], (list, tuple)): # If quats does not contain multiple quaternion lists
        quats = [quats]
    for quat in quats:
        rot = np.array(getMatrixFromQuaternion(quat)).reshape(3, 3).tolist()
        sol = ik_func(rot, pos, [torso, upper_arm])
        if sol is not None:
            solutions.extend(sol)
    # remove torso values
    return [sol[1:] for sol in solutions]


def _do_fk(config, fk_func, real_first=False):
    # a nicer wrapper for FK calls
    # returns pos, quat where pos = [x, y, z] and quat is a quaternion = [i, j, k, r]
    from pyquaternion import Quaternion
    # TODO(caelan): remove pyquaternion
    pos, rot = fk_func(config)
    quat = Quaternion(matrix=np.array(rot))
    quat = quat if quat.real >= 0 else -quat # solves q and -q being same rotation
    quat = quat.unit.elements.tolist()
    # switch from [r, i, j, k] to [i, j, k, r]
    if not real_first:
        quat = quat[1:] + [quat[0]]
    return pos, quat

euclidean = lambda s, c: abs(s - c)
# This one is definitely correct. I think the shorter one works too though
# angular = lambda s, c: min((s - c) % (2*PI), (2*PI - ((s - c)) % (2*PI)))
angular = lambda s, c: min((s - c) % (2*PI), (c - s) % (2*PI))

DIST_FN_LIST = [euclidean] * 4 + [angular, euclidean, angular]
arm_difference_fn = lambda q1, q2: [fn(p1, p2) for fn, p1, p2 in zip(DIST_FN_LIST, q1, q2)]
