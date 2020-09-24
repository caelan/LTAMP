#!/usr/bin/env python

from __future__ import print_function

# Run this file from this directory with the following command:
# python setup_pr2_ik_basic.py build
# I'm not sure that the build argument is necessary, but include it anyway.
# Ignore any the warning messages about unused variables or variables being set but not used.
# Although I removed those variables, so it shouldn't have thise warnings anymore.
# After the .so files are generated, there is an option to move them automatically.
# That will put them into the ik folder. It will also delete the build folder.
# If you don't want to use that, manually move the .so files to wherever you want to store them.
# They should be called ikLeft.so and ikRight.so.

from distutils.core import setup, Extension

# left_arm_ik.cpp and right_arm_ik.cpp contain the ik solvers I edited
# They did need some modification for the C++ to Python bindings
# ikfast.h also has a couple lines added
# All the additional code is contained within these special comments:
'''
//// Start
Extra C++ code
//// End
'''
# To add the python bindings, simply add all the code within the special comments
# That should give you working IK although it will have the unused variable warnings
# One possible source of error is the include statement for Python. The option that is working now is
'''
//// Start
#include "python2.7/Python.h"
//// End
'''
# It might help to change that to the following. It appears in ikfast.h, left_arm_ik.cpp, and right_arm_ik.cpp.
'''
//// Start
#include "Python.h"
//// End
'''

try:
    user_input = raw_input
except NameError:
    user_input = input

leftModule = Extension('ikLeft',
                    sources=['left_arm_ik.cpp'])

rightModule = Extension('ikRight',
                    sources=['right_arm_ik.cpp'])

setup(name='ikLeft',
    version='1.0',
    description="IK for PR2's left arm",
    ext_modules=[leftModule])

setup(name='ikRight',
    version='1.0',
    description="IK for PR2's right arm",
    ext_modules=[rightModule])

print('If copies of ikLeft.so and ikRight.so already exist in the current directory, they will be overriden.')
choice = user_input('Move .so files for you? (y/n) ')

if 'y' in choice.lower():
    import os, shutil, sys
    left_path = ''
    right_path = ''
    for step in os.walk(os.getcwd()):
        if 'ikLeft.so' in step[2]:
            left_path = os.path.join(step[0], 'ikLeft.so')
        if 'ikRight.so' in step[2]:
            right_path = os.path.join(step[0], 'ikRight.so')
    print('Left IK Path: ' + left_path)
    print('Right IK Path: ' + right_path)

    build_folder = os.path.join(os.getcwd(), 'build')
    ik_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)
    left_target = os.path.join(ik_folder, 'ikLeft.so')
    right_target = os.path.join(ik_folder, 'ikRight.so')

    ik_files = os.listdir(ik_folder)
    if 'ikLeft.so' in ik_files:
        print('Existing ikLeft.so found. Deleting ...')
        os.remove(left_target)
    if 'ikRight.so' in ik_files:
        print('Existing ikRight.so found. Deleting ...')
        os.remove(right_target)

    print(left_target)
    os.rename(left_path, left_target)
    print('Left IK: ' + left_target)
    os.rename(right_path, right_target)
    print('Right IK: ' + right_target)
    print('Build Folder: ' + build_folder)
    print('Deleting ...')
    shutil.rmtree(build_folder)
    print('Done')
    sys.path = [ik_folder]
    try:
        import ikLeft, ikRight
        print('IK Successful')
    except ImportError:
        print('IK Failed')