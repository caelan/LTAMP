# Forward Kinematics and Inverse Kinematics

## Existing Solutions

There are two existing solutions for FK and IK: [PyBullet](pb_ik.py) and [Compiled C++](pr2_ik.py). In spite of their names, both of these files do FK as well as IK. Both are currently only configured for the PR2 robot. [Compiled C++](pr2_ik.py) is recommended for better performance in almost every way.

### Set Up Instructions

[Compiled C++](pr2_ik.py) requires some additional work.
1. Navigate to [setup_pr2_ik_basic.py](ik_tools/pr2_with_sensor_ik/setup_pr2_ik_basic.py).
2. Make sure that [left_arm_ik.cpp](ik_tools/pr2_with_sensor_ik/left_arm_ik.cpp) and [right_arm_ik.cpp](ik_tools/pr2_with_sensor_ik/right_arm_ik.cpp) are in the same directory as [setup_pr2_ik_basic.py](ik_tools/pr2_with_sensor_ik/setup_pr2_ik_basic.py). This should be true by default.
3. Run [setup_pr2_ik_basic.py](ik_tools/pr2_with_sensor_ik/setup_pr2_ik_basic.py) to generate the ikLeft.so and ikRight.so files for Python.

   `python setup_pr2_ik_basic.py build`
4. [setup_pr2_ik_basic.py](ik_tools/pr2_with_sensor_ik/setup_pr2_ik_basic.py) has an option to automatically move the generated .so files. It will also delete the unneeded build folder. The .so files will be placed in the control_tools/ik/ directory. If copies of them already exist there, the existing copies will be deleted. This option will also check that the .so files can be imported from the [ik](.) directory where [pr2_ik.py](pr2_ik.py) is.
5. Make sure that [pr2_ik.py](pr2_ik.py) is in the same directory as the .so files. You can also place the .so files anywhere else where they can be seen.

If you don't want [setup_pr2_ik_basic.py](ik_tools/pr2_with_sensor_ik/setup_pr2_ik_basic.py) to automatically move the .so files, you will need to move them manually. Move the .so files from the generated build folder to [ik](.) or somewhere else on the path. The exact location of the .so files will depend on your machine. Once you have those two files, the build folder can be deleted.

After that, import 'pr2_ik' to use the solver. Additional set up instructions can be found inside [setup_pr2_ik_basic.py](ik_tools/pr2_with_sensor_ik/setup_pr2_ik_basic.py) and [pr2_ik.py](pr2_ik.py).

There could potentially be a problem with importing the Python header file. If so, try changing the line `#include "python2.7/Python.h"` to `#include "Python.h"`. That line appears in [ikfast.h](ik_tools/pr2_with_sensor_ik/ikfast.h), [left_arm_ik.cpp](ik_tools/pr2_with_sensor_ik/left_arm_ik.cpp), and [right_arm_ik.cpp](ik_tools/pr2_with_sensor_ik/right_arm_ik.cpp).


[PyBullet](pb_ik.py) should work without additional dependencies. Just check that the path to the [PR2 URDF file](../../models/pr2_description/pr2.urdf) in [models](../../models) is still accurate.

### Usage Instructions

For both solvers, the end effector is the gripper tool frame for each arm. On the PR2, the gripper tool frame has it's origin at the point where the two gripper fingers close. The orientation is taken relative to the hand orientation. Use PyBullet to visualize the frame if more information is needed.

For [pr2_ik](pr2_ik.py), the recommended usage is to call `arm_ik` and `arm_fk`. Note that the extra force sensor on the right arm is not accounted for. Use the [transformation utility](ik_tools/transformations.py) to move to \(-0.0356, 0, 0\) in the gripper tool frame.

#### arm_ik
* arm: 'l' or 'r', depending on which arm is being solved for
* pos: \[x, y, z\], position vector in the base link's frame
* quat: \[i, j, k, r\] or \[r, i, j, k\], quaternion in the base link's frame, format controlled by real_first
* torso: value for height of torso lift joint, available in [ros_controller](../retired/ros_controller.py)
* current: joint configuration, the solution closest to current will be selected, defaults to \[0, 0, 0, 0, 0, 0, 0\]
* return_all: if True, returns all legal solutions instead of just the one closest to current
* real_first: if True, quaternions in \[r, i, j, k\], if False, quaternions in \[i, j, k, r\], defaults to False
* dist_func: custom distance function for picking the closest IK solution, overrides current

#### arm_fk
* arm: 'l' or 'r', depending on which arm is being solved for
* config: \[torso, arm_config, ... \] or \[arm_config, ... \], if no torso value is provided, torso must be used
* torso: value for height of torso, added to config if needed
* real_first: if True, quaternions in \[r, i, j, k\], if False, quaternions in \[i, j, k, r\], defaults to False

For both solvers, the end effector pose is taken relative to the pose of the base link. This matches the behavior of `return_cartesian_pose` in [ros_controller.py](../retired/ros_controller.py). The raw C++ solver and PyQuaternion use quaternions in the form (r, i, j, k). [pr2_ik](pr2_ik.py), [pb_ik](pb_ik.py) and [ros_controller](../retired/ros_controller.py) use quaternions in the form (i, j, k, r). If you want to use [pr2_ik](pr2_ik.py) with quaternions in the form (r, i, j, k), set the optional parameter `real_first=True`. Note that for PyBullet, if you initialize the robot at `(0, 0, 0)`, the base link will be placed at `(0, 0, 0.051)`. Since FK and IK take the end effector position relative to the world coordinates, it can be simpler to just initialize the robot at `(0, 0, -0.051)`.

[pr2_ik](pr2_ik.py) is a closed form solver. Since it is set up to solve for a position and an orientation, the solver must have exactly 6 undetermined joints to work with. From the base link to the gripper tool frame, 8 joints (torso + 7 arm joints) influence the pose of the end effector. The torso value must be provided. The upper arm roll joint value is determined by repeatedly running IK for different values of that joint. Adjust the `UPPER_ARM_STEPS` parameter to change how fine the computation is. There is also a parameter `CURRENT_POSE_FIRST` which regulates whether the solver will default to the current upper arm roll value if a solution is found. If a solution cannot be found with the current value, the best solution over the entire range will be returned. The best solution is determined by the closed legal solution to the current arm configuration. The joint limits are written in the [file](pr2_ik.py). For the joints with limits, the distance is determined by the Euclidean distance in joint space. For joints without limits, the angular distance is added into the total Euclidean distance. Those joints are adjusted to be within ±π of the current value. If the current configuration is not provided, the solver defaults to 0 for all the joint values.

[pb_ik](pb_ik.py) is a numerical solver. Some of parameters like `max_iterations` and `num_attempts` can be tuned. However, this solver is far slower and vastly inferior to [pr2_ik](pr2_ik.py). In particular, [pb_ik](pb_ik.py) does not consider joint limits and often returns solutions which violate joint limits.

## Generating New Solvers

[pb_ik](pb_ik.py) can be modified to fit a new robot by using a new URDF file. From there, adjust the joints being set and the subbody being clones in the IK method. Although this may work, it is not recommended due to the poor performance of the PyBullet IK solver. This would possibly be useful for collision detection.

[pr2_ik](pr2_ik.py) is a Python wrapper for a set of C++ files which were generated from OpenRAVE and IKFast. The original C++ files can be found in [unmodified_ik_files](ik_tools/unmodified_ik_files). Those files were modified to create Python bindings for ease of use. Most of the changes are included within the special comments shown below. The only other changes are the removal of some unnecessary C++ variables.

```C++
//// START
// New Code
// New Code
//// END
```

Those C++ files can be generated using the [ik_generator.py](ik_tools/pr2_with_sensor_ik/ik_generator.py) script. Although that file is currently set up for the PR2, it can be adjusted for a new robot. OpenRAVEPy will be required to generate the original C++ files, but once those are made, the dependency is no longer needed. Switch out the path when reading the robot URI to load a new robot. Note that .dae files are recommended. Other alternatives may be possible in the OpenRAVE documentation. After that, the only line that is of any real importance is the call to `generateIkSolver`. The baselink is the link index of the base frame. The eelink is the link index of the desired end effector. Link indexes can be found by using the `GetLinks()` and `GetLink('link_name')` functions of kinbody to get the link objects. From there, each link has a `GetLinkIndex()` function. solvefn determines what type of IK Solver will be generated. For more options, check the OpenRAVE documentation. For the current 6D (position and orientation) solver, there must be exactly 6 joints between the base link and the end effector. Use `kinbody.GetChain(baselink, eelink)` to get the joints between the two links. Check that `joint.GetDOFIndex() != -1` to discard all the fixed joints. If there are more than 6 movable joints in the chain, the extra joints must be declared in freeindices. Use `GetDOFIndex` to get the value for freeindices. See [ik_generator.py](ik_tools/pr2_with_sensor_ik/ik_generator.py) for additional comments. Once the C++ files have been generated, remember to include the [ikfast.h](ik_tools/pr2_with_sensor_ik/ikfast.h) header file. Adjusting the command to include Python.h may be needed. From there, add the python bindings. Look at [left_arm_ik.cpp](ik_tools/pr2_with_sensor_ik/left_arm_ik.cpp) and [right_arm_ik.cpp](ik_tools/pr2_with_sensor_ik/right_arm_ik.cpp) for guidelines. Most of the changes are at the bottom. Finally, make a relevant version of [setup_pr2_ik_basic.py](ik_tools/pr2_with_sensor_ik/setup_pr2_ik_basic.py) to build the .so files. Follow the comments in the setup file for instructions.

## Python Version Issues

The include for the python header files should work automatically without referencing the version numbers. Be sure the install the dev version of your particular Python version so you actually have those header files. You can look for them in `usr/bin/python3.4` on linux. Make sure this include (`#include "Python.h") appears as the first include so we don't get scope issues later. The code added and the end of the C++ files should automatically support both versions of Python. The appropriate changes to the module initialization process are labelled in the code, so you can manually comment them out if problems arise.