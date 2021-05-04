<!-- https://www.markdownguide.org/basic-syntax/ -->
<!-- https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet -->

# LTAMP

Learning for Task and Motion Planning (LTAMP)

## Overview

Robotic multi-step manipulation planning using both learned and engineered models of primitive actions.


<!--## Citation-->

## References

Zi Wang*, Caelan Reed Garrett*, Leslie Pack Kaelbling, Tomás Lozano-Pérez. 
Learning compositional models of robot skills for task and motion planning, 
The International Journal of Robotics Research (IJRR), 2020.

Zi Wang, Caelan R. Garrett, Leslie P. Kaelbling, Tomás Lozano-Pérez. 
Active model learning and diverse action sampling for task and motion planning, 
International Conference on Intelligent Robots and Systems (IROS), 2018. 

## Installation

```
$ git clone --recursive git@github.com:caelan/LTAMP.git
$ cd LTAMP
LTAMP$ ./setup.bash
```
<!--LTAMP$ pip install -r requirements.txt
LTAMP$ git submodule update --init --recursive
LTAMP$ ./pddlstream/FastDownward/build.py release64-->

<!--Make sure to update the submodules and rebuild FastDownward upon pulling as they may have changed.-->



### Inverse Kinematics (IK)
<!--Forward/Inverse Kinematics-->

<!--Before using any kind of IK through [pr2_ik.py](control_tools/ik/pr2_ik.py), 
run [setup.py](control_tools/ik/ik_tools/setup.py) file from the [ik](control_tools/ik) directory.-->

<!--http://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/ikfast/ikfast_tutorial.html-->
[setup.py](control_tools/ik/ik_tools/setup.py) - compiles an [IKFast](http://openrave.org/docs/0.8.2/openravepy/ikfast/) analytical IK program for both of the PR2's

```
LTAMP$ cd control_tools/ik/
LTAMP$ control_tools/ik/$ python setup.py build
```
<!--Follow the prompts at the end of the setup script to automatically move the necessary files-->

See [README](control_tools/ik/README.md) for details about using the existing and generating new IK solvers.

---

## Examples

### Planning

[run_simulation.py](plan_tools/run_simulation.py): tests the planning module in simulation
```
LTAMP$ python -m plan_tools.run_simulation [-h] [-p PROBLEM] [-e] [-c] [-v]
```

### Data Collection

[collect_simulation.py](learn_tools/collect_simulation.py): collects manipulation-primitive data in simulation
```
LTAMP$ python -m learn_tools.run_simulation [-h] [-p PROBLEM] [-e] [-c] [-v]
```


### Learning

TBD

---

## Modules

### Planning

The planning module generates plans using the learned primitives.

Relevant planning submodules:
* [pddlstream](https://github.com/caelan/pddlstream) - Task and Motion Planning (TAMP)
* [ss-pybullet](https://github.com/caelan/ss-pybullet) - PyBullet Robot Planning

### Learning

The learning module conducts manipulation-primitive data collection experiments and learns models from the collected data.

### Control

The control module provides an interface for executing both simulated and real motion.
