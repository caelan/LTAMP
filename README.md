<!-- https://www.markdownguide.org/basic-syntax/ -->
<!-- https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet -->

# LTAMP
Learning for Task and Motion Planning

## Modules

### Learning
The learning module conducts primitive-learning experiments and learns models for the collected data.

### Planning
The planning module generates plans using the learned primitives.

Relevant repositories:
* [ss-pybullet](https://github.com/caelan/ss-pybullet)
* [pddlstream](https://github.com/caelan/pddlstream)

### Control
The control module provides an interface for performing both simulated and real motion.

---

## Installation
```
$ pip install pybullet pyquaternion pyusb psutil
$ cd ltamp-pr2
ltamp-pr2$ git submodule update --init --recursive
ltamp-pr2$ ./pddlstream/FastDownward/build.py release64
```

To train the learners, install the following:
```
$ pip install psutil scikit-learn
```

Make sure to update the submodules and rebuild FastDownward upon pulling as they may have changed.

### Forward/Inverse Kinematics
Before using any kind of IK through [pr2_ik.py](control_tools/ik/pr2_ik.py), run [setup.py](control_tools/ik/ik_tools/setup.py) file from the [ik](control_tools/ik) directory.
```
ltamp_pr2/$ cd control_tools/ik/
ltamp_pr2/control_tools/ik/$ python setup.py build
```
Follow the prompts at the end of the setup script to automatically move the necessary files.

For details about setting up and using the existing FK and IK solvers and generating new IK solvers, check the [FK and IK README](control_tools/ik).

---

## Planning in simulation
To run the LTAMP system in simulation, [run](https://github.mit.edu/Learning-and-Intelligent-Systems/ltamp_pr2/blob/master/plan_tools/run_simulation.py):
```
ltamp_pr2$ python -m plan_tools.run_simulation [-h] [-p PROBLEM] [-e] [-c] [-v]
```
