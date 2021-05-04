<!-- https://www.markdownguide.org/basic-syntax/ -->
<!-- https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet -->

# LTAMP

Learning for Task and Motion Planning (LTAMP)

## Overview

Robotic multi-step manipulation planning for a [PR2](http://wiki.ros.org/Robots/PR2) robot using both learned and engineered models of primitive actions.

* Learned: pour, push, scoop, stir
* Engineered: move, pick, place, press

<!--## Gallery-->
[<img src="https://img.youtube.com/vi/VZmfC_RWlps/0.jpg" height="200">](https://youtu.be/VZmfC_RWlps)
[<img src="https://img.youtube.com/vi/mDG69aGqGsA/0.jpg" height="200">](https://youtu.be/mDG69aGqGsA)
[<img src="https://img.youtube.com/vi/hz1EC8TkaZs/0.jpg" height="200">](https://youtu.be/hz1EC8TkaZs)

## References
<!--## Citation-->

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
[setup.py](control_tools/ik/ik_tools/setup.py) - compiles an [IKFast](http://openrave.org/docs/0.8.2/openravepy/ikfast/) analytical IK program for both of the PR2's arms

```
LTAMP$ cd control_tools/ik/
LTAMP$ control_tools/ik/$ python setup.py build
```
<!--Follow the prompts at the end of the setup script to automatically move the necessary files-->

See [README](control_tools/ik/README.md) for details about using the existing and generating new IK solvers.

<!--### Real World-->

---

## Examples

Planning, experimentation, and learning in a simulated [PyBullet](https://github.com/bulletphysics/bullet3) environment.

### Planning

[run_simulation.py](plan_tools/run_simulation.py): tests the planning module in simulation
```
LTAMP$ python -m plan_tools.run_simulation -h
usage: run_simulation.py [-h] [-c] [-e] [-p {test_block,test_coffee,test_holding,test_pour,test_push,test_shelves,test_stack_pour,test_stacking,test_stir}] [-s SEED] [-v] [-d]

optional arguments:
  -h, --help            show this help message and exit
  -c, --cfree           When enabled, disables collision checking (for debugging).
  -e, --execute         When enabled, executes the plan using physics simulation.
  -p {test_block,test_coffee,test_holding,test_pour,test_push,test_shelves,test_stack_pour,test_stacking,test_stir}, --problem {test_block,test_coffee,test_holding,test_pour,test_push,test_shelves,test_stack_pour,test_stacking,test_stir}
                        The name of the problem to solve.
  -s SEED, --seed SEED  The random seed to use.
  -v, --visualize_planning
                        When enabled, visualizes planning rather than the world (for debugging).
  -d, --disable_drawing
                        When enabled, disables drawing names and forward reachability.

```

[<img src="https://img.youtube.com/vi/t7D3elW_05E/0.jpg" height="200">](https://youtu.be/t7D3elW_05E)
[<img src="https://img.youtube.com/vi/0CetLZZ1mCM/0.jpg" height="200">](https://youtu.be/0CetLZZ1mCM)

### Data Collection

[collect_simulation.py](learn_tools/collect_simulation.py): collects manipulation-primitive data in simulation
```
LTAMP$ python -m learn_tools.collect_simulation -h
usage: collect_simulation.py [-h] [-f FN] [-n NUM] -p {pour,push,scoop,stir} [-t TIME] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -f FN, --fn FN        The parameter function to use.
  -n NUM, --num NUM     The number of samples to collect.
  -p {pour,push,scoop,stir}, --problem {pour,push,scoop,stir}
                        The name of the skill to learn.
  -t TIME, --time TIME  The max planning runtime for each trial.
  -v, --visualize       When enabled, visualizes execution.
```

[<img src="https://img.youtube.com/vi/IqocaU8iMXg/0.jpg" height="200">](https://youtu.be/IqocaU8iMXg)
[<img src="https://img.youtube.com/vi/GKYrYT0Q5yE/0.jpg" height="200">](https://youtu.be/GKYrYT0Q5yE)

<!--### Learning

```
LTAMP$ python -m learn_tools.run_active
LTAMP$ python -m learn_tools.run_pr2_active
```-->


<!--### Other

```
LTAMP$ python -m data.enunerate_trials
LTAMP$ python -m learn_tools.analyze_experiment
LTAMP$ python -m learn_tools.analyze.visualize_diverse
LTAMP$ python -m learn_tools.analysis.visualize_pours
LTAMP$ python -m learn_tools.retired.run_taskkernel
LTAMP$ python -m learn_tools.retired.run_sample
LTAMP$ python -m learn_tools.retired.collect_pr2
LTAMP$ python -m learn_tools.retired.unify_pr2_trials
LTAMP$ python -m plan_tools.retired.run_pr2
LTAMP$ python -m retired.mesh_tools.run_mesb
LTAMP$ python -m retired.utils.scale_reader
```-->

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
