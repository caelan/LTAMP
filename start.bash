# download this bash file and run 'source start.bash' in a folder where you would like to install ltamp_pr2
sudo apt install git
git clone git@github.mit.edu:Learning-and-Intelligent-Systems/ltamp_pr2.git
cd ltamp_pr2
git submodule update --init --recursive
sudo apt install python-pip cmake
pip install pyquaternion numpy scipy sklearn pybullet
cd control_tools/ik
python setup.py build
cd ../../
./plan_tools/pddlstream/FastDownward/build.py release64
