#!/usr/bin/env bash

# download this bash file and run 'source start.bash' in a folder where you would like to install LTAMP
#sudo apt install git python-pip cmake
#git clone --recursive git@github.com:caelan/LTAMP.git
#cd LTAMP
git submodule update --init --recursive
pip install -r requirements.txt
cd control_tools/ik
python setup.py build
cd ../../
./pddlstream/FastDownward/build.py release64
