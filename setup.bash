#!/usr/bin/env bash

git pull --recurse-submodules
git submodule update --init --recursive
pip install -r requirements.txt
cd control_tools/ik
python setup.py build
cd ../../
./pddlstream/FastDownward/build.py release64
