#!/usr/bin/env python

from distutils.core import setup
#from catkin_pkg.python_setup import generate_distutils_setup

#setup_args = generate_distutils_setup
setup(name='lis_ltamp',
      version='2.0.0',
      description='LTAMP Utilities',
      author='MIT LIS',
      author_email='caelan@mit.edu',
      url='none',
      packages=[
            'learn_tools',
            'perception_tools',
            'plan_tools',
            'control_tools',
            #'msg',
      ],
     )
#setup(**setup_args)

# python setup.py build
