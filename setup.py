from setuptools import find_packages
from distutils.core import setup

setup(name='my_unitree_go2_gym',
      version='1.0.0',
      author='Unitree Robotics',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='1902219511@qq.com',
      description='my_unitree_go2_gym',
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy==1.24.4', 'tensorboard', 'mujoco==3.2.3', 'pyyaml'])
