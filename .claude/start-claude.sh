#!/bin/bash

# This script is used to start the Claude agent with the necessary environment variables and additional directories.
export CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD=1
claude --add-dir /home/rkz/code/unitree_ws/src/unitree_mujoco --add-dir /home/rkz/code/Isaacgym/src/My_unitree_go2_gym
