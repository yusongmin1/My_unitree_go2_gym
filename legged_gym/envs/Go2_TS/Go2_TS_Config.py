# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import glob
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
MOTION_FILES = glob.glob('datasets/mocap_motions_go2/*')

class Go2_TS_Cfg_Yu( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 309  # =3(线速度，critic评价value用)+45(观测)+70(域随机)+4(足接触)+187(地形) 恢复
        num_terrain = 187  # terrain_obs_buf 维度（runner init_storage 需要）
        reference_state_initialization = False  # 纯 TS：不加载 AMP 动捕数据
        reference_state_initialization_prob = 0.0
        num_domain_rand = 74  # = 70(域随机参数:1摩擦+1恢复+1质量+28连杆质量比+3质心+12kp+12kd+12力矩)+4(足接触) 旧值77(已删线速度)
    class safety:
        # safety factors
        pos_limit = 0.9
        vel_limit = 1.0
        torque_limit = 0.9
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0

            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.0,#1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.0,#1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    class commands:
        curriculum = True
        max_curriculum = 1.5
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.6,0.6]   # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand:
        randomize_friction = True
        friction_range = [0.05,3.0]

        randomize_restitution = True
        restitution_range = [0.0, 0.5]

        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 1.0
        max_push_ang_vel = 1.0

        randomize_base_mass = True
        added_base_mass_range = [-1,5]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.8, 1.2]

        randomize_base_com = True
        added_base_com_range = [-0.1, 0.1]

        randomize_pd_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  
        damping_multiplier_range = [0.8, 1.2]    

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        delay = True  # HIMLoco 式 action delay：0~decimation 子步内随机切换点

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_dof_pos_limit = 0.9
        base_height_target = 0.4
        cycle_time=0.5
        target_foot_height=0.06
        class scales:
            termination = 0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            torques = -1e-5
            dof_acc = -2.5e-7
            base_height = -1.0
            # feet_air_time =  1.0
            collision = -0.5
            action_rate = -0.01
            stumble = -0.5
            # trot=0.8
            # feet_clearance=0.1 #feet clearance can increase for more
            # stand_still=-1.0
            # contact_without_command=1.
class Go2_TS_PPO_Yu( LeggedRobotCfgPPO ):
    # 纯 TS：teacher 用去 AMP 的 runner/算法（网络 ActorCriticAMP_TS 复用），蒸馏直接复用 DistillPolicyRunner
    runner_class_name = 'OnPolicyRunnerTS'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticAMP_TS'
        algorithm_class_name = 'PPO_TS'
        run_name = ''
        experiment_name = 'go2_ts'
        student_name = 'go2_ts_student'
        max_iterations = 20000 # number of policy updates

        min_normalized_std = [0.05, 0.05, 0.05] * 4

    class LSTMEncoder:
        input_size = 45
        num_steps_per_env = 24
        hidden_size = 256
        num_layers = 1
        learning_rate = 1e-3
        save_interval = 500