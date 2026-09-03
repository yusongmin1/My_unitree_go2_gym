from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2_Cts_Cfg_Yu( LeggedRobotCfg ):
    class env:
        num_envs = 8192
        num_observations = 45
        num_privileged_obs = 233
        num_critic_obs=281
        num_obs_hist=5 #  10帧正常的观测

        num_history_obs=num_obs_hist*num_observations
        num_actions = 12
        episode_length_s = 20 # episode length in seconds
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts=True


    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 70 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False# select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.20,0.15, 0.275, 0.275, 0.1]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
    class commands:
        curriculum = True
        max_curriculum = 2.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0,1.0] # min max [m/s]
            lin_vel_y = [-1.0,1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

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
    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf","base"]
        terminate_after_contacts_on =["base"]
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.00448  # 电机转子惯量（折算到关节的等效惯量）
        thickness = 0.01
    class domain_rand:
        randomize_friction = True
        friction_range = [0.2,1.25]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]

        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_base_mass = True
        added_base_mass_range = [-1,2]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.05, 0.05]

        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]  
        damping_multiplier_range = [0.9, 1.1]    
        torque_multiplier_range=[0.8,1.2] 


        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        delay = True  # HIMLoco 式 action delay：0~decimation 子步内随机切换点

    class rewards:
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.2
            base_height=-5.0
            torques = -0.0001#
            dof_acc = -2.5e-7#-7
            collision = -1.
            action_rate = -0.01
            # feet_air_time =  1.0
            # stand_still=-0.5
            dof_pos_limits=-2.0
            action_smoothness = -0.01
            stumble = -0.5
            foot_clearance=-0.5
            hip_pos=-0.05
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        base_height_target = 0.38
        max_contact_force = 120. # forces above this value are penalized
        clearance_height_target = -0.20
    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            quat = 1.
        clip_observations = 100.
        clip_actions = 100.

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

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)


class Go2_Cts_PPO_Yu(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunnerCTS'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01

        # ===== 镜像对称损失（默认关闭；置 True 启用，置换表与 obs 布局逐位对齐）=====
        sym_loss = False
        sym_coef = 1.0
        # CTS 特权输入(233=[42域随机|4接触|187地形])与地形(187)的镜像表
        privileged_permutation = [0, 1, 2, 3, -4, 5, 9, 10, 11, 6, 7, 8, 15, 16, 17, 12, 13, 14, 21, 22, 23, 18, 19, 20, 27, 28, 29, 24, 25, 26, 33, 34, 35, 30, 31, 32, 39, 40, 41, 36, 37, 38, 43, 42, 45, 44, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58, 57, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 133, 132, 131, 130, 129, 128, 127, 126, 125, 124, 123, 144, 143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 177, 176, 175, 174, 173, 172, 171, 170, 169, 168, 167, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 199, 198, 197, 196, 195, 194, 193, 192, 191, 190, 189, 210, 209, 208, 207, 206, 205, 204, 203, 202, 201, 200, 221, 220, 219, 218, 217, 216, 215, 214, 213, 212, 211, 232, 231, 230, 229, 228, 227, 226, 225, 224, 223, 222]
        terrain_permutation = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 131, 130, 129, 128, 127, 126, 125, 124, 123, 122, 121, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176]
        obs_permutation = [0.0001, -1, 2, -3, 4, -5, 6, -7, 8, -12, 13, 14, -9, 10, 11, -18, 19, 20, -15, 16, 17, -24, 25, 26, -21, 22, 23, -30, 31, 32, -27, 28, 29, -36, 37, 38, -33, 34, 35, -42, 43, 44, -39, 40, 41]
        act_permutation = [-3, 4, 5, -0.0001, 1, 2, -9, 10, 11, -6, 7, 8]

        max_grad_norm = 1.
    class runner:
        policy_class_name = 'ActorCriticCTS'
        algorithm_class_name = 'CTS'
        num_steps_per_env = 24 # per iteration
        max_iterations = 20000 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'go2_cts'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt