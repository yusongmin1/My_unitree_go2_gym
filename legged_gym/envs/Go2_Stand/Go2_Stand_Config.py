from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2_Stand_Cfg_Yu( LeggedRobotCfg ):

    class env:
        frame_stack = 1 #action stack
        c_frame_stack = 1 #critic 网络的堆叠帧数
        num_single_obs = 48 # =3(ang)+3(grav)+3(cmd)+12+12+12+2(phase)+1(stand_command)
        num_observations = int(frame_stack * num_single_obs)
        # privileged: obs(48)+lin_vel(3)+domain(~34)+contact_mask(4) = 89
        single_num_privileged_obs = 100
        num_privileged_obs = single_num_privileged_obs
        num_obs_hist = 5
        num_latent_dims = 16
        num_explicit_dims = 3  # estimated base lin_vel
        num_history_obs = num_obs_hist * num_single_obs
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        num_envs=4096
        test = False

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False# select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # [smooth slope, rough slope, stairs up, stairs down, discrete,
        #  single-gap, step-stone, stones-2rows, stones-1row, bridge, hurdle, ramp,
        #  pure_rough(±3cm)]  remainder → corridor
        terrain_proportions = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.0]
        rough_height = 0.03  # 最后一列 pure_rough: ±3cm
        add_roughness = False
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        # 四足 stand_command=0
        class ranges_quad:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.8, 0.8]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]
        # 双足 handstand/legstand stand_command=±1
        class ranges_hand:
            lin_vel_x = [-0.4, 0.4]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-3.14, 3.14]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': -0.0 ,  # [rad]
            'RR_hip_joint': -0.0,   # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 0.8,#1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 0.8,#1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 28.}  # [N*m/rad]
        damping = {'joint': 0.7}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        front_feet=['FL_foot', 'FR_foot']  # legstand 支撑腿（后腿腾空）
        rear_feet=['RL_foot', 'RR_foot']   # handstand 支撑腿（前腿腾空）
        threshold=5.0
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
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
        armature = 0.
        thickness = 0.01
    class domain_rand:
        randomize_friction = True
        friction_range = [0.2,1.2]

        randomize_restitution = True
        restitution_range = [0.0, 1.0]

        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_base_mass = True
        added_base_mass_range = [-1,2]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.3, 0.03]

        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]  
        damping_multiplier_range = [0.9, 1.1]
        torque_multiplier_range = [0.8, 1.2]

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        delay = True  # HIMLoco 式 action delay：0~decimation 子步内随机切换点

        randomize_joint_friction = True
        joint_friction_range = [0.01, 0.1]

        randomize_joint_damping = True
        joint_damping_range = [0.0,0.1]

        randomize_joint_armature = True
        joint_armature_range = [0.003, 0.006]    #
    class rewards:
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95

        class scales:
            termination = -200.0
            tracking_lin_vel = 2.25
            tracking_ang_vel = 1.0
            # lin_vel_z = 0.2
            # ang_vel_xy = 0.2
            torques = -0.0002
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            action_rate = -0.025
            collision = -1.
            dof_pos_limits=-2.0



            body_orientation_quad = 0.9
            body_orientation_handstand = 0.9
            body_orientation_legstand = 0.9
            base_height_quad = 1.
            base_height_handstand = 1.
            base_height_legstand = 1.
            default_hip_pos=-0.5
            default_pos_reward=0.2

            gait_quad = 0.5        # stand_command=0
            gait_handstand = 0.5   # stand_command=-1 后腿支撑（前腿空中）
            gait_legstand = 0.5    # stand_command=1 前腿支撑（后腿空中）
            rear_contact_hand =-1.0  # 惩罚腾空腿触地（handstand 前腿 / legstand 后腿）
            foot_slip = -0.1
            foot_clearance_quad = 0.25
            foot_clearance_handstand = 0.25
            foot_clearance_legstand = 0.25


        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        base_height_target_quad = 0.33
        base_height_target_handstand = 0.5  # 双足
        base_height_target_legstand = 0.5   # 双足
        base_height_sigma = 0.05
        body_pitch_target_quad = 0.0
        body_orientation_sigma_quad = 0.04
        body_pitch_target_handstand = -1.466   # ≈84°，机头朝上
        body_orientation_sigma_handstand = 1.2
        body_pitch_target_legstand = 1.466  # ≈-84°，机头朝下
        body_orientation_sigma_legstand = 1.2
        target_foot_height=0.06
        foot_clearance_target = 0.05
        foot_height_offset = 0.02
        foot_clearance_tracking_sigma = 0.01
        max_contact_force = 200. # forces above this value are penalized
        cycle_time=0.8
        duty_factor = 0.6  # leg_phase∈[0,duty) 支撑，[duty,1) 摆动
        gait_offsets = [0.0, 0.5, 0.5, 0.0]  # FL, FR, RL, RR 对角同相


    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
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


class Go2_Stand_PPO_Yu(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'DreamWaQRunner'
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
        max_grad_norm = 1.
        vae_kl_weight = 1.0
        sym_loss = True
        # 48 维 obs 镜像置换表（与 compute_observations 布局严格对齐）：
        # [0:3]角速度: 轴向量 (-wx, wy, -wz) | [3:6]重力投影: 极向量 (gx, -gy, gz)
        # [6:9]命令: (vx, -vy, -wyaw) | [9:21/21:33/33:45] dof_pos/vel/action: FL<->FR, RL<->RR, hip 取反
        # [45:47]相位 sin/cos 不变 | [47] stand_command 不变
        obs_permutation = [-0.0001, 1, -2,            # ang_vel
                           3, -4, 5,                   # gravity (gy 取反)
                           6, -7, -8,                  # commands
                           -12, 13, 14,                # dof_pos FL <- FR(hip取反)
                           -9, 10, 11,                 # dof_pos FR <- FL
                           -18, 19, 20,                # dof_pos RL <- RR
                           -15, 16, 17,                # dof_pos RR <- RL
                           -24, 25, 26,                # dof_vel FL <- FR
                           -21, 22, 23,                # dof_vel FR <- FL
                           -30, 31, 32,                # dof_vel RL <- RR
                           -27, 28, 29,                # dof_vel RR <- RL
                           -36, 37, 38,                # actions FL <- FR
                           -33, 34, 35,                # actions FR <- FL
                           -42, 43, 44,                # actions RL <- RR
                           -39, 40, 41,                # actions RR <- RL
                           -45, -46,                     # phase sin/cos
                           47]                         # stand_command
        act_permutation = [-3, 4, 5,-0.0001, 1, 2 , -9, 10, 11,-6, 7, 8,]#关节电机的对陈关系
        frame_stack = 5  # obs_hist = 5 * 48，与 num_obs_hist 一致
        sym_coef = 1.0
    class runner:
        policy_class_name = 'ActorCriticDreamWaQ'
        algorithm_class_name = 'PPO_DreamWaQ'
        num_steps_per_env = 24 # per iteration
        max_iterations = 15000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'go2_stand'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt