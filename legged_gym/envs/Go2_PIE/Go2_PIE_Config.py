from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2_PIE_Cfg_Yu( LeggedRobotCfg ):
    class env:
        # PIE: 45 proprio + 10-frame history + depth depth; critic uses clean proprio+vel+height
        frame_stack = 1
        num_envs = 4096
        num_single_obs = 45
        num_observations = int(frame_stack * num_single_obs)
        # critic: clean_obs(45) + lin_vel(3) + height_map(198) = 246 (mjlab PIE)
        num_height_map = 198
        num_foot_clearance = 4
        single_num_privileged_obs = 45 + 3 + 198  # 246
        num_privileged_obs = single_num_privileged_obs
        num_obs_hist = 10  # PIE paper / mjlab
        num_history_obs = num_obs_hist * num_single_obs
        num_actions = 12
        episode_length_s = 20
        env_spacing = 3.
        send_timeouts = True
        # PIE aux target scales
        foot_clearance_max = 0.6
        height_scan_max_distance = 5.0  # scale height target like mjlab (1/max_distance)

    class terrain:
        mesh_type = 'trimesh'
        horizontal_scale = 0.05
        vertical_scale = 0.005
        # border=50 @ hs=0.05 → ~14M verts / 28M tris → GPU PhysX segfault on upload.
        # Align MGDP gap_parkour stage2.
        border_size = 5
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        # mjlab terrain_scan: size (1.7, 1.0) @ 0.1 → inclusive 18×11 = 198
        measured_points_x = [-0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05,
                             0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 5
        terrain_length = 10.
        terrain_width = 4.
        num_rows = 10
        num_cols = 10
        terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0,
                               0.093, 0.093, 0.093, 0.093, 0.093, 0.093, 0.092,
                               0.10, 0.10,
                               0.15]
        add_roughness = True
        height = [0.01, 0.04]
        downsampled_scale = 0.2
        slope_treshold = 0.75

    class commands:
        # 与 Go2_DreamWaQ / AMP 一致的速度课程：初始 ±1.0，跟踪好后扩到 max_curriculum
        curriculum = True
        max_curriculum = 2.0
        num_commands = 4
        resampling_time = 10.
        heading_command = True
        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        stiffness = {'joint': 20.}
        damping = {'joint': 0.5}
        action_scale = 0.25
        decimation = 4

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = True
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.0
        armature_hip = 0.0044813859138
        armature_calf = 0.016520181032632315
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_restitution = True
        restitution_range = [0.0, 0.5]
        push_robots = True
        push_interval_s = 7  # mjlab 6~8s
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6
        randomize_base_mass = True
        added_base_mass_range = [-1, 2]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]
        randomize_base_com = True
        added_base_com_range = [-0.05, 0.05]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
        torque_multiplier_range = [0.9, 1.1]
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.015, 0.015]
        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
        delay = True

    class rewards:
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        class scales:
            # Align weights with parkour_mjlab PIE where counterparts exist
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -5.0
            dof_acc = -2.5e-7
            joint_power = -2.0e-5
            collision = -10.0
            action_rate = -0.01
            # action_smoothness = -0.01
            dof_pos_limits = -10.0
            hip_pos = -0.5
            base_height = -0.2  # L2 height error penalty (mjlab uses shaped +reward)
        only_positive_rewards = False
        tracking_sigma = 0.25
        base_height_target = 0.27
        max_contact_force = 120.
        clearance_height_target = -0.20

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
        contact_force_range = [0.0, 50.0]

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            # Obs layout (mjlab PIE): ang_vel, gravity, cmd, dof_pos, dof_vel, actions
            ang_vel = 0.2
            gravity = 0.05
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            height_measurements = 0.1

    class camera:
        camera_type = 'warp'
        use_camera = True
        use_lidar = False
        image_nums = 2
        crop_left = 10
        crop_right = 10
        output_hw = [60, 86]
        noise_gaussian = 0.03
        noise_dropout = 0.1
        normalize = True
        near_clip = 0.1
        far_clip = 3.0
        update_interval = 5
        width = 106
        height = 60
        horizontal_fov_deg = 87.0
        calculate_depth = True
        # mjlab PIE front camera
        offset_translation = [0.345, 0.0, 0.07]
        offset_rotation_deg = [0.0, 20.0, 0.0]
        offset_trans_rand = [[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]]
        offset_rot_rand = [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]
        lookat = [11., 5, 3.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1
        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class Go2_PIE_PPO_Yu(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'PIERunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        # PIE CNN / transformer (mjlab defaults)
        token_dim = 64
        map_latent_dim = 16
        vae_latent_dim = 16
        memory_hidden_dim = 128
        memory_num_layers = 1
        history_hidden_dims = [256, 128]
        successor_decoder_dims = [64, 128]
        height_decoder_dims = [64, 128]
        output_channels = [32, 64, 64]
        kernel_size = [8, 4, 3]
        stride = [4, 2, 1]
        padding = 0
        attention_heads = 1
        transformer_layers = 2
        transformer_ff_dim = 256
        transformer_dropout = 0.0
        cnn_activation = 'elu'

    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        # PIE aux (mjlab rl_cfg)
        auxiliary_loss_coef = 1.0
        velocity_loss_coef = 1.0
        foot_clearance_loss_coef = 1.0
        height_reconstruction_loss_coef = 1.0
        successor_loss_coef = 1.0
        kl_loss_coef = 4.0

    class runner:
        policy_class_name = "ActorCriticPIE"
        algorithm_class_name = "PPO_PIE"
        num_steps_per_env = 24
        run_name = ''
        experiment_name = 'go2_pie'
        save_interval = 50
        max_iterations = 1000000
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
