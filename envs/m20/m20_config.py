from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class M20DreamWaQCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 57
        num_obs_hist = 5
        # obs(57) + lin_vel(3) + contact_forces(17*3) + heights(17*10)
        num_privileged_obs = 298
        num_actions = 16

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        max_init_terrain_level = 20
        num_rows = 40
        num_cols = 20
        # smooth_slope, rough_slope, stairs_down, stairs_up, obstacles, highplatform
        terrain_proportions = [0.1, 0.15, 0.25, 0.25, 0.1, 0.15]

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.2
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-0.6, 0.6]
            ang_vel_yaw = [-1.0, 1.0]

        class highplatform:
            lin_vel_x_neg = [-1.0, -0.35]
            lin_vel_x_pos = [0.35, 1.0]
            lin_vel_y = [0.0, 0.0]
            heading = [0.0, 0.0]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.60]
        default_joint_angles = {
            'fl_hipx_joint': 0.0,
            'fl_hipy_joint': -0.6,
            'fl_knee_joint': 1.0,
            'fl_wheel_joint': 0.0,
            'fr_hipx_joint': 0.0,
            'fr_hipy_joint': -0.6,
            'fr_knee_joint': 1.0,
            'fr_wheel_joint': 0.0,
            'hl_hipx_joint': 0.0,
            'hl_hipy_joint': 0.6,
            'hl_knee_joint': -1.0,
            'hl_wheel_joint': 0.0,
            'hr_hipx_joint': 0.0,
            'hr_hipy_joint': 0.6,
            'hr_knee_joint': -1.0,
            'hr_wheel_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {'hipx_joint': 80., 'hipy_joint': 80., 'knee_joint': 80., 'wheel_joint': 0.}
        damping = {'hipx_joint': 2.0, 'hipy_joint': 2.0, 'knee_joint': 2.0, 'wheel_joint': 0.6}
        action_scale = 0.25
        vel_scale = 5.0
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/m20/urdf/m20.urdf'
        name = "m20"
        foot_name = "wheel"
        wheel_name = ["wheel"]
        penalize_contacts_on = ["hipx", "hipy", "knee", "base"]
        terminate_after_contacts_on = []
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
    class domain_rand(LeggedRobotCfg.domain_rand):
        push_interval_s = 10
        added_base_mass_range = [-1, 5]
        stiffness_multiplier_range = [0.85, 1.15]
        damping_multiplier_range = [0.85, 1.15]
        torque_multiplier_range = [0.85, 1.15]

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.95
        base_height_target = 0.58
        highplatform_world_speed_limit = 0.75
        wheel_dof_vel_limit = 25.0
        highplatform_stuck_cmd_threshold = 0.4
        highplatform_stuck_vel_threshold = 0.25

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            base_height = -10.0
            torques = -0.00005
            dof_vel = -1e-3
            dof_acc = -2.5e-7
            collision = -1.0
            action_rate = -0.01
            stand_still = -0.5
            dof_pos_limits = -5.0
            dof_vel_limits = -10.0
            hip_default = -0.5
            run_still = -0.05
            highplatform_yaw = -2.0
            highplatform_world_vel = -10.0


class M20DreamWaQCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        schedule = 'adaptive'

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'm20'
        save_interval = 100
        max_iterations = 300000
