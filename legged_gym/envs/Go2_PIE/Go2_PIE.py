import os
import numpy as np

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from collections import deque

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.Go2_PIE.Go2_PIE_Config import Go2_PIE_Cfg_Yu

# NOTE: do NOT import warp_sensor at module import time.
# `import warp` claims a CUDA context before Isaac Gym create_sim and causes
# GPU PhysX segfaults. Lazy-import inside _init_camera_buffers() after sim exists.

class Go2_PIE_Robot(BaseTask):
    def __init__(self, cfg: Go2_PIE_Cfg_Yu, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        # ===== MGDP warp 深度相机：读取噪声/归一化参数，分配图像缓冲 =====
        self._init_camera_buffers()
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs / PIE targets
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.clean_obs_buf = torch.clip(self.clean_obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # PIE API: obs, privileged, obs_hist, depth, pie_targets, rew, reset, extras
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.obs_hist_buf,
            self.depth_flat_buf,
            self._pie_targets(),
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def get_observations(self):
        return self.obs_buf, self.obs_hist_buf, self.depth_flat_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf, self._pie_targets()

    def _pie_targets(self):
        return {
            "velocity": self.velocity_target_buf,
            "height": self.height_target_buf,
            "foot": self.foot_target_buf,
            # successor for the *previous* transition is this step's clean proprio
            "successor": self.clean_obs_buf,
        }

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, obs_history, depth, pie_targets, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        )
        return obs, privileged_obs, obs_history, depth, pie_targets
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        # MGDP：policy step 计数器（驱动 update_interval 深度刷新）
        self.global_counter += 1

    def check_termination(self):
        """ Check if environments need to be reset.
        对齐 MGDP gap_parkour：跑酷地形上基座/足端掉进坑（z < -0.1）即终止。
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        # 掉坑终止：仅跑酷类（idx 5~13，含 gap/石阵/桥/栏/坡/上下楼梯）
        if hasattr(self, "env_class"):
            parkour = (self.env_class >= 5) & (self.env_class <= 13)
            base_fall = self.root_states[:, 2] < -0.1
            foot_fall = torch.min(self.feet_pos[:, :, 2], dim=1).values < -0.2
            self.reset_buf |= parkour & (base_fall | foot_fall)

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        if self.cfg.domain_rand.randomize_pd_gains:
            self.p_gains_multiplier[env_ids, :] = torch_rand_float(self.cfg.domain_rand.stiffness_multiplier_range[0], self.cfg.domain_rand.stiffness_multiplier_range[1], (len(env_ids), self.num_actions), device=self.device)
            self.d_gains_multiplier[env_ids, :] = torch_rand_float(self.cfg.domain_rand.damping_multiplier_range[0], self.cfg.domain_rand.damping_multiplier_range[1], (len(env_ids), self.num_actions), device=self.device)   
            self.torques_multiplier[env_ids, :] = torch_rand_float(self.cfg.domain_rand.torque_multiplier_range[0], self.cfg.domain_rand.torque_multiplier_range[1], (len(env_ids), self.num_actions), device=self.device)   
               
        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.mesh_type == "trimesh":
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            # Actual parkour class ids from terrain.make_terrain (env_class), not legacy slope labels.
            # 5 gap, 6 stones, 7 stones2, 8 stones1, 9 bridge, 10 hurdle,
            # 11 ramp, 12 half-stairs-0.25, 13 half-stairs-0.30, 14 rough
            _tl = self.terrain_levels.float()
            _cls = self.env_class if hasattr(self, "env_class") else None
            _names = {
                5: "gap",
                6: "stones",
                7: "stones2",
                8: "stones1",
                9: "bridge",
                10: "hurdle",
                11: "ramp",
                12: "half_stairs_025",
                13: "half_stairs_030",
                14: "rough",
            }
            if _cls is not None:
                for tid, name in _names.items():
                    mask = _cls == tid
                    self.extras["episode"][f"terrain_level_{name}"] = torch.nan_to_num(
                        _tl[mask].mean()
                    )
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]  # 角速度课程上界
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        #这里要把全局坐标系下的速度和角速度都转换到局部坐标系下，控制前进是控制向基座坐标系下的前进方向，而不是全局坐标系下的前进方向
        # 重置 env 的 history / PIE buffers
        self.obs_hist_buf[env_ids] = 0
        self.clean_obs_buf[env_ids] = 0
        self.velocity_target_buf[env_ids] = 0
        self.height_target_buf[env_ids] = 0
        self.foot_target_buf[env_ids] = 0
        # if self.use_camera and self.image_buf is not None:
        #     self.image_buf[env_ids] = 0
        #     self.depth_flat_buf[env_ids] = 0

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    def compute_observations(self):
        # History stacks previous *noisy* actor proprio (exclude current), mjlab-style.
        self.obs_hist_buf = torch.cat(
            (self.obs_hist_buf[:, self.num_obs :], self.obs_buf), dim=-1
        )

        # mjlab PIE proprio order: ang_vel, gravity, command, joint_pos, joint_vel, actions
        clean = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        self.clean_obs_buf = clean

        if self.add_noise:
            self.obs_buf = (
                clean
                + (2 * torch.rand_like(clean) - 1)
                * self.noise_scale_vec
                * self.cfg.noise.noise_level
            )
        else:
            self.obs_buf = clean.clone()

        # Height scan target / critic: (base_z - 0.5 - measured) then scale like mjlab 1/max_distance
        # Keep local clip for numerical stability; decoder target uses scan_scale.
        heights_raw = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0
        )
        scan_scale = 1.0 / float(self.cfg.env.height_scan_max_distance)
        self.height_target_buf = heights_raw * scan_scale

        # Critic: clean proprio + lin_vel + height scan (mjlab PIE critic group)
        self.privileged_obs_buf = torch.cat(
            (
                self.clean_obs_buf,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.height_target_buf,
            ),
            dim=-1,
        )

        # Velocity / foot-clearance supervision targets
        self.velocity_target_buf = self.base_lin_vel.clone()
        self.foot_target_buf = self._compute_foot_clearance_target()

        # Depth refresh at update_interval (default 5 @ 50Hz -> 10Hz)
        # global_counter is advanced once per policy step in post_physics_step
        if self.use_camera:
            if self.global_counter % self.cfg.camera.update_interval == 0:
                self.get_camera_sensor()
            self.depth_flat_buf = self.image_buf.reshape(self.num_envs, -1)
        else:
            self.depth_flat_buf.zero_()

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain)
            n_verts = len(self.terrain.vertices)
            n_tris = len(self.terrain.triangles)
            print(f"[Go2_PIE] terrain mesh: verts={n_verts}, tris={n_tris}", flush=True)
            if n_tris > 8_000_000:
                raise RuntimeError(
                    f"Terrain mesh too large for GPU PhysX ({n_tris} tris). "
                    "Reduce terrain.border_size / num_cols or increase horizontal_scale "
                    "(MGDP gap_parkour uses border_size=5, num_cols=10)."
                )
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()
    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs, 1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        
        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare friction randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs, 1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.soft_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits：软区间另存，dof_pos_limits 保持 URDF 原始值（供 min_std 等使用）
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.soft_dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.soft_dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        # mjlab Go2 分关节 armature：hip/thigh vs calf
        for i in range(len(props)):
            if "calf" in self.dof_names[i]:
                props["armature"][i] = self.cfg.asset.armature_calf
            else:
                props["armature"][i] = self.cfg.asset.armature_hip

        return props
    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            self.added_base_masses = torch_rand_float(self.cfg.domain_rand.added_base_mass_range[0], self.cfg.domain_rand.added_base_mass_range[1], (1, 1), device=self.device)
            props[0].mass += self.added_base_masses

        # randomize link masses
        if self.cfg.domain_rand.randomize_link_mass:
            self.multiplied_link_masses_ratio = torch_rand_float(self.cfg.domain_rand.multiplied_link_mass_range[0], self.cfg.domain_rand.multiplied_link_mass_range[1], (1, self.num_bodies-1), device=self.device)
    
            for i in range(1, len(props)):
                props[i].mass *= self.multiplied_link_masses_ratio[0,i-1]

        # randomize base com
        if self.cfg.domain_rand.randomize_base_com:
            self.added_base_com = torch_rand_float(self.cfg.domain_rand.added_base_com_range[0], self.cfg.domain_rand.added_base_com_range[1], (1, 3), device=self.device)
            props[0].com += gymapi.Vec3(self.added_base_com[0, 0], self.added_base_com[0, 1],self.added_base_com[0, 2])
        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            # yaw rate tracks heading target; when heading matches commands[:,3], ang_vel≈0
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading),
                self.command_ranges["ang_vel_yaw"][0],
                self.command_ranges["ang_vel_yaw"][1],
            )
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _resample_commands(self, env_ids):
        """All terrains: vx∈[0,1], vy=0, heading=0; yaw rate from heading error in callback."""
        if len(env_ids) == 0:
            return

        vx_lo = float(self.command_ranges["lin_vel_x"][0])
        vx_hi = float(self.command_ranges["lin_vel_x"][1])
        self.commands[env_ids, 0] = torch_rand_float(
            vx_lo, vx_hi, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 1] = 0.0
        self.commands[env_ids, 2] = 0.0
        self.commands[env_ids, 3] = 0.0
        self.commands[env_ids, 0] *= (self.commands[env_ids, 0] > 0.1).float()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        actions_scaled = actions * self.cfg.control.action_scale
        p_gains = self.p_gains * self.p_gains_multiplier
        d_gains = self.d_gains * self.d_gains_multiplier

        torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains * self.dof_vel
        torques*=self.torques_multiplier
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # 跑酷走廊仅宽 4m：对齐 MGDP，xy 扰动 ±0.2m（原先 ±1m 会把机器人甩进坑）
            if hasattr(self, "env_class"):
                parkour = (self.env_class[env_ids] >= 5) & (self.env_class[env_ids] <= 13)
                parkour_ids = env_ids[parkour]
                other_ids = env_ids[~parkour]
                if len(parkour_ids) > 0:
                    self.root_states[parkour_ids, :2] += torch_rand_float(-0.1, 0.1, (len(parkour_ids), 2), device=self.device)
                if len(other_ids) > 0:
                    self.root_states[other_ids, :2] += torch_rand_float(-0.1, 0.1, (len(other_ids), 2), device=self.device)
            else:
                self.root_states[env_ids, :2] += torch_rand_float(-0.2, 0.2, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.root_states[:, 10:13] = torch_rand_float(-max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device) # ang vel
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        if hasattr(self, "terrain_class"):
            self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # 角速度课程：tracking_ang_vel 奖励 > 85% 满分时扩大角速度范围，初始 [-1,1]，最大扩到 [-2,2]
        if torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length > 0.85 * self.reward_scales["tracking_ang_vel"]:
            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.2, -2.0, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.2, 0., 2.0)
        low_vel_env_ids = (env_ids > (self.num_envs * 0.2))
        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        # Keep lin_vel_x in [0, max_curriculum] (default [0, 1]); do not expand sideways/backward
        if (torch.mean(self.episode_sums["tracking_lin_vel"][low_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]) and (torch.mean(self.episode_sums["tracking_lin_vel"][high_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]):
            self.command_ranges["lin_vel_x"][0] = 0.0
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.0, 0.0, self.cfg.commands.max_curriculum
            )

    # ==================== MGDP warp 深度相机（迁移） ====================

    def _init_camera_buffers(self):
        """初始化图像缓冲 + WarpManager。
        必须在 create_sim()（terrain 已生成）之后、_init_buffers() 之前调用。
        """
        self.use_camera = self.cfg.camera.use_camera
        self.use_lidar = self.cfg.camera.use_lidar
        self.image_nums = self.cfg.camera.image_nums
        self.crop_left = int(self.cfg.camera.crop_left)
        self.crop_right = int(self.cfg.camera.crop_right)
        # 输出尺寸：高不变，宽 = width - crop_left - crop_right
        self.output_hw = list(self.cfg.camera.output_hw)  # [H, W]
        self.noise_gaussian = self.cfg.camera.noise_gaussian
        self.noise_dropout = self.cfg.camera.noise_dropout
        self.normalize = self.cfg.camera.normalize
        self.global_counter = 0

        if not self.use_camera:
            self.image_buf = None
            return

        # Lazy import: after Isaac Gym sim exists (avoids CUDA context clash / segfault)
        from legged_gym.warp_sensor import WarpManager
        from legged_gym.warp_sensor.config.sensor_config import (
            Config as WarpConfig,
            Camera as WarpCameraCfg,
            ImageProcess as WarpImageProcess,
            TransRand as WarpTransRand,
            RotRand as WarpRotRand,
        )

        assert self.output_hw[0] == self.cfg.camera.height
        assert self.output_hw[1] == self.cfg.camera.width - self.crop_left - self.crop_right

        if self.image_nums is None:
            buf_shape = (self.num_envs, 1, self.output_hw[0], self.output_hw[1])
        else:
            buf_shape = (self.num_envs, self.image_nums, self.output_hw[0], self.output_hw[1])
        self.image_buf = torch.zeros(*buf_shape, device=self.device, dtype=torch.float)

        cam_cfg = self.cfg.camera
        near = float(cam_cfg.near_clip)
        far = float(cam_cfg.far_clip)
        warp_cam = WarpCameraCfg()
        warp_cam.pattern.height = cam_cfg.height
        warp_cam.pattern.width = cam_cfg.width
        warp_cam.pattern.horizontal_fov_deg = cam_cfg.horizontal_fov_deg
        warp_cam.pattern.max_range = far
        warp_cam.pattern.min_range = near
        warp_cam.pattern.calculate_depth = cam_cfg.calculate_depth
        warp_cam.offset.translation = cam_cfg.offset_translation
        warp_cam.offset.rotation = cam_cfg.offset_rotation_deg
        warp_cam.offset.trans_rand = WarpTransRand(
            min=list(cam_cfg.offset_trans_rand[0]), max=list(cam_cfg.offset_trans_rand[1]))
        warp_cam.offset.rot_rand = WarpRotRand(
            min=list(cam_cfg.offset_rot_rand[0]), max=list(cam_cfg.offset_rot_rand[1]))
        # 不做 resize：出图保持 60×106，crop 在 get_camera_sensor 里做
        warp_cam.process = WarpImageProcess(
            resize=None,
            noise=None,
            clip=None,
            normalize=False,
        )

        warp_cfg = WarpConfig()
        warp_cfg.camera = warp_cam
        warp_cfg.lidar = None

        self.terrain.vertices -= np.array(
            [self.terrain.cfg.border_size, self.terrain.cfg.border_size, 0])
        if 'cuda' in str(self.device):
            warp_device = self.device
        elif torch.cuda.is_available():
            warp_device = 'cuda:0'
        else:
            warp_device = 'cpu'
        self.warp_manager = WarpManager(
            self.num_envs, self, cfg=warp_cfg, device=warp_device, enable_viz=False)
        self.terrain.vertices += np.array(
            [self.terrain.cfg.border_size, self.terrain.cfg.border_size, 0])

    def get_camera_sensor(self):
        """深度图流水线：
        warp 渲染(miss→far) → 左右 crop → 加噪 → clamp/far 归一化 → 入栈 image_buf。
        """
        self.warp_manager.warp_update_frame()
        cur_images = self.warp_manager['camera'].data.squeeze(1).to(self.device)  # (N,H,W)

        # parkour_mjlab：左右各裁 crop_left / crop_right（无双线性）
        w = cur_images.shape[-1]
        cur_images = cur_images[..., self.crop_left : w - self.crop_right]

        if self.noise_gaussian is not None:
            cur_images = cur_images + torch.randn_like(cur_images) * self.noise_gaussian
        if self.noise_dropout is not None:
            cur_images = cur_images * (torch.rand_like(cur_images) > self.noise_dropout)

        if self.normalize:
            near = float(self.cfg.camera.near_clip)
            far = float(self.cfg.camera.far_clip)
            if far < 1e-6:
                far = 1e-6
            cur_images = (cur_images.clamp(near, far) / far).clamp(0.0, 1.0)

        if self.image_nums:
            self.image_buf = torch.cat(
                [self.image_buf[:, 1:], cur_images.unsqueeze(1)], dim=1)
        else:
            self.image_buf = cur_images

        if self.image_nums is not None and self.image_nums > 1:
            at_reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(at_reset_env_ids) > 0:
                self.image_buf[at_reset_env_ids, 0] = cur_images[at_reset_env_ids]
                self.image_buf[at_reset_env_ids, 1] = cur_images[at_reset_env_ids]

    def _get_noise_scale_vec(self):
        """Noise scales for mjlab PIE proprio layout:
        [ang_vel(3), gravity(3), command(3), dof_pos(12), dof_vel(12), actions(12)].
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity
        noise_vec[6:9] = 0.0  # commands
        noise_vec[9:21] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.0  # previous actions
        return noise_vec

    def _compute_foot_clearance_target(self):
        """Foot-to-terrain clearance in [0, 1], scaled by 1/max_clearance (mjlab PIE)."""
        max_c = float(self.cfg.env.foot_clearance_max)
        if self.cfg.terrain.mesh_type == 'plane':
            terrain_z = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        else:
            points = self.feet_pos.clone()
            points[:, :, 0] += self.terrain.cfg.border_size
            points[:, :, 1] += self.terrain.cfg.border_size
            px = (points[:, :, 0] / self.terrain.cfg.horizontal_scale).long()
            py = (points[:, :, 1] / self.terrain.cfg.horizontal_scale).long()
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
            h1 = self.height_samples[px, py]
            h2 = self.height_samples[px + 1, py]
            h3 = self.height_samples[px, py + 1]
            terrain_z = torch.min(torch.min(h1, h2), h3) * self.terrain.cfg.vertical_scale
        clearance = (self.feet_pos[:, :, 2] - terrain_z).clamp(0.0, max_c) / max_c
        return clearance

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            self.base_height_points = self._init_base_height_points()
        self.measured_heights = self._get_heights()

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            self.default_dof_pos[i] = self.cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        #通信延迟 cmd延迟，obs延迟 ，imu延迟
        # HIMLoco 式 action delay
        self.delayed_actions = torch.zeros(self.num_envs, self.cfg.control.decimation, self.num_actions, device=self.device)
        self.obs_motor_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) 
        self.obs_imu_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  
        self.obs_hist_buf = torch.zeros(
            self.num_envs, self.cfg.env.num_history_obs, dtype=torch.float, device=self.device
        )
        self.clean_obs_buf = torch.zeros(
            self.num_envs, self.num_obs, dtype=torch.float, device=self.device
        )
        self.velocity_target_buf = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.height_target_buf = torch.zeros(
            self.num_envs, self.cfg.env.num_height_map, dtype=torch.float, device=self.device
        )
        self.foot_target_buf = torch.zeros(
            self.num_envs, self.cfg.env.num_foot_clearance, dtype=torch.float, device=self.device
        )
        depth_flat = (
            int(self.cfg.camera.image_nums or 1)
            * int(self.cfg.camera.output_hw[0])
            * int(self.cfg.camera.output_hw[1])
        )
        self.depth_flat_buf = torch.zeros(
            self.num_envs, depth_flat, dtype=torch.float, device=self.device
        )
        self.disturbance = torch.zeros(
            self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        print(f"body_names: {body_names}")
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        print(f"dof_names: {self.dof_names}")
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        self.torques_multiplier = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_zero_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) 
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains_multiplier = torch.ones(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains_multiplier = torch.ones(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        # 髋关节（hip）自由度索引，用于 hip_limit 奖励
        self.hip_indices = torch.tensor([i for i, name in enumerate(self.dof_names) if 'hip' in name],
                                        dtype=torch.long, device=self.device)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            if hasattr(self.terrain, "terrain_type"):
                self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.long)
                self.env_class = self.terrain_class[self.terrain_levels, self.terrain_types]
            else:
                self.env_class = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height

    # ------------ reward functions----------------

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # 对齐 DreamWaQ：偏离站立高度 + 仅在大致直立时计罚（鼓励站起来）
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_power(self):
        # Penalize power consumption
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        '''Penalize action smoothness'''
        action_smoothness_cost = torch.sum(torch.square(self.actions - 2*self.last_actions + self.last_last_actions), dim=-1)
        return action_smoothness_cost

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.soft_dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.dof_pos - self.soft_dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma) 

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma) 

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)*(torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_default_pos(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) 
    
    def _reward_stumble(self):
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
     

    def _reward_hip_pos(self):
        """ Reward for the hip joint position close to default position
        """
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(torch.square(
            self.dof_pos[:, hip_joint_indices] - 
            self.default_dof_pos[:, hip_joint_indices]), dim=-1)
        return dof_pos_error
    
    def _reward_power_distribution(self):
        power = self.torques*self.dof_vel
        return torch.var(torch.abs(power),dim=1)
    
    def _reward_joint_power(self):
        return torch.sum((torch.abs(self.dof_vel)*torch.abs(self.torques)),dim=1)
    
