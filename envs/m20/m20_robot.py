import numpy as np
import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, torch_rand_float
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import wrap_to_pi, get_scale_shift
from .m20_config import M20DreamWaQCfg


class M20Robot(LeggedRobot):
    """M20 wheeled-leg robot env for DreamWaQ.

    Extends LeggedRobot without modifying the base file. Adds hybrid leg/wheel
    control, high-platform terrain commands/rewards, and flip termination.
    """

    cfg: M20DreamWaQCfg

    def step(self, actions):
        self.disturbance_force = self.disturbance_force.to(self.device)
        n = self.num_obs
        self.obs_hist_buf = self.obs_hist_buf[:, n:]
        self.obs_hist_buf = torch.cat((self.obs_hist_buf, self.obs_buf), dim=-1)
        self.prev_privileged_obs_buf = self.privileged_obs_buf

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()

        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(
            1, self.cfg.control.decimation, 1)
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

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return (self.obs_buf, self.privileged_obs_buf, self.prev_privileged_obs_buf,
                self.obs_hist_buf, self.rew_buf, self.reset_buf, self.extras)

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        feet_z = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 2]
        self.reset_buf |= torch.all(feet_z > self.root_states[:, 2].unsqueeze(1), dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self.obs_hist_buf[env_ids] = 0.
        self.prev_privileged_obs_buf[env_ids] = 0.
        if self.cfg.terrain.mesh_type == "trimesh" and hasattr(self, "_col_terrain_types"):
            col_type_t = torch.tensor(
                [{"smooth_slope": 0, "rough_slope": 1, "stairs_down": 2,
                  "stairs_up": 3, "obstacles": 4, "highplatform": 5}[t]
                 for t in self._col_terrain_types], device=self.device)
            for tid, tname in enumerate(["smooth_slope", "rough_slope", "stairs_down",
                                         "stairs_up", "obstacles", "highplatform"]):
                env_mask = torch.isin(self.terrain_types, (col_type_t == tid).nonzero(as_tuple=False).flatten())
                if env_mask.any():
                    self.extras["episode"][f"terrain_level_{tname}"] = torch.mean(self.terrain_levels[env_mask].float())

    def compute_observations(self):
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:, self.wheel_indices] = 0
        dof_pos_obs = self.dof_pos.clone()
        dof_pos_obs[:, self.wheel_indices] = 0

        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            dof_pos_obs * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)

        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        contact_forces_scale, contact_forces_shift = get_scale_shift(self.cfg.normalization.contact_force_range)
        self.privileged_obs_buf = torch.cat((
            self.obs_buf,
            self.base_lin_vel * self.obs_scales.lin_vel,
            (self.contact_forces.view(self.num_envs, -1) - contact_forces_shift) * contact_forces_scale,
            heights,
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        n_a = self.num_actions
        # ang_vel(3), gravity(3), commands(3), dof_pos(16), dof_vel(16), actions(16)
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.
        noise_vec[9:9 + n_a] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9 + n_a:9 + 2 * n_a] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * n_a:9 + 3 * n_a] = 0.
        return noise_vec

    def _compute_torques(self, actions):
        dof_err = self.default_dof_pos - self.dof_pos
        dof_err[:, self.wheel_indices] = 0
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.wheel_indices] = 0
        vel_ref = torch.zeros_like(actions_scaled)
        vel_tmp = actions * self.cfg.control.vel_scale
        vel_ref[:, self.wheel_indices] = vel_tmp[:, self.wheel_indices]
        p_gains = self.p_gains * self.p_gains_multiplier
        d_gains = self.d_gains * self.d_gains_multiplier
        torques = p_gains * (actions_scaled + dof_err + self.motor_zero_offsets) + d_gains * (vel_ref - self.dof_vel)
        torques *= self.torques_multiplier
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            return
        hp_cfg = self.cfg.commands.highplatform
        highplatform_mask = self.terrain_types[env_ids] >= self.highplatform_col_min
        hp_env_ids = env_ids[highplatform_mask]
        if len(hp_env_ids) == 0:
            return
        neg_mask = (hp_env_ids % 2) == 0
        vx = torch.empty(len(hp_env_ids), device=self.device)
        n_neg = int(neg_mask.sum().item())
        if n_neg > 0:
            vx[neg_mask] = torch_rand_float(
                hp_cfg.lin_vel_x_neg[0], hp_cfg.lin_vel_x_neg[1], (n_neg, 1), device=self.device).squeeze(1)
        n_pos = len(hp_env_ids) - n_neg
        if n_pos > 0:
            vx[~neg_mask] = torch_rand_float(
                hp_cfg.lin_vel_x_pos[0], hp_cfg.lin_vel_x_pos[1], (n_pos, 1), device=self.device).squeeze(1)
        self.commands[hp_env_ids, 0] = vx
        self.commands[hp_env_ids, 1] = torch_rand_float(
            hp_cfg.lin_vel_y[0], hp_cfg.lin_vel_y[1], (len(hp_env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[hp_env_ids, 3] = torch_rand_float(
                hp_cfg.heading[0], hp_cfg.heading[1], (len(hp_env_ids), 1), device=self.device).squeeze(1)

    def _create_envs(self):
        super()._create_envs()
        wheel_names = []
        for name in self.cfg.asset.wheel_name:
            wheel_names.extend([s for s in self.dof_names if name in s])
        self.wheel_indices = torch.zeros(len(wheel_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(wheel_names)):
            self.wheel_indices[i] = self.gym.find_actor_dof_handle(
                self.envs[0], self.actor_handles[0], wheel_names[i])
        self._init_col_terrain_types()

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        if self.cfg.terrain.mesh_type in ['heightfield', 'trimesh']:
            terrain_props = np.cumsum(self.cfg.terrain.terrain_proportions)
            self.highplatform_col_min = int(np.ceil((terrain_props[-2] - 0.001) * self.cfg.terrain.num_cols))
        else:
            self.highplatform_col_min = self.cfg.terrain.num_cols

    def _init_col_terrain_types(self):
        if self.cfg.terrain.mesh_type != "trimesh":
            return
        props = np.cumsum(self.cfg.terrain.terrain_proportions)

        def _col_type(j):
            c = j / self.cfg.terrain.num_cols + 0.001
            if c < props[0]:
                return "smooth_slope"
            if c < props[1]:
                return "rough_slope"
            if c < props[3]:
                return "stairs_down" if c < props[2] else "stairs_up"
            if c < props[4]:
                return "obstacles"
            return "highplatform"

        self._col_terrain_types = [_col_type(j) for j in range(self.cfg.terrain.num_cols)]

    def _reward_dof_vel(self):
        vel = self.dof_vel.clone()
        vel[:, self.wheel_indices] *= 0.1
        return torch.sum(torch.square(vel), dim=1)

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        out_of_limits[:, self.wheel_indices] = 0.
        return torch.sum(out_of_limits, dim=1)

    def _reward_stand_still(self):
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:, self.wheel_indices] = 0
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_run_still(self):
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:, self.wheel_indices] = 0
        reward = torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            is_hp = self.terrain_types >= self.highplatform_col_min
            reward = reward * (~is_hp).float()
        return reward

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        reward = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            is_hp = self.terrain_types >= self.highplatform_col_min
            reward = reward * (~is_hp).float()
        return reward

    def _reward_hip_default(self):
        hip_err = torch.sum((self.dof_pos[:, [0, 4, 8, 12]] - self.default_dof_pos[:, [0, 4, 8, 12]]) ** 2, dim=1)
        return hip_err

    def _reward_highplatform_yaw(self):
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            return torch.zeros(self.num_envs, device=self.device)
        is_hp = self.terrain_types >= self.highplatform_col_min
        forward = quat_apply(self.base_quat, self.forward_vec)
        actual_heading = torch.atan2(forward[:, 1], forward[:, 0])
        heading_error = wrap_to_pi(self.commands[:, 3] - actual_heading)
        return torch.square(heading_error) * is_hp.float()

    def _reward_highplatform_world_vel(self):
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            return torch.zeros(self.num_envs, device=self.device)
        is_hp = self.terrain_types >= self.highplatform_col_min
        world_speed = torch.abs(self.root_states[:, 9])
        excess = torch.clamp(world_speed - self.cfg.rewards.highplatform_world_speed_limit, min=0.)
        return torch.square(excess) * is_hp.float()
