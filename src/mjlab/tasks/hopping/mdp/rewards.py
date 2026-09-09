from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class joint_vel_diff_l2:
  """Penalize joint acceleration via finite difference of joint velocities.

  Equivalent to go2_jump's _reward_dof_acc: sum((last_dof_vel - dof_vel)²).
  Uses finite difference instead of MuJoCo's qacc, which can be NaN or
  extremely large during contact impacts in early training.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    joint_ids, _ = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    self.joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    self.prev_joint_vel = torch.zeros(
      env.num_envs, len(joint_ids), device=env.device, dtype=torch.float32
    )

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    curr_vel = asset.data.joint_vel[:, self.joint_ids]
    diff = curr_vel - self.prev_joint_vel
    self.prev_joint_vel = curr_vel.clone()
    return torch.sum(torch.square(diff), dim=1)


def jump_gait_sync(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  cycle_time: float,
) -> torch.Tensor:
  """Reward synchronized contact pattern matching the gait phase.

  All 4 feet must have the same contact state AND match the expected stance
  phase. Only active when the command velocity norm exceeds 0.2 m/s.
  Matches go2_jump's _reward_jump.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  # Contact: True if foot force (found) > 0 — shape [B, 4]
  contact = sensor.data.found.squeeze(-1) > 0  # [B, 4]

  # Stance mask: phase < 0.6 means all feet should be on ground
  phase = (env.episode_length_buf * env.step_dt / cycle_time) % 1.0
  stance = phase < 0.6  # [B]

  # All 4 feet must be in the same state and match the gait phase
  all_same = (
    (contact[:, 0] == contact[:, 1])
    & (contact[:, 1] == contact[:, 2])
    & (contact[:, 2] == contact[:, 3])
  )
  matches_phase = contact[:, 0] == stance
  jump_reward = (all_same & matches_phase).float()

  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd_norm = torch.norm(command[:, :2], dim=1)
  return jump_reward * (cmd_norm > 0.2).float()


class default_hip_pos:
  """Reward keeping hip joints close to zero (default) position.

  Uses exp(-4 * sum(|hip_joint_pos|)) to encourage the robot to keep
  its hips aligned. Matches go2_jump's _reward_default_hip_pos.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    hip_ids, _ = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    self.hip_ids = torch.tensor(hip_ids, device=env.device, dtype=torch.long)

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    hip_pos = asset.data.joint_pos[:, self.hip_ids]  # [B, 4]
    joint_diff = torch.sum(torch.abs(hip_pos), dim=1)
    return torch.exp(-joint_diff * 4.0)


def jump_feet_clearance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  cycle_time: float,
  command_name: str,
) -> torch.Tensor:
  """Reward foot clearance during the swing phase of the gait.

  During swing (phase >= 0.6), rewards feet that are between 0.02 m and
  0.07 m above the ground (clamped to [0, 0.05] after offset).
  Matches go2_jump's _reward_feet_clearance.
  """
  phase = (env.episode_length_buf * env.step_dt / cycle_time) % 1.0
  swing_mask = (phase >= 0.6).float()  # [B]

  asset: Entity = env.scene[asset_cfg.name]
  feet_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - 0.02  # [B, 4]
  rew_pos = torch.clamp(feet_height, 0.0, 0.05)  # [B, 4]
  # Apply swing mask to all feet (broadcast scalar to all feet)
  rew_total = torch.sum(rew_pos * swing_mask.unsqueeze(1), dim=1)  # [B]

  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd_norm = torch.norm(command[:, :2], dim=1)
  return rew_total * (cmd_norm > 0.2).float()


def lin_vel_z_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Positive reward for low vertical (z) base velocity.

  Returns exp(-|v_z|). Matches go2_jump's _reward_lin_vel_z.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return torch.exp(-torch.abs(asset.data.root_link_lin_vel_b[:, 2]))


def ang_vel_xy_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Positive reward for low roll/pitch angular velocity.

  Returns exp(-||omega_xy||). Matches go2_jump's _reward_ang_vel_xy.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return torch.exp(-torch.norm(asset.data.root_link_ang_vel_b[:, :2], dim=1))


def base_height_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Positive reward for maintaining target base height when standing.

  Returns exp(-10 * |z - target|) only when command velocity < 0.2 m/s.
  Matches go2_jump's _reward_base_height.
  """
  asset: Entity = env.scene[asset_cfg.name]
  base_height = asset.data.root_link_pos_w[:, 2]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd_norm = torch.norm(command[:, :2], dim=1)
  return (
    torch.exp(-torch.abs(base_height - target_height) * 10.0) * (cmd_norm < 0.2).float()
  )


def default_pos_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize L1 deviation of all joints from their default positions.

  Matches go2_jump's _reward_default_pos.
  """
  asset: Entity = env.scene[asset_cfg.name]
  default_pos = asset.data.default_joint_pos
  assert default_pos is not None
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default = default_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(joint_pos - default), dim=1)


class jump_air_time:
  """Reward at landing: (air_time - min_air_time) summed over feet.

  Triggers once per foot per landing event. Negative for air_time <
  min_air_time (penalizes insufficient jumps), positive for air_time >
  min_air_time. Only active when command velocity norm exceeds 0.1 m/s.
  Matches go2_jump's _reward_feet_air_time with threshold=0.5.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
    assert sensor.data.current_air_time is not None
    n_feet = sensor.data.current_air_time.shape[1]
    self.prev_air_time = torch.zeros(
      env.num_envs, n_feet, device=env.device, dtype=torch.float32
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    min_air_time: float = 0.5,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None

    # Clear stale history at episode start to prevent spurious landing events.
    # The contact sensor resets current_air_time to 0 on episode reset, but
    # prev_air_time is not reset, causing a false just_landed on the first
    # ground contact after spawn (penalty ≈ -2 per episode start).
    self.prev_air_time[env.episode_length_buf <= 1] = 0.0

    # Detect landing: was in air last step, on ground this step
    just_landed = (self.prev_air_time > 0) & (current_air_time == 0)
    reward = torch.sum((self.prev_air_time - min_air_time) * just_landed.float(), dim=1)
    self.prev_air_time = current_air_time.clone()

    command = env.command_manager.get_command(command_name)
    assert command is not None
    return reward * (torch.norm(command[:, :2], dim=1) > 0.1).float()


def body_contact_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 0.1,
) -> torch.Tensor:
  """Count of bodies with contact force exceeding force_threshold.

  Matches go2_jump's _reward_collision exactly:
    sum(||contact_force|| > 0.1 N)
  for penalize_contacts_on=["thigh", "calf"].
  """
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force  # [B, N, 3]
  assert force is not None
  return (torch.norm(force, dim=-1) > force_threshold).float().sum(dim=1)


def foot_contact_force_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_force: float = 100.0,
) -> torch.Tensor:
  """Penalize foot contact forces exceeding max_force.

  Returns sum over feet of max(0, ||force|| - max_force).
  Matches go2_jump's _reward_feet_contact_forces.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  forces = sensor.data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  return torch.sum(torch.clamp(force_magnitude - max_force, min=0.0), dim=1)
