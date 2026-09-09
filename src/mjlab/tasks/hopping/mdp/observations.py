from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def gait_phase_encoding(env: ManagerBasedRlEnv, cycle_time: float) -> torch.Tensor:
  """Sine and cosine encoding of the current gait phase.

  Returns shape [B, 2]: [sin(2π·phase), cos(2π·phase)].
  Phase is computed from episode step count and resets each episode.
  """
  phase = (env.episode_length_buf * env.step_dt / cycle_time) % 1.0
  return torch.stack(
    [torch.sin(2 * math.pi * phase), torch.cos(2 * math.pi * phase)], dim=1
  )


def euler_xyz(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Root orientation as roll, pitch, yaw Euler angles (XYZ convention).

  Returns shape [B, 3]. Matches the observation used in go2_jump.

  Quaternion is normalized before conversion to ensure sin_pitch stays in
  [-1, 1] on CUDA. Floating-point drift in the quaternion norm can cause
  |sin_pitch| > 1, making torch.asin produce NaN even with torch.where
  guards (both branches are evaluated on GPU hardware).
  """
  asset: Entity = env.scene[asset_cfg.name]
  quat_w = asset.data.root_link_quat_w  # [B, 4] wxyz
  quat_w = F.normalize(quat_w, dim=1)
  roll, pitch, yaw = euler_xyz_from_quat(quat_w)
  return torch.nan_to_num(torch.stack([roll, pitch, yaw], dim=1), nan=0.0)


def gait_stance_mask(env: ManagerBasedRlEnv, cycle_time: float) -> torch.Tensor:
  """Gait stance mask based on phase.

  Returns shape [B, 2]:
    - [:, 0] = 1 when phase < 0.6 (all feet stance / ground contact expected)
    - [:, 1] = 1 when phase >= 0.6 (all feet swing / air expected)
  """
  phase = (env.episode_length_buf * env.step_dt / cycle_time) % 1.0
  return torch.stack([(phase < 0.6).float(), (phase >= 0.6).float()], dim=1)


def joint_pos_abs(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute joint positions (not relative to default).

  Returns shape [B, N].
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids]


def env_friction(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Sliding friction coefficient for the first geom in asset_cfg.

  With shared_random=True, all foot geoms share one value per env.
  Returns shape [B, 1]. Matches go2_jump's env_frictions privileged obs.
  """
  asset: Entity = env.scene[asset_cfg.name]
  geom_id = asset.indexing.geom_ids[asset_cfg.geom_ids][0]
  return env.sim.model.geom_friction[:, geom_id, 0].unsqueeze(-1)


def robot_total_mass(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Total robot body mass divided by 10.

  Returns shape [B, 1]. Matches go2_jump's body_mass / 10. privileged obs.
  """
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
  return env.sim.model.body_mass[:, body_ids].sum(dim=-1, keepdim=True) / 10.0
