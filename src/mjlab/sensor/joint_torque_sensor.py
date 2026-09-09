"""Joint torque sensor with substep mean/last aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.entity import Entity
from mjlab.sensor.sensor import Sensor, SensorCfg


@dataclass
class JointTorqueSensorCfg(SensorCfg):
  """Sensor configuration for joint torque sensing."""

  entity_name: str
  window_length: int = 1

  def build(self) -> JointTorqueSensor:
    return JointTorqueSensor(self)


@dataclass
class JointTorqueSensorData:
  """Joint torque outputs."""

  joint_torque_last: torch.Tensor
  joint_torque_mean: torch.Tensor
  joint_torque_limit: torch.Tensor


class JointTorqueSensor(Sensor[JointTorqueSensorData]):
  """Reads actuator force and exposes joint torque last/mean and torque limits."""

  def __init__(self, cfg: JointTorqueSensorCfg) -> None:
    super().__init__()
    self.cfg = cfg
    self._entity: Entity | None = None
    self._model: mjwarp.Model | None = None
    self._ctrl_local_ids: torch.Tensor | None = None
    self._ctrl_global_ids: torch.Tensor | None = None
    self._joint_names: tuple[str, ...] = ()

    self._joint_torque_last: torch.Tensor | None = None
    self._joint_torque_mean: torch.Tensor | None = None
    self._joint_torque_limit: torch.Tensor | None = None
    self._joint_torque_history: torch.Tensor | None = None

  def edit_spec(self, scene_spec: mujoco.MjSpec, entities: dict[str, Entity]) -> None:
    del scene_spec  # No scene edits required.
    if self.cfg.entity_name not in entities:
      raise ValueError(
        f"JointTorqueSensor '{self.cfg.name}' expected entity "
        f"'{self.cfg.entity_name}', but it was not found."
      )
    self._entity = entities[self.cfg.entity_name]

  def initialize(
    self, mj_model: mujoco.MjModel, model: mjwarp.Model, data: mjwarp.Data, device: str
  ) -> None:
    del mj_model
    if self._entity is None:
      raise RuntimeError(
        f"JointTorqueSensor '{self.cfg.name}' was not attached to an entity."
      )
    if self.cfg.window_length <= 0:
      raise ValueError(
        f"JointTorqueSensor '{self.cfg.name}' requires window_length > 0, "
        f"got {self.cfg.window_length}."
      )

    joint_name_to_idx = {name: i for i, name in enumerate(self._entity.joint_names)}
    ordered_pairs: list[tuple[int, int, int, str]] = []
    seen_joint_indices: set[int] = set()

    for local_ctrl_idx, actuator in enumerate(self._entity.spec.actuators):
      joint_name = actuator.target.split("/")[-1]
      if joint_name not in joint_name_to_idx:
        continue

      joint_idx = joint_name_to_idx[joint_name]
      if joint_idx in seen_joint_indices:
        raise ValueError(
          f"JointTorqueSensor '{self.cfg.name}' found multiple actuators targeting "
          f"joint '{joint_name}'. This sensor expects one actuator per joint."
        )
      seen_joint_indices.add(joint_idx)
      ordered_pairs.append((joint_idx, local_ctrl_idx, actuator.id, joint_name))

    if not ordered_pairs:
      raise ValueError(
        f"JointTorqueSensor '{self.cfg.name}' could not find actuated joints in "
        f"entity '{self.cfg.entity_name}'."
      )

    ordered_pairs.sort(key=lambda item: item[0])
    self._ctrl_local_ids = torch.tensor(
      [item[1] for item in ordered_pairs], dtype=torch.long, device=device
    )
    self._ctrl_global_ids = torch.tensor(
      [item[2] for item in ordered_pairs], dtype=torch.long, device=device
    )
    self._joint_names = tuple(item[3] for item in ordered_pairs)
    self._model = model

    n_envs = data.time.shape[0]
    n_joints = len(self._joint_names)
    self._joint_torque_last = torch.zeros((n_envs, n_joints), device=device)
    self._joint_torque_mean = torch.zeros((n_envs, n_joints), device=device)
    self._joint_torque_history = torch.zeros(
      (n_envs, self.cfg.window_length, n_joints), device=device
    )
    self._joint_torque_limit = self._compute_joint_torque_limit()
    self._validate_joint_torque_limit(self._joint_torque_limit)

  def _compute_data(self) -> JointTorqueSensorData:
    if (
      self._joint_torque_last is None
      or self._joint_torque_mean is None
      or self._joint_torque_limit is None
    ):
      raise RuntimeError(f"JointTorqueSensor '{self.cfg.name}' is not initialized.")
    return JointTorqueSensorData(
      joint_torque_last=self._joint_torque_last,
      joint_torque_mean=self._joint_torque_mean,
      joint_torque_limit=self._joint_torque_limit,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)

    if self._joint_torque_last is not None:
      self._joint_torque_last[env_ids] = 0.0
    if self._joint_torque_mean is not None:
      self._joint_torque_mean[env_ids] = 0.0
    if self._joint_torque_history is not None:
      self._joint_torque_history[env_ids] = 0.0

  def update(self, dt: float) -> None:
    super().update(dt)

    if (
      self._entity is None
      or self._ctrl_local_ids is None
      or self._joint_torque_last is None
      or self._joint_torque_mean is None
      or self._joint_torque_history is None
    ):
      raise RuntimeError(f"JointTorqueSensor '{self.cfg.name}' is not initialized.")

    current = self._entity.data.actuator_force[:, self._ctrl_local_ids]
    self._joint_torque_last[:] = current
    self._joint_torque_history[:] = self._joint_torque_history.roll(1, dims=1)
    self._joint_torque_history[:, 0] = current
    self._joint_torque_mean[:] = self._joint_torque_history.mean(dim=1)

    self._joint_torque_limit = self._compute_joint_torque_limit()
    self._validate_joint_torque_limit(self._joint_torque_limit)

  def get_torque(self, mode: Literal["mean", "last"]) -> torch.Tensor:
    data = self.data
    if mode == "mean":
      return data.joint_torque_mean
    if mode == "last":
      return data.joint_torque_last
    raise ValueError(
      f"JointTorqueSensor '{self.cfg.name}' unsupported mode '{mode}'. "
      "Expected one of {'mean', 'last'}."
    )

  def _compute_joint_torque_limit(self) -> torch.Tensor:
    if (
      self._model is None
      or self._ctrl_global_ids is None
      or self._joint_torque_last is None
    ):
      raise RuntimeError(f"JointTorqueSensor '{self.cfg.name}' is not initialized.")

    forcerange = self._model.actuator_forcerange[:, self._ctrl_global_ids]
    if forcerange.ndim == 2:
      forcerange = forcerange.unsqueeze(0).expand(
        self._joint_torque_last.shape[0], -1, -1
      )
    limit = torch.maximum(forcerange[..., 1].abs(), forcerange[..., 0].abs())
    return limit

  def _validate_joint_torque_limit(self, limits: torch.Tensor) -> None:
    invalid = torch.nonzero(limits <= 0.0, as_tuple=False)
    if invalid.numel() == 0:
      return

    env_id = int(invalid[0, 0].item())
    joint_id = int(invalid[0, 1].item())
    joint_name = self._joint_names[joint_id]
    value = float(limits[env_id, joint_id].item())
    raise ValueError(
      f"JointTorqueSensor '{self.cfg.name}' detected non-positive torque limit "
      f"for joint '{joint_name}' (index={joint_id}, env={env_id}, value={value}). "
      "Check actuator effort limits before training."
    )
