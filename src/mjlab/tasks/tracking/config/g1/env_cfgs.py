"""Unitree G1 flat tracking environment configurations."""

from typing import Literal

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, JointTorqueSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

TauMode = Literal["off", "reward", "critic", "actor_critic"]
TauFilter = Literal["mean", "last"]


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  tau_mode: TauMode = "off",
  tau_filter: TauFilter = "mean",
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
  if tau_mode not in {"off", "reward", "critic", "actor_critic"}:
    raise ValueError(f"Unsupported tau_mode '{tau_mode}'.")
  if tau_filter not in {"mean", "last"}:
    raise ValueError(f"Unsupported tau_filter '{tau_filter}'.")

  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  scene_sensors = [self_collision_cfg]
  if tau_mode != "off":
    scene_sensors.append(
      JointTorqueSensorCfg(
        name="joint_torque",
        entity_name="robot",
        window_length=cfg.decimation,
      )
    )
  cfg.scene.sensors = tuple(scene_sensors)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )
  motion_cmd.use_joint_tau = tau_mode != "off"

  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms
  actor_terms["command"].params["with_joint_tau"] = tau_mode == "actor_critic"
  critic_terms["command"].params["with_joint_tau"] = tau_mode in (
    "critic",
    "actor_critic",
  )

  # Keep full obs/reward terms in tracking_env_cfg, then remove by mode flags.
  if tau_mode != "actor_critic":
    actor_terms.pop("joint_torque", None)
  else:
    actor_terms["joint_torque"].params["mode"] = tau_filter

  if tau_mode not in ("critic", "actor_critic"):
    critic_terms.pop("joint_torque", None)
  else:
    critic_terms["joint_torque"].params["mode"] = tau_filter

  if tau_mode == "off":
    cfg.rewards.pop("motion_joint_torque", None)
  else:
    cfg.rewards["motion_joint_torque"].params["filter_mode"] = tau_filter

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
