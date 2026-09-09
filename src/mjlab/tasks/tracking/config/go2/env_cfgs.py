"""Unitree Go2 flat tracking environment configurations."""

from mjlab.actuator import DelayedActuatorCfg
from mjlab.asset_zoo.robots import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def unitree_go2_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
  enable_body_mass_rand: bool = True,
  enable_actuator_gains_rand: bool = True,
  enable_actuator_delay: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain tracking configuration.

  Args:
    has_state_estimation: Keep privileged state terms in the actor obs.
    play: Disable episode timeout / DR used during interactive play.
    enable_body_mass_rand: Startup body-mass add randomization.
    enable_actuator_gains_rand: Startup PD gain scale randomization.
    enable_actuator_delay: Wrap actuators with 0–3 physics-step delay.
  """
  cfg = make_tracking_env_cfg()

  robot_cfg = get_go2_robot_cfg()
  if enable_actuator_delay and not play:
    articulation = robot_cfg.articulation
    assert articulation is not None
    robot_cfg.articulation = EntityArticulationInfoCfg(
      actuators=tuple(
        DelayedActuatorCfg(
          base_cfg=act,
          delay_target="position",
          delay_min_lag=0,
          delay_max_lag=3,
        )
        for act in articulation.actuators
      ),
      soft_joint_pos_limit_factor=articulation.soft_joint_pos_limit_factor,
    )
  cfg.scene.entities = {"robot": robot_cfg}

  cfg.scene.sensors = (
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = GO2_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "trunk"
  motion_cmd.body_names = (
    "trunk",
    "FL_hip",
    "FL_thigh",
    "FL_foot",
    "FR_hip",
    "FR_thigh",
    "FR_foot",
    "RL_hip",
    "RL_thigh",
    "RL_foot",
    "RR_hip",
    "RR_thigh",
    "RR_foot",
  )

  # Go2 has no joint torque support — remove tau reward term.
  cfg.rewards.pop("motion_joint_torque", None)

  # Go2 has no tau mode — remove tau obs terms.
  cfg.observations["actor"].terms.pop("joint_torque", None)
  cfg.observations["critic"].terms.pop("joint_torque", None)

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^[FR][LR]_foot_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

  if not enable_body_mass_rand or play:
    cfg.events.pop("add_body_mass", None)
  if not enable_actuator_gains_rand or play:
    cfg.events.pop("actuator_gains", None)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
  )

  cfg.viewer.body_name = "trunk"

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

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
