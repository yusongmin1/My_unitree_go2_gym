"""Unitree Go2 hopping environment configuration."""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots import get_go2_robot_cfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.hopping.hopping_env_cfg import make_hopping_env_cfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

# Actuator gains matching go2_jump (kp=20 N·m/rad, kd=0.5 N·m·s/rad).
_HIP_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=20.0,
  damping=0.5,
  effort_limit=23.7,
)
_CALF_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=20.0,
  damping=0.5,
  effort_limit=45.43,
)

# Initial state matching go2_jump (pos z=0.42, joint angles from go2_jump).
_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.42),
  joint_pos={
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    "F[LR]_thigh_joint": 0.8,
    "R[LR]_thigh_joint": 1.0,
    ".*calf_joint": -1.5,
  },
  joint_vel={".*": 0.0},
)


def unitree_go2_hopping_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain hopping configuration."""
  cfg = make_hopping_env_cfg()

  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.mujoco.ccd_iterations = 50

  robot_cfg = get_go2_robot_cfg()
  robot_cfg.init_state = _INIT_STATE
  robot_cfg.articulation = EntityArticulationInfoCfg(
    actuators=(_HIP_ACTUATOR, _CALF_ACTUATOR),
    soft_joint_pos_limit_factor=0.9,
  )
  cfg.scene.entities = {"robot": robot_cfg}

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  # # Only terminate on trunk contact with ground (matches go2_jump's
  # # terminate_after_contacts_on=["base"]). Thigh/calf contacts are
  # # penalized but do NOT trigger termination.
  # trunk_ground_cfg = ContactSensorCfg(
  #   name="trunk_ground_contact",
  #   primary=ContactMatch(mode="body", pattern="trunk", entity="robot"),
  #   secondary=ContactMatch(mode="body", pattern="terrain"),
  #   fields=("found",),
  #   reduce="none",
  #   num_slots=1,
  # )
  # # Penalize thigh/calf ground contact (matches go2_jump's
  # # penalize_contacts_on=["thigh", "calf"]).
  # thigh_calf_cfg = ContactSensorCfg(
  #   name="thigh_calf_contact",
  #   primary=ContactMatch(
  #     mode="body",
  #     pattern=r"(FL|FR|RL|RR)_(thigh|calf)",
  #     entity="robot",
  #   ),
  #   secondary=ContactMatch(mode="body", pattern="terrain"),
  #   fields=("force",),
  #   reduce="netforce",
  #   num_slots=1,
  # )
  # cfg.scene.sensors = (feet_ground_cfg, trunk_ground_cfg, thigh_calf_cfg)

  # Action scale: 0.25 for all joints, matching go2_jump's action_scale=0.25.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = {".*": 0.25}

  cfg.viewer.body_name = "trunk"

  # Set one foot geom for friction observation (shared_random=True means all
  # foot geoms share the same value, so reading one is sufficient)
  cfg.observations["critic"].terms["env_friction"].params["asset_cfg"].geom_names = (
    geom_names[0],
  )

  # Set foot site names for feet_clearance reward
  cfg.rewards["feet_clearance"].params["asset_cfg"].site_names = site_names

  # Set trunk body for orientation reward
  cfg.rewards["orientation"].params["asset_cfg"].body_names = ("trunk",)

  # Set foot geoms for friction randomization
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

  # Set trunk body for CoM randomization
  cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk",)

  # Termination: trunk/base touches ground (matches go2_jump behavior)
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=vel_mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg
