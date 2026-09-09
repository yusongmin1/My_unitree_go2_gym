"""Hopping task configuration.

This module provides a factory function to create a base hopping task config,
closely following the go2_jump implementation from IsaacGym.
Robot-specific configurations call the factory and customize as needed.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.hopping import mdp
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# Gait cycle period matching go2_jump config (cycle_time = 1.5s)
_CYCLE_TIME = 1.5


def make_hopping_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base hopping task configuration.

  Closely follows the go2_jump IsaacGym implementation with:
  - Gait-phase-synchronized jump reward
  - Phase encoding in actor observations
  - Flat terrain only
  """

  ##
  # Observations
  ##

  actor_terms = {
    # Gait phase encoding: sin(2π·phase), cos(2π·phase) — matches go2_jump command_input[:2]
    "gait_phase": ObservationTermCfg(
      func=mdp.gait_phase_encoding,
      params={"cycle_time": _CYCLE_TIME},
    ),
    # Velocity commands — matches go2_jump command_input[2:5]
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    # IMU angular velocity — matches go2_jump obs_imu[:3]
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    # Euler angles — matches go2_jump obs_imu[3:6] (base_euler_xyz)
    "euler_xyz": ObservationTermCfg(
      func=mdp.euler_xyz,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    # Joint positions relative to default — matches go2_jump obs_motor[:12]
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Joint velocities — matches go2_jump obs_motor[12:24]
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    # Previous actions — matches go2_jump actions (12)
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    # Privileged: actual base linear velocity
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    # Privileged: absolute joint positions (goes2_jump dof_pos)
    "joint_pos_abs": ObservationTermCfg(
      func=mdp.joint_pos_abs,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # Privileged: gait stance mask (go2_jump stance_mask)
    "gait_stance_mask": ObservationTermCfg(
      func=mdp.gait_stance_mask,
      params={"cycle_time": _CYCLE_TIME},
    ),
    # Privileged: foot contact state (go2_jump contact_mask)
    "foot_contact": ObservationTermCfg(
      func=vel_mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    # Privileged: foot geom friction (go2_jump env_frictions, set per-robot)
    "env_friction": ObservationTermCfg(
      func=mdp.env_friction,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=())},
    ),
    # Privileged: total robot mass / 10 (go2_jump body_mass / 10.)
    "body_total_mass": ObservationTermCfg(
      func=mdp.robot_total_mass,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      history_length=10,  # matches go2_jump frame_stack=10
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      history_length=3,  # matches go2_jump c_frame_stack=3
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,  # Override per-robot with GO2_ACTION_SCALE
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      # Fixed 5s resampling matching go2_jump's resampling_time=5
      resampling_time_range=(5.0, 5.0),
      rel_standing_envs=0.0,
      rel_heading_envs=0.0,
      # go2_jump uses heading_command=False
      heading_command=False,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
      ),
    )
  }

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    "reset_base": EventTermCfg(
      func=vel_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.01, 0.05),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=vel_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=vel_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(4.0, 4.0),  # go2_jump push_interval_s=4
      params={
        "velocity_range": {
          "x": (-0.4, 0.4),
          "y": (-0.4, 0.4),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=vel_mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.4, 0.8),  # go2_jump friction_range=[0.4,0.8]
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=vel_mdp.randomize_encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.035, 0.035),  # go2_jump motor_zero_offset_range
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=vel_mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.02, 0.02),
          1: (-0.02, 0.02),
          2: (-0.02, 0.02),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards: dict[str, RewardTermCfg] = {
    # Velocity tracking (go2_jump tracking_lin_vel=2.0, tracking_ang_vel=2.0)
    "track_linear_velocity": RewardTermCfg(
      func=vel_mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=vel_mdp.track_angular_velocity,
      weight=2.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    # Stability rewards (go2_jump lin_vel_z=0.05, ang_vel_xy=0.2, orientation=0.6)
    "lin_vel_z": RewardTermCfg(
      func=mdp.lin_vel_z_reward,
      weight=0.05,
    ),
    "ang_vel_xy": RewardTermCfg(
      func=mdp.ang_vel_xy_reward,
      weight=0.2,
    ),
    "orientation": RewardTermCfg(
      func=vel_mdp.flat_orientation,
      weight=0.6,
      params={
        "std": math.sqrt(0.1),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    # Height reward (go2_jump base_height=1.0, target=0.3m)
    "base_height": RewardTermCfg(
      func=mdp.base_height_reward,
      weight=1.0,
      params={
        "target_height": 0.3,
        "command_name": "twist",
      },
    ),
    # Jump-specific rewards (go2_jump jump=2.0, feet_clearance=0.5)
    "jump": RewardTermCfg(
      func=mdp.jump_gait_sync,
      weight=2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "cycle_time": _CYCLE_TIME,
      },
    ),
    "feet_clearance": RewardTermCfg(
      func=mdp.jump_feet_clearance,
      weight=0.5,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        "cycle_time": _CYCLE_TIME,
        "command_name": "twist",
      },
    ),
    # Air time at landing: negative if < 0.5s, positive if > 0.5s
    # (go2_jump feet_air_time=1.0, threshold=0.5s)
    "air_time": RewardTermCfg(
      func=mdp.jump_air_time,
      weight=1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "min_air_time": 0.5,
      },
    ),
    # Hip position (go2_jump default_hip_pos=0.3)
    "default_hip_pos": RewardTermCfg(
      func=mdp.default_hip_pos,
      weight=0.3,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_joint.*",)),
      },
    ),
    # Penalty rewards
    "torques": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-0.0002,
    ),
    # Finite difference of joint velocities (matches go2_jump's dof_acc).
    # Uses velocity diff instead of MuJoCo's qacc, which can be NaN during
    # contact impacts in early training.
    "dof_acc": RewardTermCfg(
      func=mdp.joint_vel_diff_l2,
      weight=-5.5e-4,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "action_rate": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.01,
    ),
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-1.0,
    ),
    "default_pos": RewardTermCfg(
      func=mdp.default_pos_penalty,
      weight=-0.1,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "foot_contact_forces": RewardTermCfg(
      func=mdp.foot_contact_force_penalty,
      weight=-0.01,
      params={
        "sensor_name": "feet_ground_contact",
        "max_force": 100.0,
      },
    ),
    # Thigh/calf collision penalty (go2_jump collision=-1.0).
    # Uses force threshold >0.1 N to match go2_jump's _reward_collision.
    # Sensor created per-robot in robot-specific config.
    # "collision": RewardTermCfg(
    #   func=mdp.body_contact_penalty,
    #   weight=-1.0,
    #   params={"sensor_name": "thigh_calf_contact"},
    # ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=vel_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=1.5,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
    ),
    decimation=4,
    episode_length_s=24.0,  # go2_jump episode_length_s=24
  )
