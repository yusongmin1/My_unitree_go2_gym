Create or update the `Mjlab-Hopping-Flat-Unitree-Go2` task in `/home/rkz/code/mjlab/src/mjlab/tasks/hopping/`.

The task trains Unitree Go2 to hop/jump using a gait-phase-synchronized reward, closely following
the go2_jump implementation at
`/home/rkz/code/Isaacgym/src/My_unitree_go2_gym/legged_gym/envs/Go2_MoB/GO2_JUMP/`.

## File structure

```
src/mjlab/tasks/hopping/
├── __init__.py
├── hopping_env_cfg.py           make_hopping_env_cfg() factory
├── mdp/
│   ├── __init__.py
│   ├── observations.py          gait_phase_encoding, euler_xyz, gait_stance_mask,
│   │                            joint_pos_abs, env_friction, robot_total_mass
│   └── rewards.py               jump_gait_sync, default_hip_pos, jump_feet_clearance,
│                                lin_vel_z_reward, ang_vel_xy_reward, base_height_reward,
│                                default_pos_penalty, foot_contact_force_penalty,
│                                jump_air_time (stateful), body_contact_penalty,
│                                joint_vel_diff_l2 (stateful)
└── config/go2/
    ├── __init__.py              registers Mjlab-Hopping-Flat-Unitree-Go2
    ├── env_cfgs.py              unitree_go2_hopping_flat_env_cfg(play=False)
    └── rl_cfg.py                unitree_go2_hopping_ppo_runner_cfg()
```

## Gait phase

`cycle_time = 1.5s`, `phase = (episode_length_buf * step_dt / 1.5) % 1.0`

- `phase < 0.6` → stance: all 4 feet on ground (60% of cycle)
- `phase >= 0.6` → swing: all 4 feet in air (40% of cycle)

## Actor observations — 47 dims × history_length=10 → 470 total

| term | dims | source |
|------|------|--------|
| gait_phase | 2 | `sin(2π·phase), cos(2π·phase)` |
| command | 3 | twist command (vx, vy, ωz) |
| base_ang_vel | 3 | IMU angular velocity, noise ±0.2 |
| euler_xyz | 3 | roll, pitch, yaw, noise ±0.05 |
| joint_pos | 12 | joint pos relative to default, noise ±0.01 |
| joint_vel | 12 | joint velocities, noise ±1.5 |
| actions | 12 | previous actions |
| **total** | **47** | |

## Critic observations — 70 dims × history_length=3 → 210 total

All actor terms (47) plus privileged:

| term | dims | source |
|------|------|--------|
| base_lin_vel | 3 | IMU linear velocity, noise ±0.5 |
| joint_pos_abs | 12 | absolute joint positions |
| gait_stance_mask | 2 | [stance_flag, swing_flag] one-hot |
| foot_contact | 4 | foot ground contact (found > 0) |
| env_friction | 1 | sliding friction of foot geom / 10 |
| body_total_mass | 1 | total robot mass / 10 |
| **total** | **70** | |

## Rewards

| key | weight | function | notes |
|-----|--------|----------|-------|
| track_linear_velocity | +2.0 | `vel_mdp.track_linear_velocity` | std=√0.25 |
| track_angular_velocity | +2.0 | `vel_mdp.track_angular_velocity` | std=√0.25 |
| jump | +2.0 | `mdp.jump_gait_sync` | active when cmd_norm>0.2 |
| base_height | +1.0 | `mdp.base_height_reward` | target=0.3m, active when cmd_norm<0.2 |
| air_time | +1.0 | `mdp.jump_air_time` | reward=(air_time-0.5) per foot at landing |
| orientation | +0.6 | `vel_mdp.flat_orientation` | std=√0.1, trunk body |
| ang_vel_xy | +0.2 | `mdp.ang_vel_xy_reward` | exp(-‖ω_xy‖) |
| feet_clearance | +0.5 | `mdp.jump_feet_clearance` | swing phase, 0.02–0.07m height |
| default_hip_pos | +0.3 | `mdp.default_hip_pos` | exp(-4·Σ\|hip_pos\|) |
| lin_vel_z | +0.05 | `mdp.lin_vel_z_reward` | exp(-\|v_z\|) |
| torques | -0.0002 | `envs_mdp.joint_torques_l2` | |
| dof_acc | -5.5e-4 | `mdp.joint_vel_diff_l2` | finite diff, NOT qacc |
| action_rate | -0.01 | `envs_mdp.action_rate_l2` | |
| foot_contact_forces | -0.01 | `mdp.foot_contact_force_penalty` | max_force=100N |
| default_pos | -0.1 | `mdp.default_pos_penalty` | L1 from default |
| dof_pos_limits | -1.0 | `envs_mdp.joint_pos_limits` | |
| collision | -1.0 | `mdp.body_contact_penalty` | thigh/calf contact, threshold=0.1N |

## Terminations

| key | condition |
|-----|-----------|
| time_out | episode_length_buf > max_episode_length |
| fell_over | trunk tilt > 70° (`bad_orientation`) |
| illegal_contact | trunk body touches terrain (`trunk_ground_contact` sensor) |

Thigh/calf contact is **penalized** (collision reward), not terminated.

## Go2-specific configuration (env_cfgs.py)

**Actuators** — hopping-specific, kp=20 N·m/rad, kd=0.5 N·m·s/rad for all joints:
```python
_HIP_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
  stiffness=20.0, damping=0.5, effort_limit=23.7,
)
_CALF_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=20.0, damping=0.5, effort_limit=45.43,
)
```

**Init state**: `pos z=0.42`, front thigh=0.8, rear thigh=1.0, calf=-1.5,
`.*L_hip=+0.1`, `.*R_hip=-0.1`.

**Action scale**: `{".*": 0.25}` flat for all joints.

**Domain randomization**: foot friction [0.4, 0.8], push ±0.4 m/s every 4s,
encoder bias ±0.035 rad, CoM offset ±0.02 m.

## RL config (rl_cfg.py)

lr=1e-4, max_iterations=15000, num_steps_per_env=24, num_learning_epochs=5,
num_mini_batches=4, save_interval=100, experiment_name="go2_hopping".

## Known gotchas

- **`dof_acc` must use `joint_vel_diff_l2`**: MuJoCo's raw `qacc` can be NaN during contact
  impacts in early training → NaN gradients → `std < 0` crash in RSL-RL.

- **`jump` reward near 1 is normal**: During stance phase (60% of cycle), all-feet-on-ground
  scores 1.0. Actual jumping emerges from `air_time` + `base_height` together.

- **`air_time` reset guard**: `prev_air_time` is cleared when `episode_length_buf <= 1` to
  prevent a spurious ~-2 penalty on the first landing after each episode reset.

- **`euler_xyz` NaN on CUDA**: quaternion is normalized with `F.normalize` before conversion,
  then wrapped with `torch.nan_to_num(..., nan=0.0)`. Required for Blackwell (sm_120).

- **Trunk termination vs nonfoot termination**: velocity task uses `nonfoot_ground_touch`
  (all non-foot geoms). Hopping uses only `trunk_ground_contact` (body="trunk") — thigh/calf
  touches are penalized, not terminated.

- **Commands**: `heading_command=False`, resampling every 5s.

## Verification

```sh
uv run python -c "
import mjlab.tasks
from mjlab.tasks.registry import list_tasks, load_env_cfg
cfg = load_env_cfg('Mjlab-Hopping-Flat-Unitree-Go2')
actor = cfg.observations['actor']
critic = cfg.observations['critic']
print('actor terms:', list(actor.terms.keys()))
print('critic terms:', list(critic.terms.keys()))
print('rewards:', list(cfg.rewards.keys()))
print('terminations:', list(cfg.terminations.keys()))
"
uv run ty check src/mjlab/tasks/hopping/
```
