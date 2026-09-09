# Motion Command 结构说明

Tracking 任务使用 `MotionCommand`（`mdp/commands.py`）驱动机器人跟踪参考动作。

## .npz 文件格式

| 键 | 形状 | 说明 |
|----|------|------|
| `joint_pos` | `(T, n_joints)` | 关节位置（弧度） |
| `joint_vel` | `(T, n_joints)` | 关节速度（弧度/秒） |
| `body_pos_w` | `(T, n_bodies, 3)` | 所有 body 世界坐标位置 |
| `body_quat_w` | `(T, n_bodies, 4)` | 所有 body 世界坐标四元数（wxyz） |
| `body_lin_vel_w` | `(T, n_bodies, 3)` | 所有 body 世界坐标线速度 |
| `body_ang_vel_w` | `(T, n_bodies, 3)` | 所有 body 世界坐标角速度 |
| `joint_tau` | `(T, n_joints)` | 关节扭矩（可选，`use_joint_tau=True` 时必须） |

`T` 为总帧数，`n_bodies` 为机器人全部 body 数量（非仅追踪的关键点）。

## command 属性

`MotionCommand.command` 返回当前时间步的参考状态，按以下方式拼接：

| 模式 | 内容 | 形状 |
|------|------|------|
| 默认（`command_no_tau`） | `[joint_pos, joint_vel]` | `(n_envs, 2×n_joints)` |
| 含扭矩（`command_with_tau`） | `[joint_pos, joint_vel, joint_tau]` | `(n_envs, 3×n_joints)` |

## 关键点（body_names）

`MotionCommandCfg.body_names` 指定追踪的关键点子集，`anchor_body_name` 为根部锚点。

**Go2**（13 点，anchor = `trunk`）：
```
trunk, FL_hip, FL_thigh, FL_foot, FR_hip, FR_thigh, FR_foot,
RL_hip, RL_thigh, RL_foot, RR_hip, RR_thigh, RR_foot
```

**G1**（14 点，anchor = `torso_link`）：
```
pelvis,
left_hip_roll_link, left_knee_link, left_ankle_roll_link,
right_hip_roll_link, right_knee_link, right_ankle_roll_link,
torso_link,
left_shoulder_roll_link, left_elbow_link, left_wrist_yaw_link,
right_shoulder_roll_link, right_elbow_link, right_wrist_yaw_link
```

## 派生状态属性

`MotionCommand` 在每步更新后暴露以下属性，供奖励/观测使用：

| 属性 | 说明 |
|------|------|
| `joint_pos / joint_vel / joint_tau` | 参考关节状态 |
| `body_pos_w / body_quat_w` | 参考关键点世界坐标（已筛选 body_names） |
| `body_lin_vel_w / body_ang_vel_w` | 参考关键点世界速度 |
| `anchor_pos_w / anchor_quat_w` | 参考锚点世界位姿 |
| `anchor_lin_vel_w / anchor_ang_vel_w` | 参考锚点世界速度 |
| `body_pos_relative_w` | 关键点相对锚点的位置 |
| `body_quat_relative_w` | 关键点相对锚点的姿态 |
| `robot_*` 系列 | 机器人当前实际状态（与参考一一对应） |

## 重置初始化（RSI）

重置时在参考动作的采样帧附近随机扰动：

| 参数 | 范围（默认） |
|------|-------------|
| 位置 xy | ±0.05 m |
| 位置 z | ±0.01 m |
| 偏航角 | ±0.2 rad |
| 线速度 | ±0.5 m/s |
| 角速度 | ±0.52 rad/s |

## 时间步采样模式

| 模式 | 说明 |
|------|------|
| `adaptive`（默认） | 根据历史失败位置加权采样，难帧采样概率更高 |
| `uniform` | 随机均匀采样 |
| `start` | 每次从 t=0 开始 |

## 转换脚本

| 脚本 | 输入格式 | 用途 |
|------|----------|------|
| `convert_mimickit.py` | MimicKit `.pkl` | Go2 / G1 动捕数据 |
| `convert_gc_go2.py` | 广义坐标 CSV（19 列） | Go2 仿真轨迹 |
| `csv_to_npz.py` | 运动 CSV + 可选扭矩 CSV | 通用 CSV 格式 |
