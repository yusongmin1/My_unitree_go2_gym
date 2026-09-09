# Mjlab-Tracking-Flat-Unitree-Go2-No-State-Estimation

Go2 运动追踪任务（无状态估计版本）。基于动捕数据，使用 PPO 训练 Go2 四足机器人跟踪参考动作。

## 相关文件

```
src/mjlab/tasks/tracking/
├── config/go2/
│   ├── __init__.py          # 任务注册
│   ├── env_cfgs.py          # Go2 环境配置（关键点、tau_mode 等）
│   └── rl_cfg.py            # PPO 网络与训练参数
├── mdp/
│   ├── rewards.py           # 奖励函数实现
│   ├── observations.py      # 观测函数
│   ├── metrics.py           # 评估指标
│   ├── terminations.py      # 终止条件
│   └── commands.py          # MotionCommand（自适应采样）
└── tracking_env_cfg.py      # 基础配置（汇总奖励/观测/事件/终止）
```

## 关键点配置

anchor：`trunk`，body_names 共 13 个：

```python
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
```

与 G1 的对应关系：

| G1 | Go2 | 说明 |
|----|-----|------|
| `torso_link` | `trunk` | anchor |
| `*_hip_roll_link` | `*_hip` | 髋关节 |
| `*_knee_link` | `*_thigh` | 大腿段（最近膝部的 link）|
| `*_ankle_roll_link` | `*_foot` | 足端 |

## Rewards

奖励函数使用指数衰减形式 `exp(-error² / std²)`，继承自 `tracking_env_cfg.py`，与 G1 相同。

| 奖励项 | 权重 | std | 说明 |
|--------|------|-----|------|
| `motion_global_root_pos` | 0.5 | 0.3 | 全局根部位置误差 |
| `motion_global_root_ori` | 0.5 | 0.4 | 全局根部方向误差 |
| `motion_body_pos` | 1.0 | 0.3 | 相对关键点位置误差（13 点平均）|
| `motion_body_ori` | 1.0 | 0.4 | 相对关键点方向误差（13 点平均）|
| `motion_body_lin_vel` | 1.0 | 1.0 | 关键点线速度误差 |
| `motion_body_ang_vel` | 1.0 | 3.14 | 关键点角速度误差 |
| `action_rate_l2` | -0.1 | — | 动作变化率 L2 惩罚 |
| `joint_limit` | -10.0 | — | 关节超限惩罚 |
| `self_collisions` | -10.0 | — | 自碰撞惩罚 |

## Actor Observations

无状态估计：移除 `motion_anchor_pos_b` 和 `base_lin_vel`。

| 观测项 | 噪声范围 | 说明 |
|--------|---------|------|
| `command` | — | 参考关节位置 + 速度 |
| `motion_anchor_ori_b` | [-0.05, 0.05] | 参考根部在本体系方向（2×2 矩阵 → 4 维）|
| `base_ang_vel` | [-0.2, 0.2] | IMU 角速度 |
| `joint_pos` | [-0.01, 0.01] | 相对关节位置（减去默认值）|
| `joint_vel` | [-0.5, 0.5] | 相对关节速度 |
| `actions` | — | 上一步动作 |

## Critic Observations

完整状态，无噪声。

| 观测项 | 说明 |
|--------|------|
| `command` | 参考关节位置 + 速度 |
| `motion_anchor_pos_b` | 参考根部在本体系位置 |
| `motion_anchor_ori_b` | 参考根部在本体系方向 |
| `body_pos` | 所有 13 个关键点本体系位置 |
| `body_ori` | 所有 13 个关键点本体系方向 |
| `base_lin_vel` | 线速度 |
| `base_ang_vel` | 角速度 |
| `joint_pos` | 关节位置 |
| `joint_vel` | 关节速度 |
| `actions` | 上一步动作 |

## Metrics

| 指标 | 说明 |
|------|------|
| `MPKPE` | 所有关键点全局位置误差 L2 均值 |
| `R-MPKPE` | 相对根部的关键点位置误差均值（去除全局漂移）|
| `joint_velocity_error` | 关节速度 L2 范数 |
| `ee_position_error` | 四足足端位置误差均值 |
| `ee_orientation_error` | 四足足端方向误差均值 |

## Termination Conditions

| 条件 | 阈值 | 说明 |
|------|------|------|
| `time_out` | 10.0 s | Episode 超时 |
| `anchor_pos` | 1.0 m | 根部 Z 轴位置误差过大 |
| `anchor_ori` | 0.8 | 根部方向误差（投影重力向量 Z 分量）|
| `ee_body_pos` | 1.0 m | 任一足端 Z 轴位置误差过大 |

足端关键点（`ee_body_pos` 用）：`FL_foot`, `FR_foot`, `RL_foot`, `RR_foot`

## 关键设计

- **13 个关键点**：trunk + 每条腿的 hip/thigh/foot（FL/FR/RL/RR）
- **自碰撞检测**：G1 用 `pelvis` 子树，Go2 需改为 `trunk` 子树
- **脚部摩擦随机化**：G1 用 `^(left|right)_foot[1-7]_collision$`，Go2 需改为对应 geom 名称
- **自适应采样**（`sampling_mode="adaptive"`）：根据失败位置调整采样概率
- **RSI 随机化**：重置时随机化初始位置（±0.05 m）、偏航（±0.2 rad）、速度（±0.5 m/s）
- **Domain Randomization**：外力扰动（1~3 s 间隔）、重心偏移（±0.05 m）、脚部摩擦（0.3~1.2）、编码器偏差（±0.01 rad）
- **控制频率**：50 Hz（`decimation=4`，物理步长 0.005 s）
