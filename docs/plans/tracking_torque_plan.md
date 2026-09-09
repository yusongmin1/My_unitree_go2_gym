## Torque Tracking 改造计划（仅改命名，其他不变）

### Summary
1. `tracking_env_cfg` 中不新增显式“期望扭矩”观察项；扭矩参考值只通过 `command` 通道进入（按模式决定 actor/critic 的 command 是否带 tau）。
2. 配置只保留两个新开关：`tau_mode` 与 `tau_filter`。
3. `tau_mode` 统一控制 obs/reward：`off | reward | critic | actor_critic`。
4. `tau_filter` 同时作用于 `joint_torque` obs 和 `motion_joint_torque` reward。
5. `joint_tau` 时间维按要求对齐：`T_pos=T`，`T_tau=T-1`，并做 `tau_aligned[0]=0, tau_aligned[t]=joint_tau[t-1]`。
6. `tau_limit` 不做 clamp；若存在 `<=0`，在 `JointTorqueSensor.initialize()` 直接报错终止训练。
7. **仅命名变更**：`MotionCommand` 新增属性不叫 `desired_joint_torque`，改为中性命名 `joint_tau`（表示对齐后的 command tau）。

### Public API / Schema Changes

#### 1) `unitree_g1_flat_tracking_env_cfg(...)`
新增仅两个参数：
1. `tau_mode: Literal["off", "reward", "critic", "actor_critic"] = "off"`
2. `tau_filter: Literal["mean", "last"] = "mean"`

不新增 `tau_reward`，不新增 `tau_reward_std/weight` 参数。

#### 2) `MotionCommandCfg`
新增一个参数：
1. `use_joint_tau: bool = False`

含义：
1. `False` 时不读取 `joint_tau`，command 为 `2J`。
2. `True` 时必须读取 `joint_tau` 并校验，支持生成对齐后的 `joint_tau`。

#### 3) `motion.npz` 约定
1. 只支持键：`joint_tau`。
2. 不做别名兼容。
3. 形状要求：`joint_tau.shape == (T-1, J)`，其中 `joint_pos.shape == (T, J)`。

### Mode 语义（唯一真值表）

| tau_mode | actor command | critic command | actor `joint_torque` obs | critic `joint_torque` obs | torque reward |
|---|---|---|---|---|---|
| `off` | `2J` | `2J` | no | no | no |
| `reward` | `2J` | `2J` | no | no | yes |
| `critic` | `2J` | `3J` | no | yes | yes |
| `actor_critic` | `3J` | `3J` | yes | yes | yes |

说明：
1. `3J` 表示 command 中拼入对齐后的 `joint_tau`。
2. 不出现显式 `desired_*` torque 观察项。

### File-by-File 实现计划

#### A. `mjlab/src/mjlab/tasks/tracking/mdp/commands.py`
1. `MotionLoader` 增加 `joint_tau` 读取逻辑（仅当 `use_joint_tau=True`）。
2. 增加 fail-fast 校验：
1. `joint_tau.ndim == 2`
2. `joint_tau.shape[0] == joint_pos.shape[0] - 1`
3. `joint_tau.shape[1] == joint_pos.shape[1]`
3. 在 `MotionCommand` 初始化时构造对齐张量：
1. 内部缓存名可为 `_joint_tau_aligned`
2. shape `(T, J)`
3. 第 0 帧全 0
4. 后续帧右移填充 `joint_tau`
4. 新增属性：
1. `joint_tau`：返回 `_joint_tau_aligned[self.time_steps]`（中性命名，替代 `desired_joint_torque`）
2. `command_no_tau`：`cat(joint_pos, joint_vel)`
3. `command_with_tau`：`cat(joint_pos, joint_vel, joint_tau)`

#### B. `mjlab/src/mjlab/tasks/tracking/mdp/observations.py`
1. 新增 `command(env, command_name, with_joint_tau: bool)`：
1. `False` 返回 `command_no_tau`
2. `True` 返回 `command_with_tau`
2. 新增 `joint_torque(env, sensor_name, mode)`：
1. `mode=="mean"` 返回 sensor 的 mean
2. `mode=="last"` 返回 sensor 的 last

#### C. `mjlab/src/mjlab/sensor/joint_torque_sensor.py`
1. 新建 `JointTorqueSensorCfg` 与 `JointTorqueSensorData`。
2. 在 `initialize()` 中完成：
1. actuator->joint 映射
2. joint torque limit 映射
3. limit 任一 `<=0` 直接抛错（包含 joint id/name）
3. 在 `update(dt)` 中每个 substep 更新：
1. 当前 `joint_torque_last`
2. 环形缓存（长度设为 decimation）
3. `joint_torque_mean`
4. `data` 同时暴露 `last/mean/limit`。

#### D. `mjlab/src/mjlab/sensor/__init__.py`
1. 导出新 sensor 类型与 cfg/data。

#### E. `mjlab/src/mjlab/tasks/tracking/mdp/rewards.py`
1. 新增 `motion_joint_torque(env, command_name, sensor_name, filter_mode, std)`：
1. ref = `MotionCommand.joint_tau`
2. act = sensor output（按 `filter_mode`）
3. limit = sensor limit
4. `err = (ref - act) / limit`
5. `reward = exp(-norm(err,2)/std)`
2. reward term 名固定：`motion_joint_torque`。

#### F. `mjlab/src/mjlab/tasks/tracking/tracking_env_cfg.py`
1. 保持 `command` term 名不变，调用改为新 `mdp.command(...)`。
2. 不新增显式“desired torque” observation term。
3. reward 字典里增加常量配置项（固定在此，不放 `env_cfgs` 参数）：
1. `motion_joint_torque` 的 `std` 与 `weight` 常量
4. 默认 base config 下不启用 torque（由 `env_cfgs` 按 mode 覆盖启用/关闭）。

#### G. `mjlab/src/mjlab/tasks/tracking/config/g1/env_cfgs.py`
1. 新增 `tau_mode` 与 `tau_filter` 参数。
2. 按 mode 执行组合开关：
1. 配置 actor/critic 的 command `with_joint_tau`
2. 配置 actor/critic 是否添加 `joint_torque` term
3. 配置是否启用 `motion_joint_torque` reward
4. 配置 `motion_cmd.use_joint_tau`（`mode != off`）

#### H. `mjlab/src/mjlab/tasks/tracking/config/g1/__init__.py`
1. 新增 no-state torque task 注册：
1. `task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation-Torque"`
2. `tau_mode="actor_critic"`
3. `tau_filter="mean"`

### Testing and Validation

1. 数据校验测试：
1. `joint_tau` 缺失：`use_joint_tau=True` 时应报错。
2. 时间维非 `T-1`：报错。
3. 关节维非 `J`：报错。

2. 对齐逻辑测试：
1. 验证 `tau_aligned[0]=0`。
2. 验证 `tau_aligned[1:]==joint_tau[:]`。

3. sensor 初始化测试：
1. 任一关节 limit `<=0` 时初始化报错。

4. filter 一致性测试：
1. `tau_filter=mean` 时 obs 与 reward 都用 mean。
2. `tau_filter=last` 时 obs 与 reward 都用 last。

5. mode 行为测试：
1. 四种 mode 下 obs 维度、reward term 是否存在与真值表一致。

6. 回归测试：
1. `tau_mode=off` 时旧任务维度与行为不变。

### Assumptions
1. `joint_tau` 与 sim `actuator_force` 同单位（N·m）。
2. G1 任务中关节存在有效 torque limit；若无则视为配置错误并中止。
3. reward 的 `std/weight` 在 `tracking_env_cfg.py` 作为固定常量管理，不对外暴露。
