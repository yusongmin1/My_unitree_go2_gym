<div align="center">
  <h1 align="center">Unitree GO2 GYM --于松民</h1>
  <p align="center">
 
  </p>
</div>
 
<p align="center">
  🎮🚪 <strong>这是一个基于 Unitree 机器人实现强化学习的示例仓库开源后修改而成的仓库，支持 Unitree Go2。</strong> 🚪🎮
</p>
 
<p align="center">
  本仓库融合了多种强化学习算法（标准 PPO、DreamwaQ、AMP、CTS），在一个统一框架内支持 12 种训练任务。
</p>
 
---
 
## 📌 算法与任务总览
 
本仓库在 `legged_gym/envs/` 下按算法分组组织任务，在 `rsl_rl/` 下保留了全部算法栈。通过 `task_registry` 的动态 runner 调度，每个任务自动选择对应的算法、网络和 runner。
 
| 算法 | task 名 | 环境目录 | 算法类 / Runner |
|---|---|---|---|
| **标准 PPO** | `go2_trot` | `Go2_MoB/Go2_Trot/` | `PPO` / `OnPolicyRunner` |
| **标准 PPO** | `go2_jump` | `Go2_MoB/Go2_Jump/` | `PPO` / `OnPolicyRunner` |
| **标准 PPO** | `go2_spring_jump` | `Go2_Flip/Go2_Spring_Jump/` | `PPO` / `OnPolicyRunner` |
| **标准 PPO** | `go2_backflip` | `Go2_Flip/Go2_BackFlip/` | `PPO` / `OnPolicyRunner` |
| **标准 PPO** | `go2_handstand` | `Go2_Stand/Go2_Handstand/` | `PPO` / `OnPolicyRunner` |
| **标准 PPO** | `go2_leggedstand` | `Go2_Stand/Go2_Leggedstand/` | `PPO` / `OnPolicyRunner` |
| **DreamwaQ** | `go2_stairs_dreamwaq` | `Go2_DreamWaQ/` | `PPO_DreamWaQ` / `DreamWaQRunner` |
| **AMP + DreamwaQ** | `go2_amp_dreamwaq` | `Go2_AMP_DreamWaQ/` | `PPO_DreamWaQ_AMP` / `DreamWaQRunner_AMP` |
| **CTS** | `go2_cts` | `Go2_Cts/` | `CTS` / `OnPolicyRunnerCTS` |
| **AMP + CTS** | `go2_amp_cts` | `Go2_AMP_Cts/` | `AMPCTS` / `OnPolicyRunnerCTSAMP` |
| **AMP Teacher（特权）** | `go2_amp_ts` | `Go2_AMP_Ts/` | `PPO_AMP_TS` / `OnPolicyRunnerAMP_TS` |
| **AMP Student（蒸馏）** | `go2_amp_ts_student` | `Go2_AMP_Ts/` | `DistillPolicyRunner` + `ActorCritic_Distill`（LSTM） |
 
### 算法简介
- **标准 PPO**：ETH legged_gym 原版 PPO，用于各类特技动作（trot/jump/backflip/handstand 等）。
- **DreamwaQ**：基于隐式地形想象（CENet/VAE）的鲁棒四足运动，从历史观测编码出隐式地形 latent + 显式线速度估计。参考论文 [Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning](https://arxiv.org/abs/2301.10602)。
- **AMP（Adversarial Motion Priors）**：用判别器引导策略学习参考动作（mocap 数据），让运动风格更自然。参考 [rl_amp](https://github.com/fan-ziqi/rl_amp)。
- **CTS（Concurrent Teacher-Student）**：师生并行训练 + 隐变量蒸馏，teacher 吃 privileged obs，student 吃 history obs。参考论文 [arxiv 2405.10830](https://arxiv.org/abs/2405.10830)。
 
---
 
## 🔁 流程说明
 
强化学习实现运动控制的基本流程为：
 
`Train` → `Play` → `Sim2Sim` → `Sim2Real`
 
- **Train**: 通过 Gym 仿真环境，让机器人与环境互动，找到最满足奖励设计的策略。通常不推荐实时查看效果，以免降低训练效率。
- **Play**: 通过 Play 命令查看训练后的策略效果，确保策略符合预期。
- **Sim2Sim**: 将 Gym 训练完成的策略部署到其他仿真器，避免策略小众于 Gym 特性。
- **Sim2Real**: 将策略部署到实物机器人，实现运动控制。
 
## 🛠️ 使用指南
### 0. 安装依赖
python环境：3.8
#### Isaacgym 安装
 
```bash
pip install -e ~/isaacgym/python
```
#### rsl_rl 算法库（含 PPO / DreamwaQ / AMP / CTS 全部算法）
```bash
pip install -e ./rsl_rl
```
#### legged_gym
```bash
pip install -e .
```
#### AMP 额外依赖（仅 AMP 系列任务需要）
```bash
pip install pybullet   # AMP 动作数据加载（pybullet_utils.transformations）
```
 
### 1. 训练
 
运行以下命令进行训练（所有命令需在仓库根目录下执行，AMP 任务依赖 `datasets/` 相对路径）：
 
#### 标准 PPO 任务
```bash
python legged_gym/scripts/train.py --task=go2_trot --headless
python legged_gym/scripts/train.py --task=go2_jump --headless
python legged_gym/scripts/train.py --task=go2_handstand --headless
python legged_gym/scripts/train.py --task=go2_leggedstand --headless
python legged_gym/scripts/train.py --task=go2_spring_jump --headless
python legged_gym/scripts/train.py --task=go2_backflip --headless
```
 
#### DreamwaQ 任务
```bash
python legged_gym/scripts/train.py --task=go2_stairs_dreamwaq --headless
```
 
#### AMP + DreamwaQ 任务
```bash
python legged_gym/scripts/train.py --task=go2_amp_dreamwaq --headless
```
 
#### CTS 任务
```bash
python legged_gym/scripts/train.py --task=go2_cts --headless
```
 
#### AMP + CTS 任务
```bash
python legged_gym/scripts/train.py --task=go2_amp_cts --headless

#### AMP Teacher-Student 任务（先训 teacher 再蒸馏 student）
python legged_gym/scripts/train.py --task=go2_amp_ts --headless          # teacher（AMP 特权策略）
python legged_gym/scripts/train.py --task=go2_amp_ts_student --headless  # student（从 teacher 蒸馏 LSTM）
```
 
 
#### ⚙️  参数说明
- `--task`: 必选参数，可选值见上方任务总览表
- `--headless`: 默认启动图形界面，设为 true 时不渲染图形界面（效率更高）
- `--resume`: 从日志中选择 checkpoint 继续训练
- `--experiment_name`: 运行/加载的 experiment 名称
- `--run_name`: 运行/加载的 run 名称
- `--load_run`: 加载运行的名称，默认加载最后一次运行
- `--checkpoint`: checkpoint 编号，默认加载最新一次文件
- `--num_envs`: 并行训练的环境个数
- `--seed`: 随机种子
- `--max_iterations`: 训练的最大迭代次数
- `--sim_device`: 仿真计算设备，指定 CPU 为 `--sim_device=cpu`
- `--rl_device`: 强化学习计算设备，指定 CPU 为 `--rl_device=cpu`
 
**默认保存训练结果**：`logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`
 
> 各算法的训练日志已按算法分类存放在 `logs/` 下的子目录（`dreamwaq/`、`amp_dreamwaq/`、`amp_cts/`），便于回溯。
 
---
 
### 2. Play
 
不同算法使用对应的 play 脚本（因为各算法的推理/导出逻辑不同）：
 
#### 标准 PPO 任务
```bash
python legged_gym/scripts/play.py --task=go2_trot
python legged_gym/scripts/play.py --task=go2_jump
python legged_gym/scripts/play.py --task=go2_handstand
python legged_gym/scripts/play.py --task=go2_leggedstand
python legged_gym/scripts/play.py --task=go2_spring_jump
python legged_gym/scripts/play.py --task=go2_backflip
```
 
#### DreamwaQ / AMP + DreamwaQ 任务
```bash
python legged_gym/scripts/play_dreamwaq.py --task=go2_stairs_dreamwaq
python legged_gym/scripts/play_dreamwaq.py --task=go2_amp_dreamwaq
```
 
#### CTS 任务
```bash
python legged_gym/scripts/play_cts.py --task=go2_cts
```
 
#### AMP + CTS 任务
```bash
python legged_gym/scripts/play_amp_cts.py --task=go2_amp_cts

#### AMP Teacher-Student 任务
python legged_gym/scripts/play_amp_ts.py --task=go2_amp_ts
python legged_gym/scripts/play_amp_ts_student.py --task=go2_amp_ts_student
```
 
#### AMP 动作数据回放（查看 mocap 参考动作）
```bash
python legged_gym/scripts/replay_amp_data.py --task=go2_amp_cts
```
 
**说明**：
 
- Play 启动参数与 Train 相同。
- 默认加载实验文件夹上次运行的最后一个模型。
- 可通过 `load_run` 和 `checkpoint` 指定其他模型。
 
#### 💾 导出网络
 
Play 会导出 Actor 网络，保存于 `logs/{experiment_name}/exported/policies` 中：
- 普通网络（MLP）导出为 `policy_1.pt`
- DreamwaQ 网络通过 `export_policy_as_dwaq` 导出为 `policy_dwaq.pt`（含 VAE/CENet）
- CTS 网络通过 `export_policy_as_cts` 导出为 `policy_cts.pt`（含 student encoder）

---

### 3. Sim2Sim（MuJoCo 部署验证）

将训练好的策略部署到 MuJoCo 仿真器验证效果。不同算法使用对应的 sim2sim 脚本（位于 `deploy_mujoco/`）：

#### 标准 PPO 任务
```bash
python deploy_mujoco/sim2sim_go2_trot.py
python deploy_mujoco/sim2sim_go2_jump.py
python deploy_mujoco/sim2sim_go2_spring_jump.py
python deploy_mujoco/sim2sim_go2_backflip.py
python deploy_mujoco/sim2sim_go2_handstand.py
python deploy_mujoco/sim2sim_go2_leggedstand.py
```

#### DreamwaQ 任务（加载 policy_dwaq.pt）
```bash
python deploy_mujoco/sim2sim_go2_stairs_dreamwaq.py
```

#### AMP + DreamwaQ 任务（加载 policy_dwaq.pt）
```bash
python deploy_mujoco/sim2sim_go2_amp_dreamwaq.py
```

#### CTS 任务（加载 policy_cts.pt）
```bash
python deploy_mujoco/sim2sim_go2_cts.py
```

#### AMP + CTS 任务（加载 policy_cts.pt）
```bash
python deploy_mujoco/sim2sim_go2_amp_cts.py
```

#### AMP Teacher-Student 任务（student LSTM 部署，加载 policy_amp_ts.pt）
```bash
python deploy_mujoco/sim2sim_go2_amp_ts_student.py
```

**说明**：
- 各脚本默认从 `logs/<experiment_name>/exported/policies/` 加载导出的策略（需先运行对应 Play 脚本导出）。
- 可通过 `--load_model` 参数指定其他策略文件路径。
 
---
 
## 📂 仓库结构
 
```
.
├── legged_gym/
│   ├── envs/
│   │   ├── base/                 # 基类（BaseTask / LeggedRobotCfg 等）
│   │   ├── Go2_MoB/              # 标准 PPO: trot / stairs / jump
│   │   ├── Go2_Flip/             # 标准 PPO: backflip / spring_jump
│   │   ├── Go2_Stand/            # 标准 PPO: handstand / leggedstand
│   │   ├── Go2_DreamWaQ/         # DreamwaQ 任务
│   │   ├── Go2_AMP_DreamWaQ/     # AMP + DreamwaQ 任务
│   │   ├── Go2_Cts/              # CTS 任务
│   │   ├── Go2_AMP_Cts/          # AMP + CTS 任务
│   │   ├── Go2_AMP_Ts/           # AMP Teacher-Student 任务（teacher + student 配置）
│   │   └── __init__.py           # 注册全部 12 个 task
│   ├── scripts/                  # train.py + 各算法的 play 脚本
│   └── utils/                    # task_registry（动态 runner 调度）/ helpers / terrain
├── rsl_rl/                       # 算法库（PPO / DreamwaQ / AMP / CTS 全部共存）
│   └── rsl_rl/
│       ├── algorithms/           # PPO, PPO_DreamWaQ, PPO_DreamWaQ_AMP, CTS, AMPCTS, AMPDiscriminator
│       ├── modules/              # ActorCritic, ActorCriticDreamWaQ, ActorCriticCTS, VAE
│       ├── runners/              # OnPolicyRunner, DreamWaQRunner, OnPolicyRunnerCTS 等
│       ├── storage/              # RolloutStorage, RolloutStorageDreamWaQ, RolloutStorageCTS, ReplayBuffer
│       └── datasets/             # AMPLoader（AMP 动作数据加载）
├── datasets/                     # AMP 动作捕捉数据（mocap_motions_go2 / mocap_motions_a1）
├── deploy_mujoco/                # Sim2Sim 部署脚本（11 个：标准 PPO 6 + CTS 2 + DreamwaQ/AMP 2 + AMP_TS 1）+ 手柄/箭头工具
├── resources/                    # 机器人资产（urdf / mjcf / mesh）
└── logs/                         # 训练日志（按算法分类）
```
 
---
 
# TODO List
- [X] jump任务原地转圈不行，参考go2_trot的修改command即可解决此问题  2026/8/16
- [ ] jump和trot 原地转圈会偏离原始位置，原地站立由于相位的原因会抖动
- [ ] 原始PPO的全部在平坦地形上进行训练，后续需要迁移至不平坦地形
- [ ] trot 的原地转圈在isaacgym中看到基座有点晃动，需要修改
- [ ] stand 任务关节抖动，静止时后退，没有下落控制，粗糙地面没有加入训练
- [ ] 跳远要修改为可以在走路的时候直接切换以及连跳，以及可以控制下落位置和起跳高度
- [ ] 空翻任务使用amp进行约束

- [ ] 爬楼梯任务走停住会翻车 dreamwaq任务，不会立即刹停，cts ts任务可以做到 
- [ ] 步态规范程度不如开源框架himloco 以及dreamwaq
- [ ] amp后退拖地，我认为是amp参数没有调整好
- [ ] amp数据集质量不高
- [ ] 行走时基座晃动
- [ ] 原地转圈拖地，amp比不加amp效果好，但还达不到能卖出去的效果
- [x] 速度采样有死区，在amp任务中存在某些关节角度下发command不动的现象 2026/8/22 ,reset_dof_pos 0.5到1.5,尽量覆盖全部空间，此外对于amp任务，要等待至机器人可以完整上楼梯，否则会卡住导致command死区
- [x] 四条腿抬脚高度和抬脚时间不一致，推测是feet_air_time奖励的影响，去掉feet_air_time奖励，发现四条腿抬脚高度趋于一致，amp_cts任务提供的是加上的，其他任务没有加，可以进行对比2026/8/24
- [ ] 下楼梯会滑下去
- [ ] 下楼梯后腿绊在楼梯上导致翻倒
- [ ] 冲击信号下失稳
- [ ] 走歪
- [ ] 后退不走直线
- [ ] 横着走特别慢 amp任务
- [ ] 横着走会摔倒
- [ ] 第一个台阶会被绊住
- [ ] 一个楼梯没上去，后腿会发疯
- [ ] 低速度跟踪不好
- [ ] 速度过低时机器人在楼梯上被卡住
- [ ] 高速度高角速度失稳
- [ ] amp 3m/s的速度在不平坦地形上奔跑
- [ ] 自身负载还不够大
- [ ] parkour (PIE)
- [ ] 部署代码

# 参考文章
https://arxiv.org/pdf/2205.02824
 
https://arxiv.org/pdf/2309.05665
 
https://arxiv.org/abs/2212.03238
 
https://arxiv.org/abs/2409.15755
 
https://arxiv.org/abs/2401.16337
 
https://arxiv.org/abs/2301.10602
 
https://arxiv.org/abs/2312.11460
