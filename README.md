<div align="center">
  <h1 align="center">Unitree GO2 GYM --YuSongmin</h1>
  <p align="center">
    <a href="README.md">🌎 English</a> | <span>🇨🇳 中文</span>
  </p>
</div>

<p align="center">
  🎮🚪 <strong>这是一个基于 Unitree 机器人实现强化学习的示例仓库开源后修改而成的仓库，支持 Unitree Go2。</strong> 🚪🎮
</p>

---

## 🔁 流程说明

强化学习实现运动控制的基本流程为：

`Train` → `Play` → `Sim2Sim` → `Sim2Real`

- **Train**: 通过 Gym 仿真环境，让机器人与环境互动，找到最满足奖励设计的策略。通常不推荐实时查看效果，以免降低训练效率。
- **Play**: 通过 Play 命令查看训练后的策略效果，确保策略符合预期。
- **Sim2Sim**: 将 Gym 训练完成的策略部署到其他仿真器，避免策略小众于 Gym 特性。
- **Sim2Real**: 将策略部署到实物机器人，实现运动控制。

## 🛠️ 使用指南

### 1. 训练

运行以下命令进行训练：

```bash
python legged_gym/scripts/train.py --task=go2_jump --headless
```
```bash
python legged_gym/scripts/train.py --task=go2_handstand --headless
```
```bash
python legged_gym/scripts/train.py --task=go2_handstand_command --headless
```

handstand 需要把mirror loss注释掉
#### ⚙️  参数说明
- `--task`: 必选参数，值可选(go2, g1, h1, h1_2)
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

---

### 2. Play

如果想要在 Gym 中查看训练效果，可以运行以下命令：

```bash
python legged_gym/scripts/play.py --task=go2_spring_jump
```

```bash
python legged_gym/scripts/play.py --task=go2_jump
```

```bash
python legged_gym/scripts/play.py --task=go2_handstand_command
```
**说明**：

- Play 启动参数与 Train 相同。
- 默认加载实验文件夹上次运行的最后一个模型。
- 可通过 `load_run` 和 `checkpoint` 指定其他模型。

#### 💾 导出网络

Play 会导出 Actor 网络，保存于 `logs/{experiment_name}/exported/policies` 中：
- 普通网络（MLP）导出为 `policy_1.pt`


### 3. Sim2Sim (Mujoco)



#### 示例：运行 Go2 handstand

```bash
python deploy/deploy_mujoco/deploy_mujoco_48_handstand.py go2.yaml
```
```bash
python deploy/deploy_mujoco/sim2sim_GO2.py --load_model logs/go2_jump/exported/policies/policy_1.pt
```

```bash
python deploy/deploy_mujoco/sim2sim_GO2.py --load_model logs/go2_trot/exported/policies/policy_1.pt
```
-
## 运行说明
deploy_mujoco_48_handstand.py go2的handstand版本，没有base_line_vel，状态sin cos command ,command此时默认为0，后续可能加入起身下落的控制

sim2sim_GO2.py 改自众擎开源项目的sim2sim，后续都会使用这个

目前只有handstand jump是有效的，其他的还没有做完，我的想法是把其他的项目比方说BACKFLIP，难以阅读的代码整合到legged_gym框架，方便后来者进行学习
# 问题 与后续修改的计划

如何调节抬脚高度（上下楼梯）

改变相位与自身高度(walk these ways)

PAKOUR

BACKFLIP

# 参考文章
https://arxiv.org/pdf/2309.05665

https://arxiv.org/abs/2212.03238

https://arxiv.org/abs/2409.15755

https://arxiv.org/abs/2401.16337
