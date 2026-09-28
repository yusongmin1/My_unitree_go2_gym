# 深度 / 图像处理完整对比（逐步版）

本文对比以下实现（**只依据代码**，不含注释中的意图说明）：

| 简称 | 路径 |
|------|------|
| **Go2_PIE** | `My_unitree_go2_gym/legged_gym/envs/Go2_PIE` + 内嵌 `legged_gym/warp_sensor` |
| **MGDP** | `parkour/MGDP-master/.../random_dog` + `warp_sensor` |
| **extreme-parkour** | `parkour/extreme-parkour-main/.../legged_robot.py` |
| **WMP** | `parkour/WMP-master/.../legged_robot.py` |
| **parkour-main** | `parkour/parkour-main/.../legged_robot_noisy.py` |
| **本仓库无深度任务** | `Go2_DreamWaQ` / `Go2_Cts` / `Go2_AMP_*`：仅 viewer，无策略深度管线 |

默认数值取各项目主配置 / stage1；具体 run 可能被命令行覆盖。

---

## 0. 端到端数据流总图

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────┐
│ 仿真几何     │ ──► │ 深度渲染后端      │ ──► │ 几何/数值后处理   │ ──► │ 时序缓冲      │
│ (机体+地形)  │     │ Warp / Isaac Gym │     │ crop/resize/噪/ │     │ 帧栈 / 延迟   │
└─────────────┘     └──────────────────┘     │ 归一化           │     └──────┬───────┘
                                              └─────────────────┘            │
                                                                             ▼
                                                                    ┌────────────────┐
                                                                    │ 观测打包        │
                                                                    │ → runner/策略   │
                                                                    └────────────────┘
```

各项目在「渲染后端 / 后处理 / 缓冲策略 / 进策略方式」四处分叉最大。

---

## 1. 初始化阶段（创建相机 + 缓冲）

| 步骤 | Go2_PIE | MGDP | extreme-parkour | WMP | parkour-main |
|------|---------|------|-----------------|-----|--------------|
| 何时创建 | `_init_camera_buffers`（`create_sim` 后） | `__init__` 里建缓冲 + `WarpManager` | `_create_envs` → `attach_camera` | 同 extreme | sensor 配置 + tensor dict |
| 渲染器 | `WarpManager(num_envs, env, cfg)` | 同左（外部 `warp_sensor`） | `gym.create_camera_sensor` × N | 同左 | Isaac 相机句柄 |
| 原始出图尺寸 | pattern **60×106** | pattern 默认 **120×120**（`DepthCameraPattern`），再经 process resize | props **106×60**（W×H 配置名 `original`） | 同左 | 如 240×424 等可配 |
| 缓冲分配 | `image_buf (N, 2, 60, 86)` | `image_buf` + **`image_clean_buf`** 同形状 `(N, 2, H', W')`，H'W'=`resized` | `depth_buffer (N, buffer_len, H_r, W_r)` | 同左 | `forward_depth_obs_buffer` 等延迟栈 |
| Warp process 链 | **显式关掉**：`resize=None, noise=None, clip=None, normalize=False`（后处理全在 env） | 使用 `Camera.process`：默认 **resize(16,16)、clip[0.1,2.5]、normalize=True**；库内噪声多注释掉，env 再加噪 | 无 Warp | 无 | 无 |
| 地形 mesh 坐标 | 创建 Warp 前临时 `border_size`，建完加回 | 同左 | 不涉及 | 不涉及 | 不涉及 |

---

## 2. 相机外参与外参随机化

| 项 | Go2_PIE | MGDP（`sensor_config.Offset` 默认） | extreme-parkour | WMP | parkour-main |
|----|---------|-----------------------------------|-----------------|-----|--------------|
| 名义平移 (m) | `[0.345, 0, 0.07]` | `[0.34, 0, 0.07]` | `[0.27, 0, 0.03]` | 同左 | `forward_camera` 可配（含 15°/30° 俯视变体） |
| 名义旋转 | Euler deg `[0, 20, 0]` | `[0, 30, 0]` | 俯仰角均匀采样（见下） | 同左 | 配置上下界 |
| 平移随机 | 每 env：`U(min,max)`，默认 ±0.01 m | 默认 ±0.02 m（`TransRand`） | **无**（固定 position） | 无 | 可配 extrinsics 域随机 |
| 旋转随机 | 每 env：pitch 约 ±1°（`rot_rand`） | pitch 默认 ±2° | 创建时 **`angle ~ U(-5°, +5°)`** 一次采样，挂到 body | 同左 | latency / 位姿范围配置 |
| 随机生效时机 | `WarpManager` / sensor `_calcu_offset`，**初始化时采样，episode 内固定** | 同左 | `attach_camera` 创建时一次 | 同左 | 可按 resampling 时间重采样 latency |
| 跟随方式 | 每帧用 root 位姿 × offset 算相机（Warp） | 同左 | `FOLLOW_TRANSFORM` 挂 root | 同左 | Isaac attach |

**Go2_PIE 外参随机代码路径**：配置 → `WarpCameraCfg.offset.trans_rand/rot_rand` → Warp sensor `_calcu_offset`：

- `offset_p = translation + U(trans_min, trans_max)`（逐 env）
- `offset_q = quat(euler_base + U(rot_min, rot_max))`

---

## 3. 每步何时采图（调度）

| 项 | Go2_PIE | MGDP | extreme-parkour | WMP | parkour-main |
|----|---------|------|-----------------|-----|--------------|
| 计数器 | `global_counter`（`post_physics_step` 末尾 +1） | `global_counter`（`step` 末尾 +1） | `global_counter` | 同左 | `episode_length_buf` + latency |
| 刷新条件 | `global_counter % update_interval == 0` 才 `get_camera_sensor` | 同左 | `update_depth_buffer` 内同样 `% interval` | 同左 | `refresh_duration`；另有 **传感器 latency 缓冲** |
| 默认 interval | **5**（约 50Hz 控制 → 10Hz 深度） | **5** | **5** | **5** | 由 `refresh_duration / dt` 决定 |
| 非刷新步 | **复用** `image_buf` 旧内容（或 reset 后的 0） | 复用旧 `image_buf` | 不更新 `depth_buffer`；`extras["depth"]=None`（仅刷新步写出） | 同左 | 延迟帧索引可能仍推进 |

---

## 4. 渲染：原始深度如何得到

### 4.1 Warp（Go2_PIE / MGDP）

| 子步骤 | Go2_PIE | MGDP |
|--------|---------|------|
| 调用 | `warp_manager.warp_update_frame()` | 同左 |
| 内核 | `cam_kernel` 射线投射 | 同系 |
| 未命中 | 实现中 miss → **far_plane**（或历史版本写 `NO_HIT=1000`） | 同系；若未在 kernel 里改 far，可能先出大值再 clip |
| 输出张量 | `warp_manager['camera'].data` → squeeze → `(N, H, W)` 正距离 (m) | 同左 |
| 库内 ImageProcess | PIE 关闭；**米制图直接出** | 默认可 **clip + resize + normalize** 在库内完成后再给 env |

### 4.2 Isaac Gym（extreme-parkour / WMP / parkour-main）

| 子步骤 | extreme-parkour | WMP | parkour-main |
|--------|-----------------|-----|--------------|
| 图形步进 | `step_graphics` + `render_all_camera_sensors` | 同左 | 同 Isaac 管线 |
| 读图 | `get_camera_image_gpu_tensor(..., IMAGE_DEPTH)` | 同左 | `sensor_tensor_dict["forward_depth"]` |
| 符号 | Isaac 深度为**负**；后续 `* -1` | 同左 | `_process_depth_image` 里 `* -1` |
| 遍历 | **逐 env** for 循环 | 可只对 `depth_index` 子集 | 批量 stack |

---

## 5. 几何裁剪 / Resize（逐项目顺序）

### Go2_PIE（`get_camera_sensor`）

1. 得到 `(N, 60, 106)` 米制深度  
2. **左右切片**：`[..., crop_left : W-crop_right]`，默认各 10 → **`(N, 60, 86)`**  
3. **无** bilinear resize  

### MGDP（Warp process + env）

1. Warp pattern 出图（默认 120×120 量级）  
2. 库内 `ImageProcess`：可选 **clip** → 可选 **resize 到 `resized`（stage1 为 16×16）** → 可选库内 normalize（常开）  
3. env 层拿到的已是处理后尺寸；env **不再 crop**  

### extreme-parkour（`process_depth_image`）

1. **crop**：`depth[:-2, 4:-4]`（去底 2、左右各 4）  
2. 噪声（见下）  
3. `clip` 到 `[-far, -near]`（仍处负深度域时）  
4. **`Resize` → (58, 87)**（配置 `resized=(87,58)`，transform 为 (H,W)=(58,87)）  
5. normalize  

### WMP

与 extreme-parkour **同结构**，但当前代码里 **crop 与 Resize 被注释掉**，只保留噪声 + clip + normalize（以仓库现状为准）。

### parkour-main（`_process_depth_image`）

1. stack 后 `* -1` 得正深度  
2. 噪声族（contour / artifacts / stereo / sky）——在**米制**上  
3. **normalize** 到 [0,1]（按 `depth_range`）  
4. **crop**（`crop_top_bottom` / `crop_left_right`）  
5. **`T.Resize` → 网络输入分辨率**（如 48×64）  
6. `clip(0,1)`  

注意：**parkour-main 是先噪声+归一化，再 crop/resize**；extreme-parkour 是 **先 crop，再噪声，再 resize，再 normalize**。

---

## 6. 噪声与随机化（图像内容）

| 噪声 / 随机 | Go2_PIE | MGDP | extreme-parkour | WMP | parkour-main |
|-------------|---------|------|-----------------|-----|--------------|
| 逐像素高斯 | `d += N(0, 0.03)`（米制） | 同左（env 层） | 无 | 无 | 远距/近距不同 std（`stereo_*_noise_std`） |
| Dropout / 孔洞 | 像素以 p=0.1 乘 0 | 同左 | 无 | 无 | 轮廓处置 0；过近块伪影；半块 spark |
| 整图标量偏置 | 无 | 无 | `d += dis_noise*2*(U-0.5)`，默认 **dis_noise=0** | 同左 | 无此简式 |
| 立体远/近噪声 | 无 | 无 | 无 | 无 | 有：`_add_depth_stereo` |
| 天空伪影 | 无 | 无 | 无 | 无 | 有：`_add_sky_artifacts` |
| 矩形伪影块 | 无 | 无 | 无 | 无 | `_add_depth_artifacts` / full-block |
| 轮廓检测腐蚀 | 无 | 无 | 无 | 无 | `_add_depth_contour`（conv + max_pool） |
| 传感器延迟随机 | 无 | 无 | 无（仅深度刷新间隔） | 无 | **每 env** `latency ~ U(range)`，可按时间重采样 |
| 动作延迟（相关） | 可选 domain_rand delay | 有 | `action_delay` 课程 | 有 | `action_delay_range` |

**Go2_PIE / MGDP 噪声相对归一化**：先在**米制**加噪，再 normalize。  
dropout 后像素为 0 → 再 `clamp(near, far)` 时会被抬到 `near`（PIE `near=0.1`），归一化后变成「近处」而非无效标记。

---

## 7. 归一化公式（逐步）

| 项目 | 输入域 | 公式 | 输出域 |
|------|--------|------|--------|
| **Go2_PIE** | 米制（含噪） | `clamp(d, near, far) / far`，再 `clamp(0,1)`；**不减 near** | [0, 1] |
| **MGDP（env）** | 米制（含噪） | `(clamp(d,near,far)-near)/(far-near)` | [0, 1] |
| **MGDP（若库内 process.normalize）** | clip 后 | 库内 normalize（与 env 可能重复，取决于是否关掉库内） | [0, 1] |
| **extreme-parkour** | 负深度→取反 | `(d - near)/(far - near) - 0.5` | ≈ [-0.5, 0.5] |
| **WMP** | 同左 | 同左 | ≈ [-0.5, 0.5] |
| **parkour-main** | 米制噪后 | `_normalize_depth_images`（按 `depth_range` 映到 [0,1]） | [0, 1]，再 clip |

默认 near/far：

| | near | far |
|--|------|-----|
| Go2_PIE | 0.1 | 3.0 |
| MGDP stage1 | 0.0 | 5.0 |
| extreme / WMP | 0 | 2 |

策略侧二次处理：

- **Go2_PIE**：`ActorCriticPIE._encode_depth`：`image = depth.reshape(...) - 0.5`（假定缓冲已在 [0,1]）  
- **extreme-parkour**：可视化时 `depth_buffer + 0.5`；encoder 直接吃 [-0.5,0.5]  

---

## 8. 时间缓冲 / 帧栈 / Reset 行为

### 8.1 正常推进

| 项目 | 缓冲结构 | 推进方式 |
|------|----------|----------|
| Go2_PIE | `(N, 2, 60, 86)` | `cat([buf[:,1:], new.unsqueeze(1)], dim=1)` |
| MGDP | noisy + clean 各 `(N, 2, H, W)` | 同上，两路同步 |
| extreme-parkour | `(N, L, H, W)`，`L=buffer_len=2` | `cat([buf[i,1:], new], 0)` **按 env 循环** |
| WMP | 同左 | 同左 |
| parkour-main | `forward_depth_obs_buffer` 时间维 | 每步（或刷新）`cat([1:], processed)`；读出时用 **延迟下标** |

### 8.2 Reset / 新 episode

| 项目 | reset 时是否写 0 | 补帧策略 |
|------|------------------|----------|
| **Go2_PIE** | **`image_buf[env]=0`，`depth_flat_buf=0`** | 仅当本步执行了 `get_camera_sensor`：两帧都写成当前图；否则策略可能吃 **全 0** 数步 |
| **MGDP** | **不清零** | 采图时若 `reset_buf`：两帧均写当前（noisy/clean 各写） |
| **extreme-parkour** | 不清零 | 采图时若 `episode_length_buf <= 1`：整栈 `stack([frame]*L)` |
| **WMP** | 不清零 | 同 `episode_length_buf <= 1` |
| **parkour-main** | 延迟相关：`delayed_frames` 置 0 等 | 不整缓冲刷 0；靠 latency 缓冲重新灌入 |

### 8.3 非刷新步看到什么

| 项目 | 内容 |
|------|------|
| Go2_PIE | 上一刷新的栈；若刚 reset 且未刷新 → **0** |
| MGDP | 上一刷新的栈（含上一局可能残留，直到下次采图） |
| extreme-parkour | 缓冲不变；`step` 返回的 `extras["depth"]` 在非刷新步为 **None**（由 runner 决定是否复用） |
| parkour-main | 延迟索引指向历史帧，持续有输出 |

---

## 9. 图像如何进入策略（流转）

### Go2_PIE

```
image_buf (N,2,60,86)
    → reshape → depth_flat_buf (N, 2*60*86)
    → env.step / get_observations 第三项
    → PIERunner: policy(obs, hist, depth)
    → ActorCriticPIE.act:
         depth → (N,2,60,86) - 0.5
         → Conv CNN → tokens
         → 与 history_token 拼 → Transformer → GRU
         → velocity|map|foot|z → 与 proprio cat → actor MLP → action
```

### MGDP

```
image_buf (noisy) / image_clean_buf
    → 可选 World Model：encoder/decoder 更新，得到 visual_token
    → obs_dict['image_buf'] = token 或 原图
    → MGDP runner / 策略
```

### extreme-parkour / WMP

```
depth_buffer 更新（interval）
    → step 返回 extras["depth"] = depth_buffer[:, -2]  （取倒数第二帧）
    → 非刷新步 extras["depth"]=None
    → Runner 里 depth encoder / 与策略拼接（项目各异）
```

### parkour-main

```
sensor_tensor → _process_depth_image → 写入 forward_depth_obs_buffer
    → _get_forward_depth_obs：按 latency 取帧 → flatten
    → 观测向量组件 "forward_depth"
    → 学生/蒸馏策略
```

---

## 10. 逐步对照总表（同一逻辑步骤横向比）

| # | 逻辑步骤 | Go2_PIE | MGDP | extreme-parkour | WMP | parkour-main |
|---|----------|---------|------|-----------------|-----|--------------|
| 1 | 建相机 | Warp `_init_camera_buffers` | Warp 在 `__init__` | `attach_camera` | 同左 | Isaac sensor |
| 2 | 外参名义值 | 0.345/20° | 0.34/30° | 0.27 + 俯仰采样 | 同左 | 可配 |
| 3 | 外参随机 | trans±0.01, rot±1° | ±0.02 / ±2° | angle U(-5,5) 一次 | 同左 | latency+位姿范围 |
| 4 | 触发采图 | `% 5 == 0` | `% 5 == 0` | `% 5 == 0` | 同左 | refresh + latency |
| 5 | 渲染 | Warp 射线 | Warp | Isaac DEPTH | Isaac | Isaac |
| 6 | 符号处理 | 正距离 | 正距离 | ×(-1) | ×(-1) | ×(-1) |
| 7 | Crop | 左右 10 | 无（靠 resize） | 底2+左右4 | 代码中关闭 | TB/LR 配置 |
| 8 | Resize | 无 | 有（如 16×16） | 有（58×87） | 代码中关闭 | 有 |
| 9 | 内容噪声 | 高斯+dropout | 同左 | 标量 dis_noise(默认0) | 同左 | 立体全套 |
| 10 | 归一化 | `/far` → [0,1] | `(d-near)/(far-near)` | 减0.5 → [-0.5,0.5] | 同左 | [0,1] |
| 11 | 双缓冲干净图 | 无 | **有 clean** | 无 | 无 | 无 |
| 12 | 帧栈推进 | 2 帧滑窗 | 2 帧×2 路 | buffer_len 滑窗 | 同左 | 延迟缓冲 |
| 13 | Reset | **清 0** | 采图时覆写 | length≤1 整栈填 | 同左 | 延迟计数清 |
| 14 | 交策略 | flat depth | obs_dict / WM token | extras depth | 同左 | obs 组件 flatten |
| 15 | 网络入口 | −0.5 后 CNN | WM/CNN | depth encoder | WM | distill 网络 |

---

## 11. Go2_PIE 单项目流水线（逐步清单）

便于对照代码 `Go2_PIE.py` / `actor_critic_pie.py`：

1. **`_init_camera_buffers`**  
   - 读 `crop_*`、`noise_*`、`normalize`、`output_hw`  
   - 分配 `image_buf`  
   - 配置 Warp：60×106、FOV 87、near/far、offset+rand  
   - `process` 全关；创建 `WarpManager`  

2. **每个 policy step（`post_physics_step`）**  
   - 物理 → termination → reward → **`reset_idx`（可清零深度）** → `compute_observations` → `global_counter++`  

3. **`compute_observations` 深度段**  
   - 若 `use_camera` 且 `counter % update_interval == 0` → `get_camera_sensor()`  
   - 始终：`depth_flat_buf = image_buf.reshape(N, -1)`（未刷新则旧值或 0）  

4. **`get_camera_sensor`**  
   - Warp 渲染 → crop → 高斯 → dropout → `/far` 归一化 → 滑窗入栈 → reset 环境双帧填充  

5. **Runner / 策略**  
   - `depth_flat` → reshape `(2,60,86)-0.5` → CNN → … → action  

6. **外参随机**  
   - 仅初始化；训练中途不重采样（除非重建 WarpManager）  

---

## 12. 与「本仓库其它任务」的关系

| 任务 | 深度管线 |
|------|----------|
| Go2_PIE | 完整 Warp 深度 → 策略 |
| Go2_DreamWaQ | 无；CENet 仅用本体历史 |
| Go2_Cts / Go2_AMP_Cts | 无；特权高度图等，非相机深度 |
| play_pie | 强制 `use_camera`、可关噪声、`update_interval=1` 便于可视化 |

---

## 13. 代码锚点

| 项目 | 文件与函数 |
|------|------------|
| Go2_PIE | `Go2_PIE.py`：`_init_camera_buffers`, `get_camera_sensor`, `compute_observations`, `reset_idx`；`actor_critic_pie.py`：`_encode_depth` |
| Warp 库 | `legged_gym/warp_sensor/warp_manager.py`（offset 随机）；`warp_kernels/cam_kernel.py`（miss/far）；`warp_utils.py`（process clip/resize） |
| MGDP | `random_dog.py`：`get_camera_sensor`；`utils/sensor_config.py`：`Offset`/`ImageProcess`/`DepthCameraPattern` |
| extreme-parkour | `legged_robot.py`：`attach_camera`, `process_depth_image`, `normalize_depth_image`, `update_depth_buffer` |
| WMP | 同上结构（crop/resize 现状以文件为准） |
| parkour-main | `legged_robot_noisy.py`：`_process_depth_image`, `_add_depth_*`, `_get_forward_depth_obs`, latency buffers |

---

*若修改 Go2_PIE 深度逻辑，建议对照第 10 节总表与第 8 节 reset 行为，优先与 MGDP 对齐「不清零 + `(d-near)/(far-near)`」，或与 extreme-parkour 对齐「[-0.5,0.5] + episode_length 整栈填充」。*
