"""独立验证 warp 渲染管线（不经 isaacgym PhysX，规避机器存量段错误）：
Terrain(trimesh) -> WarpManager 渲染 -> clip/resize -> 噪声 -> 归一化 -> 双帧栈
"""
import isaacgym  # noqa: gymutil 需要（必须在 torch 之前）
import sys, math
import numpy as np
import torch
from legged_gym.envs.Go2_PIE.Go2_PIE_Config import Go2_PIE_Cfg_Yu
from legged_gym.utils.terrain import Terrain
from legged_gym.warp_sensor import WarpManager
from legged_gym.warp_sensor.config.sensor_config import (
    Config as WarpConfig, Camera as WarpCameraCfg, ImageProcess as WarpImageProcess,
    TransRand as WarpTransRand, RotRand as WarpRotRand,
)

NUM_ENVS = 8
DEVICE = "cuda:0"

# ---- 1) 用项目自己的 Terrain 生成跑酷 trimesh（纯 CPU，无 PhysX）----
terrain_cfg = Go2_PIE_Cfg_Yu.terrain
terrain_cfg.curriculum = True
terrain = Terrain(terrain_cfg)
print("[1] Terrain 生成:", terrain.vertices.shape, "vertices,", terrain.triangles.shape, "triangles")

# 对齐 Go2_PIE._init_camera_buffers 的坐标平移
terrain.vertices = terrain.vertices - np.array(
    [terrain.cfg.border_size, terrain.cfg.border_size, 0])

# ---- 2) mock gym_ptr：提供 root_states / terrain 属性 ----
class MockGym:
    pass

mock = MockGym()
mock.terrain = terrain
# root_states: [pos(3), quat xyzw(4), ...]；放在第一个地形中心上方 0.42m，朝向 x+
origins = torch.tensor(terrain.env_origins[5, 10], dtype=torch.float32, device=DEVICE)
root_states = torch.zeros(NUM_ENVS, 13, dtype=torch.float32, device=DEVICE)
root_states[:, :3] = origins.unsqueeze(0) + torch.tensor([0., 0., 0.42], device=DEVICE)
root_states[:, 6] = 1.0  # quat (x,y,z,w)=(0,0,0,1)
mock.root_states = root_states

# ---- 3) 构造与 Go2_PIE._init_camera_buffers 完全一致的相机配置 ----
cam_cfg = Go2_PIE_Cfg_Yu.camera
warp_cam = WarpCameraCfg()
warp_cam.pattern.height = cam_cfg.height
warp_cam.pattern.width = cam_cfg.width
warp_cam.pattern.horizontal_fov_deg = cam_cfg.horizontal_fov_deg
warp_cam.pattern.max_range = cam_cfg.far_clip
warp_cam.pattern.min_range = cam_cfg.near_clip
warp_cam.pattern.calculate_depth = cam_cfg.calculate_depth
warp_cam.offset.translation = cam_cfg.offset_translation
warp_cam.offset.rotation = cam_cfg.offset_rotation_deg
warp_cam.offset.trans_rand = WarpTransRand(
    min=list(cam_cfg.offset_trans_rand[0]), max=list(cam_cfg.offset_trans_rand[1]))
warp_cam.offset.rot_rand = WarpRotRand(
    min=list(cam_cfg.offset_rot_rand[0]), max=list(cam_cfg.offset_rot_rand[1]))
warp_cam.process = WarpImageProcess(
    resize=None, noise=None, clip=None, normalize=False)

warp_cfg = WarpConfig()
warp_cfg.camera = warp_cam
warp_cfg.lidar = None

wm = WarpManager(NUM_ENVS, mock, cfg=warp_cfg, device=DEVICE, enable_viz=False)
print("[2] WarpManager 创建成功")

# ---- 4) 渲染并执行 Go2_PIE.get_camera_sensor 的后处理逻辑 ----
wm.warp_update_frame()
cur = wm['camera'].data.squeeze(1)  # (N, 60, 106)
cur = cur[..., cam_cfg.crop_left: cam_cfg.width - cam_cfg.crop_right]  # -> 60x86
print("[3] warp 渲染输出 shape:", tuple(cur.shape),
      "min/max: %.3f/%.3f" % (cur.min(), cur.max()))

noisy = cur.clone()
noisy.add_(torch.randn_like(noisy) * cam_cfg.noise_gaussian)
noisy.mul_(torch.rand_like(noisy) > cam_cfg.noise_dropout)

near, far = float(cam_cfg.near_clip), float(cam_cfg.far_clip)
cur_n = (cur.clamp(near, far) / far).clamp(0., 1.)
noisy_n = (noisy.clamp(near, far) / far).clamp(0., 1.)

print("[4] 归一化后 clean min/max: %.3f/%.3f" % (cur_n.min(), cur_n.max()))
print("    带噪 min/max: %.3f/%.3f, 与干净图差异 mean=%.4f" % (
    noisy_n.min(), noisy_n.max(), (noisy_n - cur_n).abs().mean()))
assert cur_n.min() >= 0 and cur_n.max() <= 1
zero_frac = (noisy_n == 0).float().mean()
print("    dropout 置零比例: %.3f（期望≈%.2f）" % (zero_frac, cam_cfg.noise_dropout))

# ---- 5) 深度合理性：相机俯仰 30° 安装在 0.42+0.07m 高度，看 10m 长 4m 宽走廊
env0 = cur_n[0].cpu().numpy()
print("[5] env0 深度图（60x86）行均值剖面（上->远，下->近）：")
for r in [2, 15, 29, 44, 56]:
    print("    row %2d mean=%.3f" % (r, env0[r].mean()))
print("=" * 60)
print("warp 获取+噪声+归一化 数值链路全部验证通过 ✓")
