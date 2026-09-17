"""Go2_PIE play：10 机器人 + 头顶深度相机视场线 + matplotlib 深度画面。

操作：
  P  — 切换聚焦机器人（Isaac Gym 视角跟随 + 深度窗口显示该机画面）
  V  — 切换 viewer sync（BaseTask 原有）
  Esc — 退出

用法：
  cd /home/zju/My_unitree_go2_gym
  python legged_gym/scripts/play_pie.py --task go2_pie
"""
from __future__ import annotations

import math
import sys

import isaacgym  # noqa: F401 — 必须在 torch 前
from isaacgym import gymapi

import matplotlib
matplotlib.use("TkAgg")  # 本机 OpenCV highgui 无 GUI，与 play_dreamwaq 一致用 Tk
import matplotlib.pyplot as plt
import numpy as np
import torch
from isaacgym.torch_utils import quat_apply

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

NUM_ENVS = 10
FOV_NEAR = 0.15


def _load_policy_or_none(env, args, train_cfg):
    try:
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=env.device)
        print("[play_pie] 已加载策略:", train_cfg.runner.experiment_name)
        return policy
    except Exception as e:
        print(f"[play_pie] 未加载策略（将用零动作）: {e}")
        return None


def _camera_frustum_corners(pos, quat, hfov_deg, width, height, near, far, device):
    """视锥角点：必须与 warp kernel 一致 —— 相机局部系为 OpenCV 式 Z 朝前、X 右、Y 下。
    cam_quat 把该局部方向旋到世界系（见 cam_kernel: rd = quat_rotate(cam_quat, K_inv@(x,y,1))）。
    """
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov * 0.5) * (height / float(width)))
    corners = []
    for dist in (near, far):
        hw = dist * math.tan(hfov * 0.5)
        hh = dist * math.tan(vfov * 0.5)
        # (±X, ±Y, +Z_forward)；Y 下为正，故图像上方为 -hh
        local = torch.tensor(
            [
                [ hw, -hh, dist],
                [-hw, -hh, dist],
                [-hw,  hh, dist],
                [ hw,  hh, dist],
            ],
            device=device,
            dtype=torch.float32,
        )
        q = quat.unsqueeze(0).expand(4, -1)
        world = quat_apply(q, local) + pos.unsqueeze(0)
        corners.append(world)
    return corners[0], corners[1]


def _draw_fov(env, env_id, color=(0.1, 0.95, 0.2)):
    if env.viewer is None or not getattr(env, "use_camera", False):
        return
    cam_data = env.warp_manager["camera"]
    pos = cam_data.pos[env_id, 0].to(env.device)
    quat = cam_data.quat[env_id, 0].to(env.device)
    cfg = env.cfg.camera
    near_c, far_c = _camera_frustum_corners(
        pos,
        quat,
        cfg.horizontal_fov_deg,
        cfg.width,
        cfg.height,
        FOV_NEAR,
        float(cfg.far_clip),
        env.device,
    )
    origin = pos
    segs = []
    for i in range(4):
        segs.append((origin, far_c[i]))
        segs.append((near_c[i], far_c[i]))
        segs.append((near_c[i], near_c[(i + 1) % 4]))
        segs.append((far_c[i], far_c[(i + 1) % 4]))

    n = len(segs)
    verts = np.zeros((n * 2, 3), dtype=np.float32)
    colors = np.zeros((n, 3), dtype=np.float32)
    for i, (a, b) in enumerate(segs):
        verts[2 * i] = a.detach().cpu().numpy()
        verts[2 * i + 1] = b.detach().cpu().numpy()
        colors[i] = color

    # 光轴：局部 +Z，应指向相机正前方（略俯视地面）
    z_axis = quat_apply(quat.unsqueeze(0), torch.tensor([[0.0, 0.0, float(cfg.far_clip)]], device=env.device))[0]
    axis_end = origin + z_axis
    verts = np.vstack([verts, origin.detach().cpu().numpy(), axis_end.detach().cpu().numpy()])
    colors = np.vstack([colors, np.array([[0.2, 0.4, 1.0]], dtype=np.float32)])
    n += 1

    env.gym.clear_lines(env.viewer)
    env.gym.add_lines(env.viewer, env.envs[env_id], n, verts, colors)


class DepthViewer:
    """matplotlib 实时深度窗（避免 OpenCV highgui 无 GTK）。"""

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6.4, 4.0))
        self.fig.canvas.manager.set_window_title("Go2_PIE Depth (focused)")
        self.im = None
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()

    def update(self, env, env_id):
        if env.image_buf is None:
            return
        if env.image_buf.dim() == 4:
            img = env.image_buf[env_id, -1].detach().cpu().numpy()
        else:
            img = env.image_buf[env_id].detach().cpu().numpy()
        # image_buf：warp 深度经 near/far 归一化后的 [0,1]（0=近 0.1m，1=远 3m）
        vis = np.clip(img, 0.0, 1.0)
        title = (
            f"warp depth  robot {env_id}/{env.num_envs - 1}  [P]=next\n"
            f"normalized [0,1] = [{env.cfg.camera.near_clip},{env.cfg.camera.far_clip}] m"
        )
        if self.im is None:
            self.im = self.ax.imshow(vis, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
            self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
            self.cbar.set_label("0=near ... 1=far")
            self.ax.set_title(title)
        else:
            self.im.set_data(vis)
            self.ax.set_title(title)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self):
        plt.close(self.fig)


def _follow_robot(env, env_id, offset=(1.8, -1.8, 1.2)):
    root = env.root_states[env_id, :3].detach().cpu().numpy()
    cam_pos = root + np.array(offset, dtype=np.float64)
    env.set_camera(cam_pos, root)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = NUM_ENVS
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.commands.heading_command = False
    if hasattr(env_cfg, "camera"):
        env_cfg.camera.use_camera = True
        env_cfg.camera.update_interval = 1
        env_cfg.camera.noise_gaussian = 0.0
        env_cfg.camera.noise_dropout = 0.0

    args.headless = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_hist = env.get_observations()
    policy = _load_policy_or_none(env, args, train_cfg)

    focused = 0
    print(f"[play_pie] {env.num_envs} robots | 按 P 切换聚焦 | 当前 = {focused}")

    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_P, "next_robot")
    _orig_render = env.render

    def render_with_hotkeys(sync_frame_time=True):
        nonlocal focused
        if env.viewer is not None:
            for evt in env.gym.query_viewer_action_events(env.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit(0)
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    env.enable_viewer_sync = not env.enable_viewer_sync
                elif evt.action == "next_robot" and evt.value > 0:
                    focused = (focused + 1) % env.num_envs
                    print(f"[play_pie] focus -> robot {focused}")
                    _follow_robot(env, focused)
            if env.device != "cpu":
                env.gym.fetch_results(env.sim, True)
            if env.enable_viewer_sync:
                env.gym.step_graphics(env.sim)
                env.gym.draw_viewer(env.viewer, env.sim, True)
                if sync_frame_time:
                    env.gym.sync_frame_time(env.sim)
            else:
                env.gym.poll_viewer_events(env.viewer)
        else:
            _orig_render(sync_frame_time)

    env.render = render_with_hotkeys

    depth_viewer = DepthViewer()
    _follow_robot(env, focused)

    try:
        for i in range(50 * int(env.max_episode_length)):
            env.commands[:, 0] = 0.6
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0

            if policy is not None:
                actions = policy(obs.detach(), obs_hist.detach())
            else:
                actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

            step_out = env.step(actions.detach())
            obs, _, _, obs_hist, _, _, _ = step_out[:7]

            _draw_fov(env, focused)
            depth_viewer.update(env, focused)
            if i % 5 == 0:
                _follow_robot(env, focused)
    finally:
        depth_viewer.close()


if __name__ == "__main__":
    args = get_args()
    if args.task is None or args.task == "a1":
        args.task = "go2_pie"
    play(args)
