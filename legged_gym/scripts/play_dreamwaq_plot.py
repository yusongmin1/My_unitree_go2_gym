"""isaacgym 版 play_dreamwaq：单机器人 + 摄像头跟随 + matplotlib 实时双曲线。

子图 1：世界系 z 轴线速度（root_states[:, 8]）
子图 2：_get_base_heights 计算的高度

用法：
    python legged_gym/scripts/play_dreamwaq_plot.py --task go2_stairs_dreamwaq
    python legged_gym/scripts/play_dreamwaq_plot.py --task go2_amp_dreamwaq

    下楼梯的高度竟然是在0.4m左右，和实际高度不符，可能是计算方法有问题。
"""
import os
import sys
import numpy as np
from collections import deque

import isaacgym  # noqa: F401  # 必须在 torch 之前导入
import torch

import matplotlib
matplotlib.use('TkAgg')  # Qt 的 xcb 插件在本机不可用，强制 Tk 后端
import matplotlib.pyplot as plt

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


class RealtimePlotter:
    """实时双子图：世界系 z 轴线速度 + _get_base_heights 高度"""

    def __init__(self, dt, window_s=15.0):
        self.max_points = max(2, int(window_s / dt))
        self.t_buf = deque(maxlen=self.max_points)
        self.vz_buf = deque(maxlen=self.max_points)
        self.h_buf = deque(maxlen=self.max_points)

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        try:
            self.fig.canvas.manager.set_window_title('DreamWaQ Play (real-time)')
        except Exception:
            pass
        self.line_vz, = self.ax1.plot([], [], color='tab:blue', linewidth=1.2, label='world lin_vel_z')
        self.ax1.set_ylabel('lin vel z [m/s]')
        self.ax1.grid(True, alpha=0.4)
        self.ax1.legend(loc='upper right')
        self.line_h, = self.ax2.plot([], [], color='tab:orange', linewidth=1.2, label='_get_base_heights')
        self.ax2.set_ylabel('height [m]')
        self.ax2.set_xlabel('time [s]')
        self.ax2.grid(True, alpha=0.4)
        self.ax2.legend(loc='upper right')
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    def update(self, t, vz, height):
        self.t_buf.append(float(t))
        self.vz_buf.append(float(vz))
        self.h_buf.append(float(height))
        ts = np.asarray(self.t_buf)
        self.line_vz.set_data(ts, np.asarray(self.vz_buf))
        self.line_h.set_data(ts, np.asarray(self.h_buf))
        for ax in (self.ax1, self.ax2):
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.001)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # 只留一个机器人
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.heading_command = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_hist, _ = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    plotter = RealtimePlotter(dt=env.dt)
    camera_offset = np.array([2.0, 0.0, 0.5])  # 相机固定在机器人侧面 2m 处，只平移跟随

    for it in range(int(20 * env.max_episode_length)):
        env.commands[:, 0] = 1.0
        env.commands[:, 1] = 0.
        env.commands[:, 2] = 0.0

        actions = policy(obs.detach(), obs_hist.detach())
        # stairs_dreamwaq 返回 7 个值，amp_dreamwaq 返回 9 个，取前 7 个兼容两者
        obs, _, obs_hist, _, rews, dones, infos = env.step(actions.detach())[:7]

        # 摄像头盯着机器人
        lookat = env.root_states[0, :3].cpu().numpy()
        env.set_camera(lookat + camera_offset, lookat)

        # 世界系 z 轴线速度 + _get_base_heights 高度
        vz = env.root_states[0, 8].item()
        if hasattr(env, '_get_base_heights'):
            height = env._get_base_heights()[0].item()
        else:  # 平地退化：直接用基座 z
            height = env.root_states[0, 2].item()
        plotter.update(it * env.dt, vz, height)


if __name__ == '__main__':
    args = get_args()
    play(args)
