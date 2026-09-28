"""Go2_PIE play：策略推理 + 状态/奖励日志 + 深度 FOV 可视化。

操作：
  P   — 切换聚焦机器人（视角跟随 + 深度窗口）
  V   — 切换 viewer sync
  Esc — 退出

用法：
  cd /home/zju/My_unitree_go2_gym
  python legged_gym/scripts/play_pie.py --task go2_pie
"""
from __future__ import annotations

import math
import os
import sys

import isaacgym  # noqa: F401 — 必须在 torch 前
from isaacgym import gymapi

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["toolbar"] = "None"  # 规避 TkAgg 工具栏图标 0 尺寸崩溃
import matplotlib.pyplot as plt
import numpy as np
import torch
from isaacgym.torch_utils import quat_apply

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, Logger

NUM_ENVS = 50
FOV_NEAR = 0.15
EXPORT_POLICY = False
RECORD_FRAMES = False
MOVE_CAMERA = False


def _load_depth_policy(env, args, train_cfg):
    """Load PIE depth policy (ActorCriticPIE via PIERunner). Fail hard if not PIE."""
    runner_name = train_cfg.runner_class_name
    if runner_name != "PIERunner":
        raise RuntimeError(
            f"[play_pie] expect PIERunner depth policy, got runner_class_name={runner_name!r}. "
            "Use --task go2_pie."
        )
    if not getattr(env.cfg.camera, "use_camera", False):
        raise RuntimeError("[play_pie] camera.use_camera must be True to load depth policy.")

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    if type(ppo_runner).__name__ != "PIERunner":
        raise RuntimeError(
            f"[play_pie] alg runner is {type(ppo_runner).__name__}, expected PIERunner."
        )
    if not hasattr(ppo_runner.alg.actor_critic, "depth_encoder"):
        raise RuntimeError(
            "[play_pie] checkpoint has no depth_encoder — not a PIE depth policy."
        )

    policy = ppo_runner.get_inference_policy(device=env.device)
    print(
        "[play_pie] loaded PIE depth policy:",
        train_cfg.runner.experiment_name,
        "| actor=",
        type(ppo_runner.alg.actor_critic).__name__,
    )
    _export_sim2sim_jit(env, ppo_runner, train_cfg)
    return policy, ppo_runner, train_cfg


def _export_sim2sim_jit(env, ppo_runner, train_cfg):
    """导出 sim2sim 用完整 jit：三输入(obs/obs_hist/depth) + 显式 GRU hidden。
    通用 export_policy_as_jit 只导裸 actor MLP（单输入），对 PIE 不可用。"""
    import copy
    try:
        ac = copy.deepcopy(ppo_runner.alg.actor_critic).to("cpu").eval()
        ac.init_hidden(1, "cpu")
        hdim = int(ac.memory_hidden.shape[-1])

        class _JitWrapper(torch.nn.Module):
            def __init__(self, ac):
                super().__init__()
                self.ac = ac

            def forward(self, obs, obs_hist, depth, hidden):
                est, new_hidden = self.ac._estimate_tensors(
                    obs, obs_hist, depth,
                    hidden_state=hidden, update_internal=False, sample_latent=False,
                )
                return self.ac.actor(torch.cat([obs, est["latent"]], dim=-1)), new_hidden

        c = env.cfg.camera
        zeros = (torch.zeros(1, env.num_obs),
                 torch.zeros(1, env.cfg.env.num_history_obs),
                 torch.zeros(1, int(c.image_nums) * int(c.output_hw[0]) * int(c.output_hw[1])),
                 torch.zeros(1, 1, hdim))
        with torch.no_grad():
            traced = torch.jit.trace(_JitWrapper(ac), zeros, strict=False)
        out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name,
                               "exported", "policies")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "policy_pie.pt")
        traced.save(out)
        print("[play_pie] 已导出 sim2sim jit(显式hidden) ->", out)
    except Exception as e:
        print("[play_pie] sim2sim jit 导出失败:", e)


def _camera_frustum_corners(pos, quat, hfov_deg, width, height, near, far, device):
    """视锥角点：与 warp kernel 一致（Z 朝前、X 右、Y 下）。"""
    hfov = math.radians(hfov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov * 0.5) * (height / float(width)))
    corners = []
    for dist in (near, far):
        hw = dist * math.tan(hfov * 0.5)
        hh = dist * math.tan(vfov * 0.5)
        local = torch.tensor(
            [
                [hw, -hh, dist],
                [-hw, -hh, dist],
                [-hw, hh, dist],
                [hw, hh, dist],
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
    if not hasattr(env, "warp_manager") or env.warp_manager is None:
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

    z_axis = quat_apply(
        quat.unsqueeze(0),
        torch.tensor([[0.0, 0.0, float(cfg.far_clip)]], device=env.device),
    )[0]
    axis_end = origin + z_axis
    verts = np.vstack(
        [verts, origin.detach().cpu().numpy(), axis_end.detach().cpu().numpy()]
    )
    colors = np.vstack([colors, np.array([[0.2, 0.4, 1.0]], dtype=np.float32)])
    n += 1

    env.gym.clear_lines(env.viewer)
    env.gym.add_lines(env.viewer, env.envs[env_id], n, verts, colors)


class DepthViewer:
    """matplotlib 实时深度窗。"""

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6.4, 4.0))
        self.fig.canvas.manager.set_window_title("Go2_PIE Depth (focused)")
        self.im = None
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()

    def update(self, env, env_id):
        if getattr(env, "image_buf", None) is None:
            return
        if env.image_buf.dim() == 4:
            img = env.image_buf[env_id, -1].detach().cpu().numpy()
        else:
            img = env.image_buf[env_id].detach().cpu().numpy()
        vis = np.clip(img, 0.0, 1.0)
        title = (
            f"warp depth  robot {env_id}/{env.num_envs - 1}  [P]=next\n"
            f"normalized [0,1] = [{env.cfg.camera.near_clip},{env.cfg.camera.far_clip}] m"
        )
        if self.im is None:
            self.im = self.ax.imshow(
                vis, cmap="gray", vmin=0.0, vmax=1.0, origin="upper"
            )
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

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, NUM_ENVS)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
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
    obs, obs_hist, depth = env.get_observations()
    if depth is None:
        raise RuntimeError("[play_pie] depth observation is None; cannot run depth policy.")

    policy, ppo_runner, train_cfg = _load_depth_policy(env, args, train_cfg)

    with torch.no_grad():
        actions0 = policy(obs.detach(), obs_hist.detach(), depth.detach())
    a0 = actions0[0].detach().cpu().numpy()
    print(
        f"[play_pie] env0 initial actions (12): "
        f"{np.array2string(a0, precision=4, floatmode='fixed')}",
        flush=True,
    )

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 1000
    stop_rew_log = env.max_episode_length + 1
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    focused = 0
    print(f"[play_pie] {env.num_envs} robots | P=next focus | focused={focused}")

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
        for i in range(10 * int(env.max_episode_length)):
            env.commands[:, 0] = 0.7
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0

            actions = policy(obs.detach(), obs_hist.detach(), depth.detach())

            step_out = env.step(actions.detach())
            # PIE: obs, privileged, obs_hist, depth, pie_targets, rew, reset, extras
            obs, _, obs_hist, depth, _, rews, dones, infos = step_out

            _draw_fov(env, focused)
            depth_viewer.update(env, focused)
            if not MOVE_CAMERA and i % 5 == 0:
                _follow_robot(env, focused)

            if RECORD_FRAMES and i % 2 == 0 and train_cfg is not None:
                filename = os.path.join(
                    LEGGED_GYM_ROOT_DIR,
                    "logs",
                    train_cfg.runner.experiment_name,
                    "exported",
                    "frames",
                    f"{img_idx}.png",
                )
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1

            if MOVE_CAMERA:
                camera_position += camera_vel * env.dt
                env.set_camera(camera_position, camera_position + camera_direction)

            if i < stop_state_log:
                logger.log_states(
                    {
                        "dof_pos_target": actions[robot_index, joint_index].item()
                        * env.cfg.control.action_scale,
                        "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                        "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                        "dof_torque": env.torques[robot_index, joint_index].item(),
                        "command_x": env.commands[robot_index, 0].item(),
                        "command_y": env.commands[robot_index, 1].item(),
                        "command_yaw": env.commands[robot_index, 2].item(),
                        "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                        "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                        "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                        "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                        "contact_forces_z": env.contact_forces[
                            robot_index, env.feet_indices, 2
                        ]
                        .cpu()
                        .numpy(),
                    }
                )
            elif i == stop_state_log:
                logger.plot_states()

            if 0 < i < stop_rew_log:
                if infos.get("episode"):
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes > 0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i == stop_rew_log:
                logger.print_rewards()
    finally:
        depth_viewer.close()


if __name__ == "__main__":
    args = get_args()
    if args.task is None or args.task == "a1":
        args.task = "go2_pie"
    play(args)
