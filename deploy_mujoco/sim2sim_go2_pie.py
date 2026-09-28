"""Go2_PIE sim2sim（MuJoCo）：DreamWaQ 式 obs 滚动 + PIE 头部深度相机可视化。

- obs 按 PIE 布局：[角速度3, 重力3, 命令3, 关节角12, 关节速12, 动作12] = 45
- obs_hist：10 帧 × 45 滚动（deque，先推旧帧）
- depth：mujoco 离屏渲染 pie_head_depth 相机（106×60 → 裁 10/10 → 60×86，
  near 0.1 / far 3.0 归一化），双帧历史，matplotlib 实时窗口显示
- 策略可选：--load_model 指向导出的 jit；加载失败则用零动作（纯看相机）

用法（仓库根目录）:
    python deploy_mujoco/sim2sim_go2_pie.py [--load_model xxx.pt] [--terrain]
"""
from pie_policy_loader import load_pie_policy  # 必须最先导入（isaacgym 先于 torch）
from sim2sim_config_loader import load_cfg
Go2_PIE_Cfg_Yu = load_cfg(
    "legged_gym/envs/Go2_PIE/Go2_PIE_Config.py",
    "Go2_PIE_Cfg_Yu",
)
import math
import os
import signal
import numpy as np
import mujoco, mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["toolbar"] = "None"  # 规避 TkAgg 工具栏图标 0 尺寸崩溃
import matplotlib.pyplot as plt

from viewer_utils import VelocityArrowViewer

# PIE 相机参数（对齐 Go2_PIE_Config.camera）
CAM_W, CAM_H = 106, 60
CROP_L, CROP_R = 10, 10
CAM_NEAR, CAM_FAR = 0.1, 3.0
DEPTH_FRAMES = 2          # image_nums
DEPTH_UPDATE_INTERVAL = 5  # 每 5 个 policy 步刷新（50Hz -> 10Hz）
OBS_HIST = 10             # num_obs_hist


def get_obs(data, model):
    '''Extracts an observation from the mujoco data structure'''
    q = data.qpos[7:19].astype(np.double)
    dq = data.qvel[6:].astype(np.double)
    quat = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]]
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = r.apply(data.qvel[3:6], inverse=True).astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return q, dq, quat, v, omega, gvec


def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    '''Calculates torques from position commands'''
    return (target_q + cfg.robot_config.default_dof_pos - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg, policy_is_jit=False):
    vel_arrow = VelocityArrowViewer(max_cmd=(1.5, 0.5, 1.2),
                                    gears=[0.5, 1.0, 1.5])
    # 初始前进命令（对齐 play_pie 的 cmd_x 档位取 0.5）：
    # cmd 为 0 时 _add_arrow 渲染成 mjGEOM_NONE，箭头不可见
    vel_arrow.cmd[:] = [0.5, 0.0, 0.0]
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt

    _extent = float(model.stat.extent)  # vis.map.znear/zfar 以 stat.extent 为单位

    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos
    mujoco.mj_step(model, data)

    # --- 深度相机（离屏渲染）---
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'pie_head_depth')
    assert cam_id >= 0, "go2.xml 里没有 pie_head_depth 相机"
    renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)
    if hasattr(renderer, 'enable_depth_rendering'):   # mujoco >= 3.3 新 API
        renderer.enable_depth_rendering()
        render_depth = lambda: renderer.render()
    else:                                             # 旧 API: render(depth=True)
        render_depth = lambda: renderer.render(depth=True)

    def get_depth_img():
        """返回 (米制图[显示用], 归一化图[策略输入用])。
        显示给完整米制深度不截止；策略输入按训练约定 [0.1, 3.0] 归一化。"""
        # 仅深度渲染时收紧裁剪面提高深度精度，渲染完恢复，避免影响 viewer 视角
        _old_zn, _old_zf = model.vis.map.znear, model.vis.map.zfar
        model.vis.map.znear = CAM_NEAR / _extent
        model.vis.map.zfar = 6.0 / _extent
        renderer.update_scene(data, camera=cam_id)
        z = np.asarray(render_depth(), dtype=np.float32).copy()
        model.vis.map.znear, model.vis.map.zfar = _old_zn, _old_zf
        # render() 已返回米（平面深度 optical-z）且方向正确（row0=顶，col0=左），
        # 不做任何翻转/旋转；无效/超远像素置 far（背景）
        misses = ~np.isfinite(z) | (z <= 0.0) | (z >= 5.9)
        z[misses] = 6.0   # 背景置显示最远端（策略侧 clip 到 [0.1,3] 后仍为 1.0，不受影响）
        meters = z[:, CROP_L:CAM_W - CROP_R]                # (60, 86) 米
        # 对齐训练 Go2_PIE.get_camera_sensor:784 —— clamp(near, far) / far，不是 (z-near)/(far-near)
        norm = (np.clip(meters, CAM_NEAR, CAM_FAR) / CAM_FAR).clip(0.0, 1.0)
        return meters, norm

    # --- FOV 视锥线 + 相机红色标记（对齐 play_pie._draw_fov）---
    HFOV = math.radians(87.0)
    VFOV = 2.0 * math.atan(math.tan(HFOV * 0.5) * (CAM_H / float(CAM_W)))

    def _capsule(scn, a, b, rgba, width=0.008):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.eye(3).flatten(), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        scn.ngeom += 1

    def _sphere(scn, a, rgba, size=0.025):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, size),
                            a, np.eye(3).flatten(), rgba)
        scn.ngeom += 1

    def draw_cam_debug(viewer):
        """红色相机标记 + 视锥线（near 0.15 / far 3.0，与 play_pie 一致）。
        不清场（ngeom 不归零）：user_scn 里还有 vel_arrow 的速度箭头。"""
        scn = viewer.user_scn
        pos = data.cam_xpos[cam_id].copy()
        Rm = data.cam_xmat[cam_id].reshape(3, 3)
        right, up, forward = Rm[:, 0], Rm[:, 1], -Rm[:, 2]

        near_d, far_d = 0.15, CAM_FAR
        corners = []
        for dist in (near_d, far_d):
            hw = dist * math.tan(HFOV * 0.5)
            hh = dist * math.tan(VFOV * 0.5)
            for sx, sy in ((1, -1), (-1, -1), (-1, 1), (1, 1)):
                v = right * (sx * hw) + up * (sy * hh) + forward * dist
                corners.append(pos + v / np.linalg.norm(v) * dist)  # 每条锥线严格 dist 长
        near_c, far_c = corners[:4], corners[4:]

        green = (0.1, 0.95, 0.2, 0.6)
        red = (1.0, 0.1, 0.1, 1.0)
        blue = (0.2, 0.4, 1.0, 0.9)
        # 相机本体：红色球 + 红色光轴
        _sphere(scn, pos, red, size=0.03)
        _capsule(scn, pos, pos + forward * 0.15, red, width=0.006)
        # 视锥 4 条棱 + near/far 矩形
        for i in range(4):
            _capsule(scn, pos, far_c[i], green, width=0.003)
            _capsule(scn, near_c[i], far_c[i], green, width=0.003)
            _capsule(scn, near_c[i], near_c[(i + 1) % 4], green, width=0.003)
            _capsule(scn, far_c[i], far_c[(i + 1) % 4], green, width=0.003)
        # 光轴延长线（蓝）
        _capsule(scn, pos + forward * 0.15, pos + forward * far_d, blue, width=0.004)

    # --- 深度可视化窗口（单窗，go2 相机量程 [0.1, 3.0]m）---
    plt.ion()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    fig.canvas.manager.set_window_title("go2 PIE depth (pie_head_depth)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("depth (m)")
    import matplotlib.colors as mcolors
    im = ax.imshow(np.zeros((CAM_H, CAM_W - CROP_L - CROP_R)), cmap="gray",
                   norm=mcolors.LogNorm(vmin=0.1, vmax=6.0), origin="upper")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("meters")
    fig.tight_layout()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

        target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
        action = np.zeros((cfg.env.num_actions), dtype=np.double)

        # DreamWaQ 式 obs 滚动：10 帧 × 45
        hist_obs = deque()
        for _ in range(OBS_HIST):
            hist_obs.append(np.zeros([1, 45], dtype=np.double))
        depth_hist = deque(maxlen=DEPTH_FRAMES)      # 归一化，喂策略
        meters_last = np.zeros((CAM_H, CAM_W - CROP_L - CROP_R))  # 米制，仅显示
        for _ in range(DEPTH_FRAMES):
            depth_hist.append(np.zeros((CAM_H, CAM_W - CROP_L - CROP_R)))

        count_lowlevel = 1
        pie_hidden = torch.zeros(1, 1, 128)  # jit 模型的显式 GRU hidden
        policy_input = np.zeros([1, 45 * OBS_HIST], dtype=np.float32)
        depth_flat = np.zeros([1, DEPTH_FRAMES * CAM_H * (CAM_W - CROP_L - CROP_R)], dtype=np.float32)

        np.set_printoptions(formatter={'float': '{:0.4f}'.format})
        sim_step = 0
        import time as _time
        while True:
            step_start = _time.time()
            cmd = vel_arrow.update_cmd()
            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = cmd[0], cmd[1], cmd[2]

            q, dq, quat, v, omega, gvec = get_obs(data, model)

            if count_lowlevel % cfg.sim_config.decimation == 0:
                # --- PIE 45 维 obs ---
                obs = np.zeros([1, 45], dtype=np.float32)
                obs[0, 0:3] = omega * cfg.normalization.obs_scales.ang_vel
                obs[0, 3:6] = gvec
                obs[0, 6:9] = np.array([x_vel_cmd, y_vel_cmd, yaw_vel_cmd]) * np.array(
                    [cfg.normalization.obs_scales.lin_vel,
                     cfg.normalization.obs_scales.lin_vel,
                     cfg.normalization.obs_scales.ang_vel])
                obs[0, 9:21] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = action
                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                # DreamWaQ 式滚动：旧帧在前，最新帧在后
                hist_obs.append(obs)
                hist_obs.popleft()
                for i in range(OBS_HIST):
                    policy_input[0, i * 45:(i + 1) * 45] = hist_obs[i][0, :]

                # --- 深度刷新（10Hz）---
                if (count_lowlevel // cfg.sim_config.decimation) % DEPTH_UPDATE_INTERVAL == 0:
                    meters_last, _norm = get_depth_img()
                    depth_hist.append(_norm)
                for i in range(DEPTH_FRAMES):
                    depth_flat[0, i * CAM_H * 86:(i + 1) * CAM_H * 86] = depth_hist[i].reshape(-1)

                # --- 策略（三输入：obs, obs_hist, depth）---
                if policy is not None:
                    try:
                        if policy_is_jit:
                            a, pie_hidden = policy(torch.tensor(obs), torch.tensor(policy_input),
                                                    torch.tensor(depth_flat), pie_hidden)
                        else:
                            a, _ = policy.act_inference(torch.tensor(obs), torch.tensor(policy_input), torch.tensor(depth_flat))
                        action[:] = a[0].detach().numpy()
                    except Exception as e:
                        if sim_step % 500 == 0:
                            print(f"[policy] 调用失败，用零动作: {e}")
                        action[:] = 0.0
                else:
                    action[:] = 0.0
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                target_q = action * cfg.control.action_scale

                # 深度窗口刷新（最新一帧）+ RGB 对照
                im.set_data(meters_last)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            if sim_step < 100:
                tau = pd_control(np.zeros((cfg.env.num_actions)), q, cfg.robot_config.kps,
                                 target_dq, dq, cfg.robot_config.kds, cfg)
            else:
                tau = pd_control(target_q, q, cfg.robot_config.kps,
                                 target_dq, dq, cfg.robot_config.kds, cfg)
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)

            data.ctrl = tau
            mujoco.mj_step(model, data)

            vel_arrow.draw_arrows(viewer, data)
            draw_cam_debug(viewer)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (_time.time() - step_start)
            if time_until_next_step > 0:
                _time.sleep(time_until_next_step)
            count_lowlevel += 1
            sim_step += 1

    renderer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Go2_PIE sim2sim deployment.')
    parser.add_argument('--load_model', type=str,
                        default=f'{LEGGED_GYM_ROOT_DIR}/logs/go2_pie/exported/policies/policy_pie.pt',
                        help='训练 checkpoint（raw .pt，经 ActorCriticPIE 加载）。')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    scene_xml = 'scene_terrain.xml' if args.terrain else 'scene.xml'

    class Sim2simCfg(Go2_PIE_Cfg_Yu):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/go2/{scene_xml}'
            sim_duration = 12200000000.0
            dt = 0.005
            decimation = 4

        class robot_config:
            kps = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], dtype=np.double)
            kds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.double)
            tau_limit = 45 * np.ones(12, dtype=np.double)
            default_dof_pos = np.array([0.1, 0.8, -1.5,    # FL
                                        -0.1, 0.8, -1.5,   # FR
                                         0.1, 1.0, -1.5,   # RL
                                        -0.1, 1.0, -1.5], dtype=np.double)  # RR

    # 策略：优先 jit 导出模型（四输入含显式 hidden）；否则按 raw checkpoint 加载
    policy = None
    policy_is_jit = False
    if os.path.exists(args.load_model):
        try:
            policy = torch.jit.load(args.load_model, map_location='cpu').eval()
            policy_is_jit = True
            print(f"[sim2sim_pie] 已加载 jit 策略(显式hidden): {args.load_model}")
        except Exception:
            try:
                policy = load_pie_policy(args.load_model)
            except Exception as e:
                print(f"[sim2sim_pie] 策略加载失败（用零动作）: {e}")
    else:
        print(f"[sim2sim_pie] 未找到 {args.load_model}，用零动作，仅看相机")

    # Ctrl+C 立即退出，不等 viewer 回滚
    exited = {'done': False}
    def _save_now_and_exit(signum=None, frame=None):
        if not exited['done']:
            exited['done'] = True
            print('\n[退出] 收到 Ctrl+C，立即退出...')
        os._exit(0)
    signal.signal(signal.SIGINT, _save_now_and_exit)

    run_mujoco(policy, Sim2simCfg(), policy_is_jit)
