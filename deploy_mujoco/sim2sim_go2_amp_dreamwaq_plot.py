"""sim2sim_go2_amp_dreamwaq 实时曲线版：
一个窗口内 3 个等尺寸子图实时滚动 —— ①左前腿(FL)关节角度 ②FL关节角速度 ③基座欧拉角。
坐标轴无标题、横坐标隐藏、纵轴与图例中文；欧拉角连续化（不出现 ±π 跳变）。
Ctrl+C 退出时立刻保存 3 张 PNG 并强制退出（不等 viewer / 弹窗回滚）。
"""
from sim2sim_config_loader import load_cfg
Go2_AMP_DreamWaQ_Cfg_Yu = load_cfg(
    "legged_gym/envs/Go2_AMP_DreamWaQ/Go2_AMP_DreamWaQ_Config.py",
    "Go2_AMP_DreamWaQ_Cfg_Yu",
)
import os
import signal
import math
import numpy as np
import mujoco, mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import time
from viewer_utils import VelocityArrowViewer

import matplotlib
matplotlib.use('TkAgg')  # 本机 Qt 的 xcb 插件不可用，强制 Tk 后端
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体回退（按本机可用性）
for _f in ['Noto Serif CJK SC', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'AR PL UMing TW MBE', 'SimHei']:
    if any(_f == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [_f]
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 13            # 全局中文字体加大

x_vel_max, y_vel_max, yaw_vel_max = 1.5, 1.0, 1.5


def quaternion_to_euler_array(quat):
    x, y, z, w = quat
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])


class EulerUnwrapper:
    """欧拉角连续化：新值与上个连续值比较，跨 ±π 时累计 ±2π 偏移，消除 -3.14↔3.14 跳变。"""

    def __init__(self):
        self.offset = np.zeros(3)
        self.last = None

    def __call__(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        if self.last is None:
            self.last = raw.copy()
            return raw.copy()
        cand = raw + self.offset
        for i in range(3):
            d = cand[i] - self.last[i]
            if d > np.pi:
                self.offset[i] -= 2 * np.pi
            elif d < -np.pi:
                self.offset[i] += 2 * np.pi
        out = raw + self.offset
        self.last = out.copy()
        return out


class ThreePanelPlotter:
    """1×3 等尺寸子图实时滚动；无标题、横坐标隐藏、纵轴中文单位、图例中文。
    Ctrl+C 后三个子图各自另存为同尺寸 PNG（共 3 张）。"""

    SAVE_FILES = ['fl_joint_angles.png', 'fl_joint_velocities.png', 'base_euler_angles.png']
    YLABELS = ['角度（弧度）', '角速度（弧度/秒）', '角度（弧度）']
    LEGENDS = [['髋关节', '大腿关节', '小腿关节'],
               ['髋关节', '大腿关节', '小腿关节'],
               ['横滚角', '俯仰角', '偏航角']]

    def __init__(self, dt=0.02, window_s=10.0, refresh_every=10, save_dir='.'):
        self.refresh_every = refresh_every
        self.max_points = max(2, int(window_s / dt))
        self.save_dir = save_dir
        self._count = 0
        self.t_buf = deque(maxlen=self.max_points)
        self.bufs = [deque(maxlen=self.max_points) for _ in range(3)]

        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(11, 8))  # 3 行 1 列
        try:
            self.fig.canvas.manager.set_window_title('Go2 实时监测：左前腿关节与基座姿态')
        except Exception:
            pass
        self.lines = []
        for ax, ylab, legend in zip(self.axes, self.YLABELS, self.LEGENDS):
            ls = []
            for lbl in legend:
                line, = ax.plot([], [], linewidth=1.2, label=lbl)
                ls.append(line)
            self.lines.append(ls)
            ax.set_ylabel(ylab)                # 纵坐标中文加单位
            ax.grid(True, alpha=0.35)
            ax.legend(loc='upper right', fontsize=13)
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    def update(self, t, fl_angles, fl_vels, euler_cont):
        self.t_buf.append(float(t))
        self.bufs[0].append(np.asarray(fl_angles, dtype=np.float64).copy())
        self.bufs[1].append(np.asarray(fl_vels, dtype=np.float64).copy())
        self.bufs[2].append(np.asarray(euler_cont, dtype=np.float64).copy())
        self._count += 1
        if self._count % self.refresh_every != 0:
            return
        ts = np.asarray(self.t_buf)
        for i in range(3):
            data = np.asarray(self.bufs[i])
            for line, col in zip(self.lines[i], data.T):
                line.set_data(ts, col)
            self.axes[i].relim()
            self.axes[i].autoscale_view()
        plt.pause(0.001)

    def save_three(self):
        """把三个子图分别保存为 3 张同尺寸 PNG（不阻塞等待窗口）。"""
        if len(self.t_buf) == 0:
            print('[保存] 无数据，跳过')
            return
        ts = np.asarray(self.t_buf)
        try:
            plt.ioff()
            plt.close(self.fig)
        except Exception:
            pass
        for i, (fname, ylab, legend) in enumerate(zip(self.SAVE_FILES, self.YLABELS, self.LEGENDS)):
            fig, ax = plt.subplots(figsize=(9, 5))
            data = np.asarray(self.bufs[i])
            if data.size == 0:
                plt.close(fig)
                continue
            for col, lbl in zip(data.T, legend):
                ax.plot(ts, col, linewidth=1.2, label=lbl)
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.35)
            ax.legend(loc='upper right', fontsize=13)
            fig.tight_layout()
            out = os.path.join(self.save_dir, fname)
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f'[保存] {out}')


def get_obs(data, model):
    q = data.qpos[7:19].astype(np.double)
    dq = data.qvel[6:].astype(np.double)
    quat = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]]
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)
    omega = data.qvel[3:6].astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    base_pos = data.qpos[0:3].astype(np.double)
    foot_positions = []
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if 'foot' in body_name:
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces = data.cfrc_ext[i][2].copy().astype(np.double)
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)


def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    torque_out = (target_q + cfg.robot_config.default_dof_pos - q) * kp + (target_dq - dq) * kd
    return torque_out


def run_mujoco(policy, cfg):
    vel_arrow = VelocityArrowViewer(max_cmd=(x_vel_max, y_vel_max, yaw_vel_max),
                                    gears=[0.5, 1.0, 1.5, 2.0])
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos
    mujoco.mj_step(model, data)

    plotter = ThreePanelPlotter(dt=cfg.sim_config.dt * cfg.sim_config.decimation)
    euler_unwrap = EulerUnwrapper()
    saved = {'done': False}

    def _save_now_and_exit(signum=None, frame=None):
        """Ctrl+C：立刻存图并强制退出，不等 viewer / matplotlib 回滚清理。"""
        if not saved['done']:
            saved['done'] = True
            print('\n[退出] 立即保存曲线...')
            try:
                plotter.save_three()
            except Exception as e:
                print(f'[保存失败] {e}')
        os._exit(0)

    signal.signal(signal.SIGINT, _save_now_and_exit)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    hist_obs = deque()
    for _ in range(6):
        hist_obs.append(np.zeros([1, 45], dtype=np.double))
    count_lowlevel = 1
    sim_step = 0
    policy_input = np.zeros([1, cfg.env.num_observations * 6], dtype=np.float32)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

        while True:
            step_start = time.time()
            cmd = vel_arrow.update_cmd()
            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = cmd[0], cmd[1], cmd[2]

            q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data, model)

            if count_lowlevel % cfg.sim_config.decimation == 0:
                obs = np.zeros([1, 45], dtype=np.float32)
                obs[0, 0] = x_vel_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 1] = y_vel_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 2] = yaw_vel_cmd * cfg.normalization.obs_scales.ang_vel
                obs[0, 3:6] = omega * cfg.normalization.obs_scales.ang_vel
                obs[0, 6:9] = gvec
                obs[0, 9:21] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = action
                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                hist_obs.append(obs)
                hist_obs.popleft()
                for i in range(6):
                    policy_input[0, i * 45: (i + 1) * 45] = hist_obs[i][0, :]
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                target_q = action * cfg.control.action_scale

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
            viewer.sync()

            # 实时曲线：FL 前 3 个关节（髋/大腿/小腿），欧拉角经连续化
            euler_raw = quaternion_to_euler_array(quat)
            euler_cont = euler_unwrap(euler_raw)
            plotter.update(data.time, q[0:3], dq[0:3], euler_cont)

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            count_lowlevel += 1
            sim_step += 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AMP DreamWaQ sim2sim with 3-panel live plot')
    parser.add_argument('--load_model', type=str,
                        default=f'{LEGGED_GYM_ROOT_DIR}/logs/go2_amp_dreamwaq/exported/policies/policy_dwaq.pt')
    parser.add_argument('--terrain', action='store_true', help='stairs terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(Go2_AMP_DreamWaQ_Cfg_Yu):
        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/go2/cross_stairs.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/go2/scene.xml'
            sim_duration = 120.0
            dt = 0.005
            decimation = 4

        class robot_config:
            kps = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], dtype=np.double)
            kds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.double)
            tau_limit = 45 * np.ones(12, dtype=np.double)
            default_dof_pos = np.array([0.1, 0.8, -1.5,
                                        -0.1, 0.8, -1.5,
                                        0.1, 1.0, -1.5,
                                        -0.1, 1.0, -1.5], dtype=np.double)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
