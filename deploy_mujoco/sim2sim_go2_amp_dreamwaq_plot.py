"""sim2sim_go2_amp_dreamwaq 实时曲线版：
一个窗口内 3 个等尺寸子图实时滚动 —— ①左前腿(FL)关节角度 ②FL关节角速度 ③基座欧拉角。
坐标轴无标题、横坐标隐藏、纵轴与图例中文；欧拉角半圈 [-π, π]，贴边不跳，站起缓慢回 0。
Ctrl+C 退出时立刻保存 3 张 PNG 并强制退出（不等 viewer / 弹窗回滚）。

PD 状态控制（电机 q/dq/kp/kd 位控）：
  键盘 p=策略+PD / o=纯PD默认站姿 / i=纯PD锁定当前关节角
  手柄 X=策略 / A=默认站姿 / B=锁定当前角
"""
from sim2sim_config_loader import load_cfg
Go2_AMP_DreamWaQ_Cfg_Yu = load_cfg(
    "legged_gym/envs/Go2_AMP_DreamWaQ/Go2_AMP_DreamWaQ_Config.py",
    "Go2_AMP_DreamWaQ_Cfg_Yu",
)
import os
import signal
import numpy as np
import mujoco, mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import time
from pynput import keyboard
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
plt.rcParams['font.size'] = 16            # 全局中文字体加大
plt.rcParams['axes.labelsize'] = 18       # 纵轴中文标签
plt.rcParams['xtick.labelsize'] = 15      # 坐标轴刻度
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 14

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


class EulerHalfCircle:
    """半圈显示 ∈ [-π, π]：正常时最短角差跟随；越界则贴边；
    贴边后若 raw 已回到半圈内，则向 raw 缓慢靠拢（站起来会回到 0），不跳边。"""

    def __init__(self, max_step=0.15):
        self.cont = None
        self.max_step = float(max_step)

    def __call__(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        if self.cont is None:
            self.cont = np.clip(raw.copy(), -np.pi, np.pi)
            return self.cont.copy()

        out = np.empty(3, dtype=np.float64)
        for i in range(3):
            delta = np.arctan2(np.sin(raw[i] - self.cont[i]), np.cos(raw[i] - self.cont[i]))
            cand = self.cont[i] + delta
            if abs(cand) <= np.pi:
                out[i] = cand
                continue
            # 越界：先贴在 ±π
            out[i] = np.pi if cand > 0.0 else -np.pi
            # raw 已回到半圈内 → 向 raw 靠拢（站立恢复）；跨支点时先往 0 走
            if abs(raw[i]) < 0.9 * np.pi:
                if abs(raw[i] - out[i]) <= np.pi:
                    err = raw[i] - out[i]
                else:
                    err = -out[i]  # 先回到 0
                out[i] = np.clip(out[i] + np.clip(err, -self.max_step, self.max_step), -np.pi, np.pi)
        self.cont = out.copy()
        return out


class ThreePanelPlotter:
    """1×3 等尺寸子图实时滚动；第 1 子图仅实测关节角。"""

    SAVE_FILES = ['fl_joint_angles.png', 'fl_joint_velocities.png', 'base_euler_angles.png']
    YLABELS = ['角度（弧度）', '角速度（弧度/秒）', '角度（弧度）']
    LEGENDS = [['髋关节', '大腿关节', '小腿关节'],
               ['髋关节', '大腿关节', '小腿关节'],
               ['横滚角', '俯仰角', '偏航角']]
    YLABEL_FS = 18
    TICK_FS = 15
    LEGEND_FS = 14

    def __init__(self, dt=0.005, window_s=5.0, refresh_every=10, save_dir='.'):
        self.refresh_every = refresh_every
        self.window_s = float(window_s)
        # dt 必须是 update() 的调用间隔（物理步），不能用控制步
        self.max_points = max(2, int(round(self.window_s / dt)))
        self.save_dir = save_dir
        self._count = 0
        self.t_buf = deque(maxlen=self.max_points)
        self.bufs = [deque(maxlen=self.max_points) for _ in range(3)]

        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(11, 8))
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
            ax.set_ylabel(ylab, fontsize=self.YLABEL_FS)
            ax.tick_params(axis='both', labelsize=self.TICK_FS)
            ax.grid(True, alpha=0.35)
            ax.legend(loc='upper right', fontsize=self.LEGEND_FS)
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    def update(self, t, fl_angles, fl_vels, euler):
        self.t_buf.append(float(t))
        self.bufs[0].append(np.asarray(fl_angles, dtype=np.float64).copy())
        self.bufs[1].append(np.asarray(fl_vels, dtype=np.float64).copy())
        self.bufs[2].append(np.asarray(euler, dtype=np.float64).copy())
        self._count += 1
        if self._count % self.refresh_every != 0:
            return
        ts = np.asarray(self.t_buf)
        t_end = float(ts[-1])
        # 横轴固定 5 s：一开始就是 [0, 5]，之后滚动
        if t_end < self.window_s:
            x0, x1 = 0.0, self.window_s
        else:
            x0, x1 = t_end - self.window_s, t_end
        for i in range(3):
            data = np.asarray(self.bufs[i])
            for line, col in zip(self.lines[i], data.T):
                line.set_data(ts, col)
            self.axes[i].set_xlim(x0, x1)
            self.axes[i].relim()
            self.axes[i].autoscale_view(scalex=False, scaley=True)
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
            t_end = float(ts[-1])
            ax.set_xlim(t_end - self.window_s, t_end)
            ax.set_ylabel(ylab, fontsize=self.YLABEL_FS)
            ax.tick_params(axis='both', labelsize=self.TICK_FS)
            ax.grid(True, alpha=0.35)
            ax.legend(loc='upper right', fontsize=self.LEGEND_FS)
            fig.tight_layout()
            out = os.path.join(self.save_dir, fname)
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f'[保存] {out}')


class PdModeController:
    """PD 状态控制模式切换。
    - policy: 策略目标角 + PD
    - pd_default: 纯 PD 锁默认站姿
    - pd_hold: 纯 PD 锁进入模式时的关节角
    """

    MODE_POLICY = 'policy'
    MODE_PD_DEFAULT = 'pd_default'
    MODE_PD_HOLD = 'pd_hold'
    MODE_NAMES = {
        MODE_POLICY: '策略+PD',
        MODE_PD_DEFAULT: '纯PD·默认站姿',
        MODE_PD_HOLD: '纯PD·锁定当前角',
    }
    KEY_TO_MODE = {
        'p': MODE_POLICY, 'P': MODE_POLICY,
        'o': MODE_PD_DEFAULT, 'O': MODE_PD_DEFAULT,
        'i': MODE_PD_HOLD, 'I': MODE_PD_HOLD,
    }
    # Xbox: A=0, B=1, X=2
    BTN_TO_MODE = {2: MODE_POLICY, 0: MODE_PD_DEFAULT, 1: MODE_PD_HOLD}

    def __init__(self, num_actions, initial=MODE_POLICY, joystick=None):
        self.mode = initial
        self.joystick = joystick
        self.hold_target_q = np.zeros(num_actions, dtype=np.double)
        self._prev_btn = {b: False for b in self.BTN_TO_MODE}
        self._pending = None
        self._listener = keyboard.Listener(on_press=self._on_key)
        self._listener.daemon = True
        self._listener.start()
        print('[PD] 模式: p=策略+PD / o=纯PD默认站姿 / i=纯PD锁定当前角（手柄 X/A/B）')
        print(f'[PD] 当前: {self.MODE_NAMES[self.mode]}')

    def _set_mode(self, mode, q=None, default_dof_pos=None):
        if mode == self.MODE_PD_HOLD and q is not None and default_dof_pos is not None:
            self.hold_target_q = (q - default_dof_pos).astype(np.double)
        if mode != self.mode:
            self.mode = mode
            print(f'\n[PD] 模式 -> {self.MODE_NAMES[self.mode]}')

    def _on_key(self, key):
        try:
            ch = key.char
        except AttributeError:
            return
        if ch in self.KEY_TO_MODE:
            self._pending = self.KEY_TO_MODE[ch]

    def update(self, q, default_dof_pos):
        if self.joystick is not None:
            try:
                nbtn = self.joystick.get_numbuttons()
                for btn, mode in self.BTN_TO_MODE.items():
                    pressed = nbtn > btn and bool(self.joystick.get_button(btn))
                    if pressed and not self._prev_btn[btn]:
                        self._pending = mode
                    self._prev_btn[btn] = pressed
            except Exception:
                pass
        if self._pending is not None:
            self._set_mode(self._pending, q=q, default_dof_pos=default_dof_pos)
            self._pending = None

    def resolve_target_q(self, policy_target_q):
        if self.mode == self.MODE_POLICY:
            return policy_target_q
        if self.mode == self.MODE_PD_HOLD:
            return self.hold_target_q
        return np.zeros_like(policy_target_q)


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
    foot_forces = 0.0
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name is not None and 'foot' in body_name:
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces = data.cfrc_ext[i][2].copy().astype(np.double)
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)


def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    """电机 PD 状态控制：τ = kp(q* + q0 − q) + kd(dq* − dq)。"""
    return (target_q + cfg.robot_config.default_dof_pos - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg):
    vel_arrow = VelocityArrowViewer(max_cmd=(x_vel_max, y_vel_max, yaw_vel_max),
                                    gears=[0.5, 1.0, 1.5, 2.0])
    pd_mode = PdModeController(
        num_actions=cfg.env.num_actions,
        joystick=vel_arrow.joystick if vel_arrow.use_joystick else None,
    )
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos
    mujoco.mj_step(model, data)

    plotter = ThreePanelPlotter(dt=cfg.sim_config.dt, window_s=5.0)
    euler_half = EulerHalfCircle()
    saved = {'done': False}

    def _save_now_and_exit(signum=None, frame=None):
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
            pd_mode.update(q, cfg.robot_config.default_dof_pos)

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

            # PD 状态控制：按模式选目标角 → τ = kp/kd
            policy_target_q = target_q if sim_step >= 100 else np.zeros_like(target_q)
            cmd_target_q = pd_mode.resolve_target_q(policy_target_q)
            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            tau = pd_control(cmd_target_q, q, cfg.robot_config.kps,
                             target_dq, dq, cfg.robot_config.kds, cfg)
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl = tau

            mujoco.mj_step(model, data)

            vel_arrow.draw_arrows(viewer, data)
            viewer.sync()

            # 半圈显示：曲线 ∈ [-π, π]，过界贴边不跳
            euler = euler_half(quaternion_to_euler_array(quat))
            plotter.update(data.time, q[0:3], dq[0:3], euler)

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
