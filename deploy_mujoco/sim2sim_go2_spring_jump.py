from sim2sim_config_loader import load_cfg
Go2_Spring_Jump_Cfg_Yu = load_cfg(
    "legged_gym/envs/Go2_Flip/Go2_Spring_Jump/Go2_Spring_Jump_Config.py",
    "Go2_Spring_Jump_Cfg_Yu",
)
Go2_Spring_Jump_PPO_Yu = load_cfg(
    "legged_gym/envs/Go2_Flip/Go2_Spring_Jump/Go2_Spring_Jump_Config.py",
    "Go2_Spring_Jump_PPO_Yu",
)
import math
import numpy as np
import mujoco, mujoco.viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
# from legged_gym.envs import *
import torch
import time
from viewer_utils import VelocityArrowViewer
import pygame
from threading import Thread
import matplotlib
matplotlib.use('TkAgg')  # 本机 Qt 的 xcb 插件不可用，强制使用 Tk 后端
import matplotlib.pyplot as plt


class ActionMonitor:
    """实时滚动绘制策略输出监控图，一个窗口内竖排三个子图：
      1) 12 维 action 曲线
      2) 逐时刻 action_rate = sum((action - last_action)^2)（与奖励定义一致）
      3) 最近 250 个策略步（=5s，一个 episode 长度）内 action_rate 的滚动总和
    MuJoCo 关节顺序为 FL, FR, RL, RR，每条腿 [hip, thigh, calf]。
    """
    ACTION_LABELS = ['FL hip', 'FL thigh', 'FL calf',
                     'FR hip', 'FR thigh', 'FR calf',
                     'RL hip', 'RL thigh', 'RL calf',
                     'RR hip', 'RR thigh', 'RR calf']

    def __init__(self, policy_dt, window_s=10.0, refresh_every=10,
                 rate_sum_window=250):
        """
        Args:
            policy_dt: 策略推理周期 [s]（= dt * decimation）
            window_s: 曲线滚动窗口长度 [s]
            refresh_every: 每多少次策略推理刷新一次绘图
            rate_sum_window: action_rate 滚动求和的步数（250 步 = 5s = 一个 episode）
        """
        self.refresh_every = refresh_every
        self.max_points = max(2, int(window_s / (policy_dt * refresh_every)))
        self.t_buf = deque(maxlen=self.max_points)
        self.a_buf = deque(maxlen=self.max_points)
        self.rate_buf = deque(maxlen=self.max_points)
        # 滚动求和窗口（策略步数），与 episode 长度对齐
        self.rate_hist = deque(maxlen=rate_sum_window)
        self.sum_buf = deque(maxlen=self.max_points)
        self._count = 0
        self.last_action = None

        plt.ion()
        self.fig, (self.ax_action, self.ax_rate, self.ax_sum) = \
            plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        try:
            self.fig.canvas.manager.set_window_title('Action Monitor (real-time)')
        except Exception:
            pass

        self.action_lines = []
        for label in self.ACTION_LABELS:
            line, = self.ax_action.plot([], [], label=label, linewidth=1.0)
            self.action_lines.append(line)
        self.ax_action.set_ylabel('action')
        self.ax_action.set_title('Policy action')
        self.ax_action.grid(True, alpha=0.4)
        self.ax_action.legend(loc='upper right', ncol=4, fontsize=6)

        self.rate_line, = self.ax_rate.plot([], [], color='tab:red', linewidth=1.2)
        self.ax_rate.set_ylabel('action rate')
        self.ax_rate.set_title('sum((action - last_action)^2)')
        self.ax_rate.grid(True, alpha=0.4)

        self.sum_line, = self.ax_sum.plot([], [], color='tab:purple', linewidth=1.2)
        self.ax_sum.set_ylabel('rate sum')
        self.ax_sum.set_xlabel('time [s]')
        self.ax_sum.set_title('Rolling sum of action_rate over last 250 policy steps (5s, one episode)')
        self.ax_sum.grid(True, alpha=0.4)

        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    def update(self, t, action):
        """每次策略推理后调用一次：缓存数据，按 refresh_every 频率重绘。"""
        action = np.asarray(action)
        if self.last_action is not None:
            rate = float(np.sum(np.square(action - self.last_action)))
            self.rate_hist.append(rate)                     # 定长 250 步队列
            rate_sum = float(np.sum(self.rate_hist))        # 最近 250 步总和
            self.t_buf.append(float(t))
            self.a_buf.append(action.copy())
            self.rate_buf.append(rate)
            self.sum_buf.append(rate_sum)
        self.last_action = action.copy()
        self._count += 1
        if self._count % self.refresh_every != 0:
            return
        ts = np.asarray(self.t_buf)
        for line, col in zip(self.action_lines, np.asarray(self.a_buf).T):
            line.set_data(ts, col)
        self.rate_line.set_data(ts, np.asarray(self.rate_buf))
        self.sum_line.set_data(ts, np.asarray(self.sum_buf))
        for ax in (self.ax_action, self.ax_rate, self.ax_sum):
            ax.relim()
            ax.autoscale_view()
        plt.pause(0.001)

    def close(self):
        plt.close(self.fig)


x_vel_max, y_vel_max, yaw_vel_max = 1.5, 1.0, 1.5
    
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data,model):
    '''Extracts an observation from the mujoco data structure
    '''

    # print(data.qpos.astype(np.double).shape,data.qvel.astype(np.double).shape)
    q = data.qpos[7:19].astype(np.double)
    dq = data.qvel[6:].astype(np.double)
    quat = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]]
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.qvel[3:6].astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    base_pos = data.qpos[0:3].astype(np.double)
    foot_positions = []
    # foot_forces = data.cfrc_ext[0][2].copy().astype(np.double)
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        # print(body_name)
        if 'foot' in body_name: 
            # print(body_name)
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces = data.cfrc_ext[i][2].copy().astype(np.double)
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)

def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    '''Calculates torques from position commands
    '''
    torque_out = (target_q + cfg.robot_config.default_dof_pos - q ) * kp + (target_dq - dq)* kd
    return torque_out


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
    # 初始化手柄/键盘输入 + 箭头绘制工具（手柄优先，无手柄回退键盘）
    vel_arrow = VelocityArrowViewer(max_cmd=(x_vel_max, y_vel_max, yaw_vel_max))
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    
    model.opt.timestep = cfg.sim_config.dt
    
    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions  # This sx_vel_cmdhould match the number of actuated joints in your model
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos

    mujoco.mj_step(model, data)
    
    # 使用官方 mujoco.viewer（支持 user_scn 绘制箭头）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] =np.array([0.0,-0.25,0.824])

        target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
   
        action = np.zeros((cfg.env.num_actions), dtype=np.double)

        hist_obs = deque()
        for _ in range(cfg.env.frame_stack):
            hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))

        count_lowlevel = 1
    

        np.set_printoptions(formatter={'float': '{:0.4f}'.format})

        # 实时 action 监控窗口：action / action_rate / 250 步 rate 总和（竖排子图）
        action_monitor = ActionMonitor(
            policy_dt=cfg.sim_config.dt * cfg.sim_config.decimation,
            window_s=10.0, refresh_every=10, rate_sum_window=250)

        sim_step = 0
        while True:  # 无限循环，直到手动关闭程序
            # 每帧更新命令（手柄优先，无手柄则用键盘累积值）
            step_start = time.time()  # 实时同步：记录本步开始时间
            cmd = vel_arrow.update_cmd()
            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = cmd[0], cmd[1], cmd[2]

            # Obtain an observation
            q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data,model)
            # q = q[-cfg.env.num_actions:]
            # dq = dq[-cfg.env.num_actions:]
        
            # 1000hz -> 100hz
            # if count_lowlevel>300:
            #     x_vel_cmd=1.0
            if count_lowlevel % cfg.sim_config.decimation == 0:
                print(data.qpos[:3])
                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi

                # 45 维布局（与训练一致）：[0:3]命令 [3:6]角速度 [6:9]重力投影 [9:21]关节角 [21:33]关节速度 [33:45]动作
                obs[0, 0] = 0.7
                obs[0, 1] = 0
                obs[0, 2] = x_vel_cmd
                obs[0, 3:6] = omega*cfg.normalization.obs_scales.ang_vel
                obs[0, 6:9] = gvec

                obs[0, 9:21] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = action


                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

                target_q = action * cfg.control.action_scale
                # 实时更新 action 监控图（三个竖排子图）
                action_monitor.update(data.time, action)

            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

            # Generate PD control
            if sim_step <100:
                tau = pd_control(np.zeros((cfg.env.num_actions)), q, cfg.robot_config.kps,
                                target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques
                tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques

            else:
                # if sim_step <150:
                #     x_vel_cmd=1.0
                tau = pd_control(target_q, q, cfg.robot_config.kps,
                                target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques
                tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        
            data.ctrl = tau
            applied_tau = data.actuator_force

            mujoco.mj_step(model, data)

            vel_arrow.draw_arrows(viewer, data)
            viewer.sync()
            # 实时同步：若本步耗时小于仿真步长则等待补足，防止加速播放
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            count_lowlevel += 1
            sim_step += 1
        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default="logs/go2_spring_jump/exported/policies/policy_1.pt",help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(Go2_Spring_Jump_Cfg_Yu):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/go2/scene.xml'
            sim_duration = 120.0
            dt = 0.005
            decimation = 4

        class robot_config:
            kps = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], dtype=np.double)
            kds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.double)
            tau_limit = 25 * np.ones(12, dtype=np.double)
            default_dof_pos = np.array( [0.,0.8,-1.5,
                -0.,0.8,-1.5,
                 0.,1.0,-1.5,
                -0.,1. ,-1.5], dtype=np.double)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())