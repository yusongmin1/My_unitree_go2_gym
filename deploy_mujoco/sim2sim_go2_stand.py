from sim2sim_config_loader import load_cfg
Go2_Stand_Cfg_Yu = load_cfg(
    "legged_gym/envs/Go2_Stand/Go2_Stand_Config.py",
    "Go2_Stand_Cfg_Yu",
)
import math
import numpy as np
import mujoco, mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import time
from viewer_utils import VelocityArrowViewer
from pynput import keyboard

MODE_NAMES = {-1: "handstand", 0: "quad", 1: "legstand"}


def _range_max(lo_hi):
    return float(max(abs(lo_hi[0]), abs(lo_hi[1])))


def cmd_limits_for_mode(cfg, stand_command):
    """按 stand_command 返回 (vx_max, vy_max, yaw_max)。"""
    ranges = cfg.commands.ranges_quad if stand_command == 0 else cfg.commands.ranges_hand
    return (
        _range_max(ranges.lin_vel_x),
        _range_max(ranges.lin_vel_y),
        _range_max(ranges.ang_vel_yaw),
    )


class StandModeController:
    """A/B/Y（手柄或键盘）切换三种步态。
    A → quad(0), B → handstand(-1), Y → legstand(+1)。
    键盘无手柄时也可用 2/3/4。
    """

    # Xbox / 常见手柄：A=0, B=1, X=2, Y=3
    BTN_A, BTN_B, BTN_Y = 0, 1, 3
    BTN_TO_MODE = {BTN_A: 0.0, BTN_B: -1.0, BTN_Y: 1.0}
    KEY_TO_MODE = {
        "a": 0.0, "A": 0.0,
        "b": -1.0, "B": -1.0,
        "y": 1.0, "Y": 1.0,
        "2": -1.0, "3": 0.0, "4": 1.0,
    }

    def __init__(self, initial=0.0, joystick=None):
        self.stand_command = float(initial)
        self.changed = True
        self.joystick = joystick
        self._prev_btn = {self.BTN_A: False, self.BTN_B: False, self.BTN_Y: False}
        self._listener = keyboard.Listener(on_press=self._on_key)
        self._listener.daemon = True
        self._listener.start()
        print("[stand] 模式: A=四足(0), B=handstand(-1), Y=legstand(+1)（键盘 a/b/y 或 2/3/4）")
        print(f"[stand] 当前模式: {MODE_NAMES[int(self.stand_command)]} ({int(self.stand_command)})")

    def set_joystick(self, joystick):
        self.joystick = joystick

    def _set_mode(self, mode):
        mode = float(mode)
        if mode != self.stand_command:
            self.stand_command = mode
            self.changed = True
            print(
                f"\n[stand] 模式 -> {MODE_NAMES[int(self.stand_command)]} "
                f"({int(self.stand_command)})"
            )

    def _on_key(self, key):
        try:
            ch = key.char
        except AttributeError:
            return
        if ch in self.KEY_TO_MODE:
            self._set_mode(self.KEY_TO_MODE[ch])

    def update(self):
        """每帧轮询手柄 A/B/Y（边沿触发）。需在 pygame.event.pump 之后调用。"""
        if self.joystick is None:
            return
        try:
            nbtn = self.joystick.get_numbuttons()
        except Exception:
            return
        for btn, mode in self.BTN_TO_MODE.items():
            pressed = nbtn > btn and bool(self.joystick.get_button(btn))
            if pressed and not self._prev_btn[btn]:
                self._set_mode(mode)
            self._prev_btn[btn] = pressed

    def consume_changed(self):
        flag = self.changed
        self.changed = False
        return flag


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
        if body_name is not None and "foot" in body_name:
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces = data.cfrc_ext[i][2].copy().astype(np.double)
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)


def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    return (target_q + cfg.robot_config.default_dof_pos - q) * kp + (target_dq - dq) * kd


def apply_mode_cmd_limits(vel_arrow, cfg, stand_command):
    """按模式更新速度上限，并裁剪当前命令。"""
    limits = cmd_limits_for_mode(cfg, stand_command)
    vel_arrow.max_cmd = limits
    if stand_command == 0:
        vel_arrow.gears = [0.5, 1.0]
    else:
        # 双足最大 0.4，档位覆盖 ranges_hand
        vel_arrow.gears = [0.2, 0.4]
        vel_arrow.gear_idx = min(vel_arrow.gear_idx, len(vel_arrow.gears) - 1)
    vel_arrow._clip_cmd()
    if stand_command != 0:
        vel_arrow.cmd[1] = 0.0  # 双足不采侧向速度
    print(f"[stand] 速度上限 vx/vy/yaw = {limits}")


def run_mujoco(policy, cfg):
    mode_ctrl = StandModeController(initial=0.0)
    vx0, vy0, yaw0 = cmd_limits_for_mode(cfg, mode_ctrl.stand_command)
    vel_arrow = VelocityArrowViewer(max_cmd=(vx0, vy0, yaw0), gears=[0.5, 1.0])
    if vel_arrow.use_joystick:
        mode_ctrl.set_joystick(vel_arrow.joystick)
    apply_mode_cmd_limits(vel_arrow, cfg, mode_ctrl.stand_command)

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos
    mujoco.mj_step(model, data)

    n_obs = cfg.env.num_single_obs  # 48
    n_hist = cfg.env.num_obs_hist  # 5 历史 + 当前 → 共 6 帧（DreamWaQ CENet）
    hist_len = n_hist + 1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] = np.array([0.0, -0.25, 0.824])

        target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
        action = np.zeros((cfg.env.num_actions), dtype=np.double)

        hist_obs = deque(maxlen=hist_len)
        for _ in range(hist_len):
            hist_obs.append(np.zeros([1, n_obs], dtype=np.float32))

        policy_input = np.zeros([1, n_obs * hist_len], dtype=np.float32)
        count_lowlevel = 1
        sim_step = 0
        np.set_printoptions(formatter={"float": "{:0.4f}".format})

        while True:
            step_start = time.time()
            cmd = vel_arrow.update_cmd()  # 内部 pygame.event.pump
            mode_ctrl.update()  # 手柄 A/B/Y
            if mode_ctrl.consume_changed():
                apply_mode_cmd_limits(vel_arrow, cfg, mode_ctrl.stand_command)

            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = float(cmd[0]), float(cmd[1]), float(cmd[2])
            stand_command = mode_ctrl.stand_command
            if stand_command != 0:
                y_vel_cmd = 0.0

            q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data, model)

            if count_lowlevel % cfg.sim_config.decimation == 0:
                obs = np.zeros([1, n_obs], dtype=np.float32)
                # 与训练一致：[0:3]ang [3:6]grav [6:9]cmd [9:21]dof_pos
                # [21:33]dof_vel [33:45]action [45:47]phase [47]stand_command
                obs[0, 0:3] = omega * cfg.normalization.obs_scales.ang_vel
                obs[0, 3:6] = gvec
                obs[0, 6] = x_vel_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 7] = y_vel_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 8] = yaw_vel_cmd * cfg.normalization.obs_scales.ang_vel
                obs[0, 9:21] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = action

                phase = (count_lowlevel * cfg.sim_config.dt) % cfg.rewards.cycle_time / cfg.rewards.cycle_time
                obs[0, 45] = math.sin(2 * math.pi * phase)
                obs[0, 46] = math.cos(2 * math.pi * phase)
                obs[0, 47] = stand_command

                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                hist_obs.append(obs)

                for i, frame in enumerate(hist_obs):
                    policy_input[0, i * n_obs : (i + 1) * n_obs] = frame[0, :]

                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                target_q = action * cfg.control.action_scale

            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            if sim_step < 300:
                tau = pd_control(
                    np.zeros((cfg.env.num_actions)), q, cfg.robot_config.kps,
                    target_dq, dq, cfg.robot_config.kds, cfg,
                )
            else:
                tau = pd_control(
                    target_q, q, cfg.robot_config.kps,
                    target_dq, dq, cfg.robot_config.kds, cfg,
                )
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl = tau

            mujoco.mj_step(model, data)
            vel_arrow.draw_arrows(viewer, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            count_lowlevel += 1
            sim_step += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deployment script.")
    parser.add_argument(
        "--load_model",
        type=str,
        default=f"{LEGGED_GYM_ROOT_DIR}/logs/go2_stand/exported/policies/policy_dwaq.pt",
        help="Run to load from.",
    )
    parser.add_argument("--terrain", action="store_true", help="terrain or plane")
    args = parser.parse_args()

    class Sim2simCfg(Go2_Stand_Cfg_Yu):
        class sim_config:
            mujoco_model_path = f"{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/go2/scene.xml"
            sim_duration = 120.0
            dt = 0.005
            decimation = 4

        class robot_config:
            kps = np.array([28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28], dtype=np.double)
            kds = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], dtype=np.double)
            tau_limit = 33.5 * np.ones(12, dtype=np.double)
            # MuJoCo 顺序 FL, FR, RL, RR；与训练 default_joint_angles 对齐
            default_dof_pos = np.array(
                [
                    0.0, 0.8, -1.5,
                    0.0, 0.8, -1.5,
                    0.0, 0.8, -1.5,
                    0.0, 0.8, -1.5,
                ],
                dtype=np.double,
            )

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
