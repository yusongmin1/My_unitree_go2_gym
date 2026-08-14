"""
Sim2Sim 可视化工具：手柄输入 + 速度箭头绘制。
- 手柄优先（pygame），无手柄时自动回退到键盘（pynput）。
- 在 MuJoCo viewer 中绘制两个箭头：绿色=目标速度命令，蓝色=实际速度。
参考自 deploy/deploy_mujoco/utils.py。
"""
import numpy as np
import mujoco
import pygame
from pynput import keyboard


class VelocityArrowViewer:
    """封装手柄/键盘输入 + MuJoCo user_scn 箭头绘制。"""

    def __init__(self, max_cmd=(1.5, 1.0, 1.5), dead_zone=0.1):
        """
        Args:
            max_cmd: (vx_max, vy_max, yaw_max) 命令最大值，默认前进 1.5。
            dead_zone: 手柄摇杆死区。
        """
        self.max_cmd = max_cmd
        self.dead_zone = dead_zone

        # 当前命令（vx, vy, yaw）
        self.cmd = np.zeros(3, dtype=np.float32)
        # 键盘增量步长
        self.key_step = np.array([0.3, 0.3, 0.5], dtype=np.float32)

        # ----- 手柄初始化（pygame）-----
        pygame.init()
        self.use_joystick = False
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.use_joystick = True
            print(f"[viewer_utils] 检测到手柄: {self.joystick.get_name()}")
        else:
            print("[viewer_utils] 未检测到手柄，使用键盘控制。")
            print("  键盘: 6/7=前进/后退, 8/9=左/右平移, -/=左转/右转, 1=归零")

        # ----- 键盘监听（手柄不可用时启用，或作为后备）-----
        self._kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self._kb_listener.daemon = True
        if not self.use_joystick:
            self._kb_listener.start()

    # ---------- 输入处理 ----------
    def _on_key_press(self, key):
        """键盘控制（无手柄时使用）。"""
        try:
            if key.char == '6':
                self.cmd[0] += self.key_step[0]
            elif key.char == '7':
                self.cmd[0] -= self.key_step[0]
            elif key.char == '8':
                self.cmd[1] += self.key_step[1]
            elif key.char == '9':
                self.cmd[1] -= self.key_step[1]
            elif key.char == '-':
                self.cmd[2] += self.key_step[2]
            elif key.char == '=':
                self.cmd[2] -= self.key_step[2]
            elif key.char == '1':
                self.cmd[:] = 0
            # 限幅
            self.cmd[0] = np.clip(self.cmd[0], -self.max_cmd[0], self.max_cmd[0])
            self.cmd[1] = np.clip(self.cmd[1], -self.max_cmd[1], self.max_cmd[1])
            self.cmd[2] = np.clip(self.cmd[2], -self.max_cmd[2], self.max_cmd[2])
            print(f"\r[键盘] cmd: vx={self.cmd[0]:.2f}, vy={self.cmd[1]:.2f}, yaw={self.cmd[2]:.2f}", end='')
        except AttributeError:
            pass

    def update_cmd(self):
        """每帧调用：从手柄读取命令（若使用手柄）。返回当前 cmd。"""
        if self.use_joystick:
            pygame.event.pump()
            lx = self.joystick.get_axis(0)   # 左摇杆左右
            ly = self.joystick.get_axis(1)   # 左摇杆上下
            rx = self.joystick.get_axis(3)   # 右摇杆左右
            if abs(lx) < self.dead_zone: lx = 0.0
            if abs(ly) < self.dead_zone: ly = 0.0
            if abs(rx) < self.dead_zone: rx = 0.0
            self.cmd[0] = -ly * self.max_cmd[0]   # 前进
            self.cmd[1] = -lx * self.max_cmd[1]   # 平移
            self.cmd[2] = -rx * self.max_cmd[2]   # 转向
        return self.cmd

    # ---------- 箭头绘制 ----------
    @staticmethod
    def _add_arrow(geom_elem, pos, vec, rgba, scale=0.7):
        """在 user_scn 中添加一个箭头几何体。"""
        vel_norm = np.linalg.norm(vec)
        display_norm = min(vel_norm * scale, 1.0)

        if display_norm < 0.10:
            mujoco.mjv_initGeom(
                geom_elem,
                type=mujoco.mjtGeom.mjGEOM_NONE,
                size=[0, 0, 0], pos=pos, mat=np.eye(3).flatten(), rgba=[0, 0, 0, 0]
            )
            return

        mat = np.zeros(9)
        target_quat = np.zeros(4)
        vec_normalized = vec / vel_norm
        mujoco.mju_quatZ2Vec(target_quat, vec_normalized)
        mujoco.mju_quat2Mat(mat, target_quat)

        mat = mat.reshape(3, 3)
        mat[:, 2] *= display_norm

        mujoco.mjv_initGeom(
            geom_elem,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[0.02, 0.02, display_norm],  # [height, width, length]
            pos=pos,
            mat=mat.flatten(),
            rgba=rgba
        )

    def draw_arrows(self, viewer, mj_data):
        """在 viewer 的 user_scn 中绘制目标速度（绿）和实际速度（蓝）两个箭头。

        Args:
            viewer: mujoco.viewer.Handle（launch_passive 返回的上下文对象）。
            mj_data: mujoco.MjData，用于读取机器人位姿和速度。
        """
        # 重置 user scene 几何体
        viewer.user_scn.ngeom = 0

        base_pos_world = mj_data.qpos[:3]
        base_quat = mj_data.qpos[3:7]

        # 箭头起点：机器人基座上方 0.2m
        offset_body = np.array([0.0, 0.0, 0.2])
        offset_world = np.zeros(3)
        mujoco.mju_rotVecQuat(offset_world, offset_body, base_quat)
        start_pos = base_pos_world + offset_world

        # 目标速度（机器人坐标系 → 世界坐标系）
        tgt_vel_body = np.array([self.cmd[0], self.cmd[1], 0.0])
        tgt_vel_world = np.zeros(3)
        mujoco.mju_rotVecQuat(tgt_vel_world, tgt_vel_body, base_quat)

        # 实际速度（世界坐标系 → 机器人坐标系，取 xy → 世界坐标系）
        raw_cur_vel_world = mj_data.qvel[:3]
        raw_cur_vel = np.zeros(3)
        neg_quat = np.zeros(4)
        mujoco.mju_negQuat(neg_quat, base_quat)
        mujoco.mju_rotVecQuat(raw_cur_vel, raw_cur_vel_world, neg_quat)
        cur_vel_body = np.array([raw_cur_vel[0], raw_cur_vel[1], 0.0])
        cur_vel_world = np.zeros(3)
        mujoco.mju_rotVecQuat(cur_vel_world, cur_vel_body, base_quat)

        COLOR_CMD = [0, 1, 0, 1]    # 绿色：目标速度命令
        COLOR_REAL = [0, 0, 1, 1]   # 蓝色：实际速度

        # 目标速度箭头（绿）
        self._add_arrow(viewer.user_scn.geoms[0], start_pos, tgt_vel_world, COLOR_CMD)
        # 实际速度箭头（蓝）
        self._add_arrow(viewer.user_scn.geoms[1], start_pos, cur_vel_world, COLOR_REAL)
        viewer.user_scn.ngeom = 2
