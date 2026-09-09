"""将 MimicKit .pkl 动作文件转换为 .npz 格式并上传到 wandb。

支持 Go2（四足）和 G1（人形）机器人。机器人类型通过 --robot 参数指定，
默认根据文件名前缀自动检测（go2_*.pkl → go2，g1_*.pkl → g1）。
"""

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_from_angle_axis,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

RobotType = Literal["go2", "g1"]

# MimicKit joint 顺序（深度优先，来自各自的 .xml 资产文件）。
GO2_JOINT_NAMES = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)

G1_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

ROBOT_JOINT_NAMES: dict[RobotType, tuple[str, ...]] = {
  "go2": GO2_JOINT_NAMES,
  "g1": G1_JOINT_NAMES,
}

# 每种机器人 frame 向量中 DOF 数量（= 总列数 - 6）。
ROBOT_DOF_COUNT: dict[RobotType, int] = {
  "go2": 12,
  "g1": 29,
}


def _detect_robot(input_file: str) -> RobotType:
  """根据文件名前缀自动检测机器人类型。"""
  stem = Path(input_file).stem.lower()
  if stem.startswith("g1"):
    return "g1"
  if stem.startswith("go2"):
    return "go2"
  raise ValueError(
    f"无法从文件名 '{input_file}' 自动检测机器人类型，"
    "请通过 --robot 显式指定（go2 或 g1）。"
  )


def _make_scene(
  robot: RobotType, output_fps: int, device: str
) -> tuple[Scene, Simulation]:
  """根据机器人类型创建对应的场景和仿真实例。"""
  if robot == "go2":
    from mjlab.tasks.tracking.config.go2.env_cfgs import (
      unitree_go2_flat_tracking_env_cfg,
    )

    env_cfg = unitree_go2_flat_tracking_env_cfg()
  else:
    from mjlab.tasks.tracking.config.g1.env_cfgs import (
      unitree_g1_flat_tracking_env_cfg,
    )

    env_cfg = unitree_g1_flat_tracking_env_cfg()

  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return scene, sim


def _exp_map_to_quat(exp_map: torch.Tensor) -> torch.Tensor:
  """将 3D 指数映射转换为四元数 (wxyz)。

  Args:
    exp_map: 形状 (N, 3)，方向为轴，模长为角度。

  Returns:
    四元数 (w, x, y, z)，形状 (N, 4)。
  """
  angle = torch.norm(exp_map, dim=-1)  # (N,)
  eps = 1e-8
  axis = exp_map / (angle.unsqueeze(-1) + eps)  # (N, 3)
  return quat_from_angle_axis(angle, axis)


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    robot: RobotType,
    output_fps: int,
    device: torch.device | str,
  ):
    self.motion_file = motion_file
    self.robot = robot
    self.output_fps = output_fps
    self.output_dt = 1.0 / output_fps
    self.current_idx = 0
    self.device = device
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    with open(self.motion_file, "rb") as f:
      data = pickle.load(f)

    self.input_fps: int = data["fps"]
    self.input_dt = 1.0 / self.input_fps
    frames: np.ndarray = np.array(data["frames"], dtype=np.float32)
    self.input_frames = frames.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

    n_dof = ROBOT_DOF_COUNT[self.robot]
    expected_cols = 6 + n_dof
    if frames.shape[1] != expected_cols:
      raise ValueError(
        f"机器人 '{self.robot}' 期望每帧 {expected_cols} 列"
        f"（3 位置 + 3 旋转 + {n_dof} DOF），"
        f"但得到 {frames.shape[1]} 列。"
      )

    frames_t = torch.from_numpy(frames).to(torch.float32).to(self.device)
    self.motion_base_poss_input = frames_t[:, :3]  # (T, 3)
    exp_map = frames_t[:, 3:6]  # (T, 3)
    self.motion_base_rots_input = _exp_map_to_quat(exp_map)  # (T, 4) wxyz
    self.motion_dof_poss_input = frames_t[:, 6:]  # (T, n_dof)

  def _interpolate_motion(self):
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_times = times
    self.output_frames = times.shape[0]
    idx0, idx1, blend = self._compute_frame_blend(times)

    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[idx0],
      self.motion_base_poss_input[idx1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[idx0],
      self.motion_base_rots_input[idx1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[idx0],
      self.motion_dof_poss_input[idx1],
      blend.unsqueeze(1),
    )
    print(
      f"动作加载完成：{self.input_frames} 帧 @ {self.input_fps} fps → "
      f"{self.output_frames} 帧 @ {self.output_fps} fps "
      f"({self.duration:.2f}s)"
    )

  def _lerp(self, a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return a * (1 - t) + b * t

  def _slerp(self, a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a)
    for i in range(a.shape[0]):
      out[i] = quat_slerp(a[i], b[i], float(t[i]))
    return out

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = times / self.duration
    idx0 = (phase * (self.input_frames - 1)).floor().long()
    idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - idx0
    return idx0, idx1, blend

  def _compute_velocities(self):
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    q_prev = self.motion_base_rots[:-2]
    q_next = self.motion_base_rots[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
    self.motion_base_ang_vels = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

  def get_next_state(self):
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset = self.current_idx >= self.output_frames
    if reset:
      self.current_idx = 0
    return state, reset


def run_sim(
  sim: Simulation,
  scene: Scene,
  robot: RobotType,
  input_file: str,
  output_fps: int,
  output_name: str,
  render: bool,
  renderer: OffscreenRenderer | None,
):
  motion = MotionLoader(
    motion_file=input_file,
    robot=robot,
    output_fps=output_fps,
    device=sim.device,
  )

  robot_entity: Entity = scene["robot"]
  joint_names = ROBOT_JOINT_NAMES[robot]
  robot_joint_indexes = robot_entity.find_joints(joint_names, preserve_order=True)[0]

  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  frames = []
  scene.reset()
  file_saved = False

  pbar = tqdm(
    total=motion.output_frames,
    desc="处理帧",
    unit="帧",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while not file_saved:
    (
      (
        base_pos,
        base_rot,
        base_lin_vel,
        base_ang_vel,
        dof_pos,
        dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot_entity.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot_entity.write_root_state_to_sim(root_states)

    joint_pos = robot_entity.data.default_joint_pos.clone()
    joint_vel = robot_entity.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = dof_pos
    joint_vel[:, robot_joint_indexes] = dof_vel
    robot_entity.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    log["joint_pos"].append(robot_entity.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot_entity.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot_entity.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(
      robot_entity.data.body_link_quat_w[0].cpu().numpy().copy()
    )
    log["body_lin_vel_w"].append(
      robot_entity.data.body_link_lin_vel_w[0].cpu().numpy().copy()
    )
    log["body_ang_vel_w"].append(
      robot_entity.data.body_link_ang_vel_w[0].cpu().numpy().copy()
    )

    frame_count += 1
    pbar.update(1)
    if frame_count % 100 == 0:
      pbar.set_description(f"处理帧 (t={frame_count / output_fps:.1f}s)")

    if reset_flag:
      file_saved = True
      pbar.close()

      print("\n合并数组...")
      for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ):
        log[k] = np.stack(log[k], axis=0)

      print("保存到 /tmp/motion.npz...")
      np.savez("/tmp/motion.npz", **log)

      print("上传到 Weights & Biases...")
      import wandb

      run = wandb.init(project="mimickit", name=output_name)
      print(f"[INFO]: 上传动作到 wandb: {output_name}")
      REGISTRY = "motions"
      artifact = run.log_artifact(
        artifact_or_path="/tmp/motion.npz",
        name=output_name,
        type=REGISTRY,
      )
      run.link_artifact(
        artifact=artifact,
        target_path=f"wandb-registry-{REGISTRY}/{output_name}",
      )
      print(f"[INFO]: 动作已保存到 wandb registry: {REGISTRY}/{output_name}")

      if render and frames:
        import mediapy as media

        print("生成视频...")
        media.write_video("./motion.mp4", frames, fps=output_fps)
        print("上传视频到 wandb...")
        wandb.log({"motion_video": wandb.Video("./motion.mp4", format="mp4")})

      wandb.finish()


def main(
  input_file: str,
  output_name: str,
  robot: RobotType | None = None,
  output_fps: int = 50,
  device: str = "cuda:0",
  render: bool = False,
):
  """将 MimicKit .pkl 动作文件转换为 .npz 并上传到 wandb。

  Args:
    input_file: 输入 .pkl 文件路径。
    output_name: wandb registry 中的 artifact 名称。
    robot: 机器人类型（go2 或 g1）。默认根据文件名自动检测。
    output_fps: 输出帧率（默认 50 Hz）。
    device: 计算设备。
    render: 是否渲染视频并上传到 wandb。
  """
  if robot is None:
    robot = _detect_robot(input_file)
    print(f"[INFO]: 自动检测机器人类型: {robot}")

  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA 不可用，回退到 CPU。")
    device = "cpu"

  scene, sim = _make_scene(robot, output_fps, device)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
    renderer.initialize()

  run_sim(
    sim=sim,
    scene=scene,
    robot=robot,
    input_file=input_file,
    output_fps=output_fps,
    output_name=output_name,
    render=render,
    renderer=renderer,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
