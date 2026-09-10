"""将 Go2 广义坐标 CSV 动作文件转换为 .npz 格式并上传到 wandb。

输入 CSV 格式（无表头，逗号分隔，共 19 列）：
  列 0:3  — 根节点位置 (x, y, z)，单位米
  列 3:7  — 根节点四元数 (x, y, z, w)，xyzw 顺序
  列 7:19 — 12 个关节角度（弧度），顺序如下：

  Go2 关节顺序（深度优先，与 MimicKit/mjlab 一致）：
    0  FL_hip_joint      前左髋
    1  FL_thigh_joint    前左大腿
    2  FL_calf_joint     前左小腿
    3  FR_hip_joint      前右髋
    4  FR_thigh_joint    前右大腿
    5  FR_calf_joint     前右小腿
    6  RL_hip_joint      后左髋
    7  RL_thigh_joint    后左大腿
    8  RL_calf_joint     后左小腿
    9  RR_hip_joint      后右髋
    10 RR_thigh_joint    后右大腿
    11 RR_calf_joint     后右小腿
"""

from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.go2.env_cfgs import unitree_go2_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

# Go2 关节顺序（深度优先，与 convert_mimickit.py 中 GO2_JOINT_NAMES 完全一致）。
# CSV 第 7~18 列依次对应以下关节：
GO2_JOINT_NAMES = (
  "FL_hip_joint",    # col 7
  "FL_thigh_joint",  # col 8
  "FL_calf_joint",   # col 9
  "FR_hip_joint",    # col 10
  "FR_thigh_joint",  # col 11
  "FR_calf_joint",   # col 12
  "RL_hip_joint",    # col 13
  "RL_thigh_joint",  # col 14
  "RL_calf_joint",   # col 15
  "RR_hip_joint",    # col 16
  "RR_thigh_joint",  # col 17
  "RR_calf_joint",   # col 18
)

_EXPECTED_COLS = 19  # 3 pos + 4 quat + 12 dof


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: float,
    output_fps: float,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / input_fps
    self.output_dt = 1.0 / output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  @staticmethod
  def _has_header(csv_path: str) -> bool:
    """检查 CSV 首行是否为表头（含非数字字符）。"""
    with open(csv_path) as f:
      first_line = f.readline().strip()
    if not first_line:
      return False
    try:
      [float(v) for v in first_line.split(",")]
      return False
    except ValueError:
      return True

  def _load_csv(self, csv_path: str) -> torch.Tensor:
    header_rows = 1 if self._has_header(csv_path) else 0
    if self.line_range is None:
      data = np.loadtxt(csv_path, delimiter=",", skiprows=header_rows)
    else:
      # line_range 始终指数据行编号（1-indexed，不含表头）
      data = np.loadtxt(
        csv_path,
        delimiter=",",
        skiprows=header_rows + self.line_range[0] - 1,
        max_rows=self.line_range[1] - self.line_range[0] + 1,
      )
    tensor = torch.from_numpy(data).to(torch.float32).to(self.device)
    if tensor.ndim == 1:
      tensor = tensor.unsqueeze(0)
    return tensor

  def _load_motion(self):
    motion = self._load_csv(self.motion_file)

    if motion.shape[1] < _EXPECTED_COLS:
      raise ValueError(
        f"CSV 列数不足：至少需要 {_EXPECTED_COLS} 列"
        f"（3 位置 + 4 四元数 + 12 DOF），实际得到 {motion.shape[1]} 列。"
      )
    # se3_trajopt 导出常含 vel/tau 后缀列；只取广义坐标前 19 列。
    if motion.shape[1] > _EXPECTED_COLS:
      print(
        f"[INFO]: CSV 有 {motion.shape[1]} 列，仅使用前 {_EXPECTED_COLS} 列 "
        "（pos + quat_xyzw + dof）。"
      )
      motion = motion[:, :_EXPECTED_COLS]

    self.motion_base_poss_input = motion[:, :3]   # (T, 3) xyz
    quat_xyzw = motion[:, 3:7]                    # (T, 4) xyzw
    # 转换为 wxyz（mjlab 内部约定）
    self.motion_base_rots_input = quat_xyzw[:, [3, 0, 1, 2]]  # (T, 4) wxyz
    self.motion_dof_poss_input = motion[:, 7:]    # (T, 12)

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

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
      f"动作插值完成：输入 {self.input_frames} 帧 @ {self.input_fps} fps → "
      f"输出 {self.output_frames} 帧 @ {self.output_fps} fps "
      f"（时长 {self.duration:.2f}s）"
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
  input_file: str,
  input_fps: float,
  output_fps: float,
  output_name: str,
  render: bool,
  line_range: tuple[int, int] | None,
  renderer: OffscreenRenderer | None,
  output_file: str | None = None,
):
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(GO2_JOINT_NAMES, preserve_order=True)[0]

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
      (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = dof_pos
    joint_vel[:, robot_joint_indexes] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

    frame_count += 1
    pbar.update(1)
    if frame_count % 100 == 0:
      pbar.set_description(f"处理帧 (t={frame_count / output_fps:.1f}s)")

    if reset_flag:
      file_saved = True
      pbar.close()

      print("\n合并数组并保存...")
      for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      ):
        log[k] = np.stack(log[k], axis=0)

      if output_file is not None:
        # 仅保存到本地，跳过 wandb 上传。
        print(f"保存到 {output_file} ...")
        np.savez(output_file, **log)
        if render and frames:
          import mediapy as media

          print("生成视频...")
          media.write_video("./motion.mp4", frames, fps=output_fps)
        return

      print("保存到 /tmp/motion.npz...")
      np.savez("/tmp/motion.npz", **log)

      print("上传到 Weights & Biases...")
      import wandb

      run = wandb.init(project="gc_go2", name=output_name)
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
  input_fps: float = 50.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
  output_file: str | None = None,
):
  """将 Go2 广义坐标 CSV（7+12 列）转换为 .npz 并上传到 wandb。

  Args:
    input_file: 输入 CSV 文件路径（无表头，19 列：pos3 + quat_xyzw4 + dof12）。
    output_name: wandb registry 中的 artifact 名称。
    input_fps: CSV 的原始帧率。
    output_fps: 输出/仿真帧率。
    device: 计算设备。
    render: 是否渲染视频并上传到 wandb。
    line_range: 只处理 CSV 的第 [a, b] 行（1-indexed，闭区间）。
    output_file: 本地 .npz 输出路径；设置后仅保存本地，跳过 wandb 上传。
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[WARNING]: CUDA 不可用，回退到 CPU。")
    device = "cpu"

  from mjlab.tasks.tracking.config.go2.env_cfgs import unitree_go2_flat_tracking_env_cfg

  scene = Scene(unitree_go2_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

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
    input_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    output_name=output_name,
    render=render,
    line_range=line_range,
    renderer=renderer,
    output_file=output_file,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
