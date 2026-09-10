"""Convert 100STYLE TensorDict memmap dataset to per-clip mjlab .npz motions."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import tyro
from tqdm import tqdm

import mjlab
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.entity import Entity

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


def _finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
  v = np.zeros_like(x)
  if x.shape[0] == 1:
    return v
  v[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
  v[0] = (x[1] - x[0]) / dt
  v[-1] = (x[-1] - x[-2]) / dt
  return v


def _quat_diff_ang_vel(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
  """Approximate body angular velocity (rad/s) from wxyz quaternions."""
  t = quat_wxyz.shape[0]
  w = np.zeros((t, 3), dtype=np.float32)
  if t < 2:
    return w
  # dq ≈ 0.5 * omega ⊗ q  => omega ≈ 2 * dq ⊗ q^{-1}
  for i in range(t - 1):
    q = quat_wxyz[i]
    qn = quat_wxyz[i + 1]
    # ensure same hemisphere
    if np.dot(q, qn) < 0:
      qn = -qn
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)
    # quat mul qn * q_inv
    w0, x0, y0, z0 = qn
    w1, x1, y1, z1 = q_inv
    dq = np.array(
      [
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
      ],
      dtype=np.float32,
    )
    # axis-angle from dq
    dq = dq / max(np.linalg.norm(dq), 1e-8)
    angle = 2.0 * np.arccos(np.clip(dq[0], -1.0, 1.0))
    s = np.sqrt(max(1.0 - dq[0] * dq[0], 0.0))
    if s < 1e-6:
      axis = np.zeros(3, dtype=np.float32)
    else:
      axis = dq[1:] / s
    w[i] = axis * (angle / dt)
  w[-1] = w[-2]
  return w


def _style_and_stem(source_path: str) -> tuple[str, str]:
  parts = Path(source_path).parts
  stem = Path(source_path).stem
  if "100STYLE" in parts:
    i = parts.index("100STYLE")
    style = parts[i + 1] if i + 1 < len(parts) else "unknown"
  elif "AMASS_hard" in parts:
    i = parts.index("AMASS_hard")
    # CMU/05/foo.npz -> CMU_05 ; CMU_ood/foo.npz -> CMU_ood
    rest = parts[i + 1 : -1]
    style = "_".join(rest) if rest else "AMASS_hard"
  else:
    style = Path(source_path).parent.name or "unknown"
  return style, stem


def _build_g1_model() -> tuple[mujoco.MjModel, list[int], list[int]]:
  """Compile standalone G1 model; return model, joint qpos addrs, body ids."""
  robot = Entity(get_g1_robot_cfg())
  # Drop keyframes if any (same as scene attach path).
  while robot.spec.keys:
    robot.spec.delete(robot.spec.keys[0])
  model = robot.spec.compile()
  joint_qpos_addrs: list[int] = []
  for name in G1_JOINT_NAMES:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
      raise ValueError(f"Joint '{name}' not found in G1 model.")
    joint_qpos_addrs.append(int(model.jnt_qposadr[jid]))
  # Bodies excluding world (id 0).
  body_ids = list(range(1, model.nbody))
  return model, joint_qpos_addrs, body_ids


def convert_segment(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_qpos_addrs: list[int],
  body_ids: list[int],
  root_pos: np.ndarray,
  root_quat_wxyz: np.ndarray,
  joint_pos: np.ndarray,
  fps: float,
) -> dict[str, np.ndarray]:
  t = root_pos.shape[0]
  dt = 1.0 / fps
  joint_vel = _finite_diff(joint_pos, dt)

  body_pos = np.zeros((t, len(body_ids), 3), dtype=np.float32)
  body_quat = np.zeros((t, len(body_ids), 4), dtype=np.float32)
  out_joint_pos = joint_pos.astype(np.float32, copy=True)
  out_joint_vel = joint_vel.astype(np.float32, copy=True)

  for i in range(t):
    data.qpos[:] = 0.0
    data.qpos[0:3] = root_pos[i]
    data.qpos[3:7] = root_quat_wxyz[i]
    for j, adr in enumerate(joint_qpos_addrs):
      data.qpos[adr] = joint_pos[i, j]
    mujoco.mj_forward(model, data)
    for bi, bid in enumerate(body_ids):
      body_pos[i, bi] = data.xpos[bid]
      body_quat[i, bi] = data.xquat[bid]

  body_lin_vel = _finite_diff(body_pos.reshape(t, -1), dt).reshape(t, len(body_ids), 3)
  body_ang_vel = np.stack(
    [_quat_diff_ang_vel(body_quat[:, bi], dt) for bi in range(len(body_ids))],
    axis=1,
  ).astype(np.float32)

  return {
    "fps": np.array([fps], dtype=np.float64),
    "joint_pos": out_joint_pos,
    "joint_vel": out_joint_vel,
    "body_pos_w": body_pos,
    "body_quat_w": body_quat,
    "body_lin_vel_w": body_lin_vel.astype(np.float32),
    "body_ang_vel_w": body_ang_vel,
  }


def main(
  input_dir: str = "/home/zju/Downloads/amass_hard",
  output_dir: str = "motions/amass_hard",
  fps: float = 50.0,
  max_clips: int | None = None,
  skip_existing: bool = True,
) -> None:
  """Convert G1 TensorDict memmap clips (100STYLE / AMASS_hard) to mjlab .npz.

  Args:
    input_dir: Dataset root with ``id_label.json``, ``meta_motion.json``,
      and ``_tensordict/``.
    output_dir: Local output directory under the mjlab project.
    fps: Assumed source frame rate (dataset has no fps field; default 50).
    max_clips: If set, only convert the first N clips (for smoke tests).
    skip_existing: Skip clips whose .npz already exists.
  """
  root = Path(input_dir)
  out_root = Path(output_dir)
  out_root.mkdir(parents=True, exist_ok=True)

  with open(root / "id_label.json") as f:
    labels: list[dict] = json.load(f)
  with open(root / "meta_motion.json") as f:
    meta = json.load(f)
  with open(root / "_tensordict" / "meta.json") as f:
    td_meta = json.load(f)

  if list(meta["joint_names"]) != list(G1_JOINT_NAMES):
    raise ValueError("Dataset joint_names do not match mjlab G1 order.")

  n_total = int(td_meta["shape"][0])
  td = root / "_tensordict"
  root_pos_mm = np.memmap(
    td / "root_pos_w.memmap", dtype=np.float16, mode="r", shape=(n_total, 3)
  )
  root_quat_mm = np.memmap(
    td / "root_quat_w.memmap", dtype=np.float16, mode="r", shape=(n_total, 4)
  )
  joint_pos_mm = np.memmap(
    td / "joint_pos.memmap", dtype=np.float16, mode="r", shape=(n_total, 29)
  )

  model, joint_qpos_addrs, body_ids = _build_g1_model()
  data = mujoco.MjData(model)

  clips = labels[:max_clips] if max_clips is not None else labels
  converted = 0
  skipped = 0

  pbar = tqdm(clips, desc=root.name, unit="clip")
  for idx, lab in enumerate(pbar):
    style, stem = _style_and_stem(lab["source_path"])
    start = int(lab["segment_start"])
    end = int(lab["segment_end"])
    if idx < len(meta["starts"]):
      start = int(meta["starts"][idx])
      end = int(meta["ends"][idx])

    out_dir = out_root / style
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{start:07d}_{end:07d}.npz"
    if skip_existing and out_path.exists():
      skipped += 1
      continue

    rp = np.asarray(root_pos_mm[start:end], dtype=np.float32)
    rq = np.asarray(root_quat_mm[start:end], dtype=np.float32)
    jp = np.asarray(joint_pos_mm[start:end], dtype=np.float32)
    rq /= np.linalg.norm(rq, axis=-1, keepdims=True).clip(min=1e-8)

    motion = convert_segment(
      model=model,
      data=data,
      joint_qpos_addrs=joint_qpos_addrs,
      body_ids=body_ids,
      root_pos=rp,
      root_quat_wxyz=rq,
      joint_pos=jp,
      fps=fps,
    )
    np.savez_compressed(out_path, **motion)
    converted += 1
    pbar.set_postfix(done=converted, skip=skipped, style=style[:16])

  print(f"[INFO]: Converted {converted}, skipped {skipped}, out={out_root.resolve()}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
