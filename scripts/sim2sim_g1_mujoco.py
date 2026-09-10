"""Sim-to-sim: run an mjlab / Isaac G1 tracking policy in MuJoCo (explicit PD torque).

PD gains / armature / per-joint action scales match
``mjlab.asset_zoo.robots.unitree_g1.g1_constants`` (NATURAL_FREQ=10 Hz, ζ=2):

    motor       armature      kp         kd        effort   action_scale (0.25*e/kp)
    5020        0.003609725   14.2506    0.9072    25       0.4386   (shoulder/elbow/wrist_roll)
    7520_14     0.010177520   40.1792    2.5579    88       0.5475   (hip_pitch/yaw, waist_yaw)
    7520_22     0.025101925   99.0984    6.3088   139       0.3507   (hip_roll, knee)
    4010        0.004250000   16.7783    1.0681     5       0.0745   (wrist_pitch/yaw)
    2×5020      0.007219450   28.5012    1.8144    50       0.4386   (ankle, waist_pitch/roll)

Anchor body is ``torso_link`` (same as G1 tracking env). Observation default is
Wo-State-Estimation (154-D); pass ``--with_state_estimation`` for 160-D.

.. code-block:: bash

    cd /path/to/mjlab-dev-go2-mimic
    python scripts/sim2sim_g1_mujoco.py \\
        --policy path/to/policy.pt \\
        --motion_file motions/g1_fallandgetup1_850_940.npz \\
        --match_isaac

    # plant check (PD tracks reference joints, no policy)
    python scripts/sim2sim_g1_mujoco.py --policy unused.pt --motion_file ... --replay --match_isaac --headless
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import time

import numpy as np

parser = argparse.ArgumentParser(description="G1 sim2sim in MuJoCo (mjlab kp/kd).")
parser.add_argument("--policy", type=str, required=True, help="TorchScript policy.pt / rsl_rl model_*.pt / onnx.")
parser.add_argument("--motion_file", type=str, required=True, help="Reference motion npz.")
parser.add_argument("--xml", type=str, default=None, help="MuJoCo g1 xml (default: mjlab unitree_g1/xmls/g1.xml).")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed.")
parser.add_argument("--kp", type=float, default=None, help="Uniform PD stiffness override.")
parser.add_argument("--kd", type=float, default=None, help="Uniform PD damping override.")
parser.add_argument(
    "--action_scale",
    type=float,
    default=None,
    help="Uniform action scale override (default: per-joint mjlab G1_ACTION_SCALE).",
)
parser.add_argument(
    "--with_state_estimation",
    action="store_true",
    help="Include motion_anchor_pos_b + base_lin_vel (160-D). Default is Wo-SE 154-D.",
)
parser.add_argument("--no_loop", action="store_true", help="Stop at end of motion.")
parser.add_argument("--no_ghost", action="store_true", help="Do not draw reference ghost.")
parser.add_argument(
    "--match_isaac",
    action="store_true",
    help="Set joint armature to mjlab values and zero passive dof damping/frictionloss.",
)
parser.add_argument("--headless", action="store_true", help="No viewer; one motion pass then exit.")
parser.add_argument(
    "--replay",
    action="store_true",
    help="Bypass policy; PD-track reference joint angles (plant sanity check).",
)
args = parser.parse_args()

import mujoco
import mujoco.viewer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MJLAB_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DEFAULT_XML = os.path.join(
    _MJLAB_ROOT, "src", "mjlab", "asset_zoo", "robots", "unitree_g1", "xmls", "g1.xml"
)

# Joint order used by mjlab csv_to_npz / G1 entity (matches g1.xml qpos order).
JOINT_NAMES = [
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
]
N = len(JOINT_NAMES)  # 29

# mjlab g1_constants: armature from two-stage planetary reflected inertia.
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520041
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# (regex, kp, kd, armature, effort_limit)
_ACTUATOR_GROUPS = [
    (r".*_elbow_joint|.*_shoulder_pitch_joint|.*_shoulder_roll_joint|.*_shoulder_yaw_joint|.*_wrist_roll_joint",
     STIFFNESS_5020, DAMPING_5020, ARMATURE_5020, 25.0),
    (r".*_hip_pitch_joint|.*_hip_yaw_joint|waist_yaw_joint",
     STIFFNESS_7520_14, DAMPING_7520_14, ARMATURE_7520_14, 88.0),
    (r".*_hip_roll_joint|.*_knee_joint",
     STIFFNESS_7520_22, DAMPING_7520_22, ARMATURE_7520_22, 139.0),
    (r".*_wrist_pitch_joint|.*_wrist_yaw_joint",
     STIFFNESS_4010, DAMPING_4010, ARMATURE_4010, 5.0),
    (r"waist_pitch_joint|waist_roll_joint",
     2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 2.0 * ARMATURE_5020, 50.0),
    (r".*_ankle_pitch_joint|.*_ankle_roll_joint",
     2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 2.0 * ARMATURE_5020, 50.0),
]


def _joint_props(name: str):
    for pat, kp, kd, arm, effort in _ACTUATOR_GROUPS:
        if re.fullmatch(pat, name):
            return kp, kd, arm, effort
    raise KeyError(f"No actuator group matched joint '{name}'")


# Default pose = mjlab KNEES_BENT_KEYFRAME / G1 init_state
DEFAULT_JOINT_POS = {n: 0.0 for n in JOINT_NAMES}
for n in JOINT_NAMES:
    if n.endswith("hip_pitch_joint"):
        DEFAULT_JOINT_POS[n] = -0.312
    elif n.endswith("knee_joint"):
        DEFAULT_JOINT_POS[n] = 0.669
    elif n.endswith("ankle_pitch_joint"):
        DEFAULT_JOINT_POS[n] = -0.363
    elif n.endswith("elbow_joint"):
        DEFAULT_JOINT_POS[n] = 0.6
DEFAULT_JOINT_POS["left_shoulder_roll_joint"] = 0.2
DEFAULT_JOINT_POS["left_shoulder_pitch_joint"] = 0.2
DEFAULT_JOINT_POS["right_shoulder_roll_joint"] = -0.2
DEFAULT_JOINT_POS["right_shoulder_pitch_joint"] = 0.2

# Tracking keypoint bodies for ghost (same as G1 env body_names)
GHOST_BODIES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
GHOST_EDGES = [
    ("pelvis", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("pelvis", "torso_link"),
    ("torso_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]

ANCHOR_BODY = "torso_link"

# ----------------------------------------------------------------- setup
xml_path = args.xml or _DEFAULT_XML
# g1.xml has no ground plane; inject one so the robot does not freefall forever.
spec = mujoco.MjSpec.from_file(xml_path)
floor = spec.worldbody.add_geom()
floor.name = "floor"
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size[:] = [0, 0, 0.05]
floor.rgba[:] = [0.25, 0.35, 0.45, 1.0]
model = spec.compile()
data = mujoco.MjData(model)
model.opt.timestep = 0.005
ctrl_dt = 0.02
decimation = round(ctrl_dt / model.opt.timestep)

motion = np.load(args.motion_file, allow_pickle=True)
fps = float(np.asarray(motion["fps"]).item())
assert abs(fps * ctrl_dt - 1.0) < 1e-6, f"motion fps ({fps}) must match control rate ({1 / ctrl_dt})"

if "joint_names" in motion:
    npz_joint_names = [str(n) for n in np.asarray(motion["joint_names"]).ravel().tolist()]
else:
    npz_joint_names = list(JOINT_NAMES)
    print("[INFO]: npz has no joint_names; assuming mjlab G1 order")
assert len(npz_joint_names) == N, f"expected {N} joints, got {len(npz_joint_names)}"

ref_joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
ref_joint_vel = np.asarray(motion["joint_vel"], dtype=np.float64)
ref_all_body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)  # (T, B, 3)
ref_all_body_quat = np.asarray(motion["body_quat_w"], dtype=np.float64)  # (T, B, 4) wxyz
ref_all_body_lin = np.asarray(motion["body_lin_vel_w"], dtype=np.float64)
ref_all_body_ang = np.asarray(motion["body_ang_vel_w"], dtype=np.float64)
T = ref_joint_pos.shape[0]

# Body names in npz == mujoco bodies excluding world (index 0 = pelvis)
mj_body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)]
if "body_names" in motion:
    npz_body_names = [str(n) for n in np.asarray(motion["body_names"]).ravel().tolist()]
else:
    npz_body_names = mj_body_names
    assert len(npz_body_names) == ref_all_body_pos.shape[1], (
        f"body count mismatch: model {len(npz_body_names)} vs npz {ref_all_body_pos.shape[1]}"
    )
    print("[INFO]: npz has no body_names; assuming mujoco order without world")

anchor_npz_idx = npz_body_names.index(ANCHOR_BODY)
ref_anchor_pos = ref_all_body_pos[:, anchor_npz_idx]
ref_anchor_quat = ref_all_body_quat[:, anchor_npz_idx]
# Root (pelvis) for reset / free-joint state
pelvis_npz_idx = npz_body_names.index("pelvis")
ref_pelvis_pos = ref_all_body_pos[:, pelvis_npz_idx]
ref_pelvis_quat = ref_all_body_quat[:, pelvis_npz_idx]
ref_pelvis_lin = ref_all_body_lin[:, pelvis_npz_idx]
ref_pelvis_ang = ref_all_body_ang[:, pelvis_npz_idx]

# name maps: policy/npz order <-> mujoco qpos/dof
mj_qpos_adr, mj_dof_adr = [], []
for name in npz_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert jid >= 0, f"joint '{name}' not in mujoco model"
    mj_qpos_adr.append(model.jnt_qposadr[jid])
    mj_dof_adr.append(model.jnt_dofadr[jid])
mj_qpos_adr = np.asarray(mj_qpos_adr)
mj_dof_adr = np.asarray(mj_dof_adr)

default_q = np.array([DEFAULT_JOINT_POS[n] for n in npz_joint_names], dtype=np.float64)
kp_j = np.zeros(N, dtype=np.float64)
kd_j = np.zeros(N, dtype=np.float64)
arm_j = np.zeros(N, dtype=np.float64)
effort_j = np.zeros(N, dtype=np.float64)
scale_j = np.zeros(N, dtype=np.float64)
for i, name in enumerate(npz_joint_names):
    kp, kd, arm, effort = _joint_props(name)
    kp_j[i] = args.kp if args.kp is not None else kp
    kd_j[i] = args.kd if args.kd is not None else kd
    arm_j[i] = arm
    effort_j[i] = effort
    scale_j[i] = args.action_scale if args.action_scale is not None else (0.25 * effort / kp)

print("kp:", np.array2string(kp_j, precision=3, suppress_small=True))
print("kd:", np.array2string(kd_j, precision=3, suppress_small=True))
print("action_scale:", np.array2string(scale_j, precision=4, suppress_small=True))

if args.match_isaac:
    for adr, arm in zip(mj_dof_adr, arm_j):
        model.dof_armature[adr] = arm
        model.dof_damping[adr] = 0.0
        model.dof_frictionloss[adr] = 0.0
    print("[INFO]: joint armature/damping/frictionloss matched to mjlab Isaac setup")

free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
free_adr = model.jnt_qposadr[free_jid]
anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY)
assert anchor_body_id >= 0

obs_dim = 160 if args.with_state_estimation else 154
print(f"obs_dim={obs_dim} ({'with' if args.with_state_estimation else 'wo'} state estimation)")


def load_policy(path: str):
    """obs(obs_dim,) -> action(29,)."""
    if path.endswith(".pt"):
        import torch

        try:
            module = torch.jit.load(path, map_location="cpu").eval()

            @torch.no_grad()
            def run_ts(obs: np.ndarray) -> np.ndarray:
                out = module(torch.from_numpy(obs.astype(np.float32))[None, :], torch.zeros(1, 1, dtype=torch.long))
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return out[0].numpy()

            # smoke-check dim
            try:
                run_ts(np.zeros(obs_dim, dtype=np.float32))
            except Exception as e:
                print(f"[WARN]: torchscript smoke test failed ({e}); continuing anyway")
            return run_ts
        except RuntimeError:
            pass

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        actor_layers = []
        i = 0
        while f"actor.{i}.weight" in sd:
            actor_layers.append(torch.nn.Linear(sd[f"actor.{i}.weight"].shape[1], sd[f"actor.{i}.weight"].shape[0]))
            actor_layers.append(torch.nn.ELU())
            i += 2
        del actor_layers[-1]
        actor = torch.nn.Sequential(*actor_layers)
        actor.load_state_dict({k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}, strict=True)
        actor.eval()
        norm = ckpt["obs_norm_state_dict"]
        norm_mean, norm_std = norm["_mean"][0], norm["_std"][0]

        @torch.no_grad()
        def run_ckpt(obs: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(obs.astype(np.float32))
            x = (x - norm_mean) / (norm_std + 1e-2)
            return actor(x).numpy()

        return run_ckpt

    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def run(obs: np.ndarray) -> np.ndarray:
        return session.run(["actions"], {"obs": obs[None, :].astype(np.float32), "time_step": np.array([[0]], dtype=np.int64)})[0][0]

    return run


policy = (lambda obs: np.zeros(N, dtype=np.float32)) if args.replay else load_policy(args.policy)
print(f"policy: {'REPLAY' if args.replay else args.policy}")
print(f"motion: {args.motion_file} ({T} frames @ {fps} fps, {T / fps:.2f} s)")

ghost_edge_idx = []
if not args.no_ghost:
    for a, b in GHOST_EDGES:
        if a in npz_body_names and b in npz_body_names:
            ghost_edge_idx.append((npz_body_names.index(a), npz_body_names.index(b)))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """wxyz -> 3x3 (isaaclab / mjlab matrix_from_quat)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def clip_effort(effort: np.ndarray) -> np.ndarray:
    return np.clip(effort, -effort_j, effort_j)


def reset(t0: int = 0):
    mujoco.mj_resetData(model, data)
    data.qpos[free_adr : free_adr + 3] = ref_pelvis_pos[t0]
    data.qpos[free_adr + 3 : free_adr + 7] = ref_pelvis_quat[t0]
    data.qpos[mj_qpos_adr] = ref_joint_pos[t0]
    R = quat_to_mat(ref_pelvis_quat[t0])
    data.qvel[0:3] = ref_pelvis_lin[t0]
    data.qvel[3:6] = R.T @ ref_pelvis_ang[t0]
    data.qvel[mj_dof_adr] = ref_joint_vel[t0]
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def draw_ghost(scn, t: int):
    scn.ngeom = 0
    for a, b in ghost_edge_idx:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.rgba[:] = [0.2, 1.0, 0.2, 0.6]
        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_LINE,
            0.01,
            ref_all_body_pos[t, a].astype(np.float64),
            ref_all_body_pos[t, b].astype(np.float64),
        )
        scn.ngeom += 1


def apply_pd(q_des: np.ndarray):
    """Write joint torques into qfrc_applied (g1.xml has no <actuator>)."""
    q = data.qpos[mj_qpos_adr]
    qd = data.qvel[mj_dof_adr]
    tau = clip_effort(kp_j * (q_des - q) - kd_j * qd)
    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[mj_dof_adr] = tau


# ----------------------------------------------------------------- loop
obs = np.zeros(obs_dim, dtype=np.float32)
last_action = np.zeros(N, dtype=np.float32)
frame_dt = ctrl_dt / max(args.speed, 1e-6)


@contextlib.contextmanager
def maybe_viewer():
    if args.headless:
        yield None
    else:
        with mujoco.viewer.launch_passive(model, data) as v:
            yield v


with maybe_viewer() as viewer:
    reset(0)
    # settle: hold frame-0 with PD for 0.5 s
    for _ in range(int(0.5 / model.opt.timestep)):
        apply_pd(ref_joint_pos[0])
        mujoco.mj_step(model, data)

    t = 0
    step = 0
    err_pos = []
    while True:
        if viewer is None:
            if step >= T:
                print(f"headless pass finished. mean anchor pos error: {np.mean(err_pos):.3f} m")
                break
        elif not viewer.is_running():
            break

        start = time.time()

        # ---- observation ------------------------------------------------------
        cmd = np.concatenate([ref_joint_pos[t], ref_joint_vel[t]])  # (58,)
        p_anc = data.xpos[anchor_body_id].copy()
        q_anc = data.xquat[anchor_body_id].copy()  # wxyz
        R_anc = quat_to_mat(q_anc)
        # mjlab: mat[..., :2].reshape -> first TWO COLUMNS
        R_rel = R_anc.T @ quat_to_mat(ref_anchor_quat[t])
        anchor_ori_b = R_rel[:, :2].reshape(-1)  # (6,)
        # pelvis / free-joint rates (body frame ang vel already in qvel[3:6])
        R_pelvis = quat_to_mat(data.qpos[free_adr + 3 : free_adr + 7])
        base_ang_vel_b = data.qvel[3:6].copy()
        joint_pos_rel = data.qpos[mj_qpos_adr] - default_q
        joint_vel_rel = data.qvel[mj_dof_adr].copy()

        parts = [cmd]
        if args.with_state_estimation:
            anchor_pos_b = R_anc.T @ (ref_anchor_pos[t] - p_anc)
            base_lin_vel_b = R_pelvis.T @ data.qvel[0:3]
            parts.extend([anchor_pos_b, anchor_ori_b, base_lin_vel_b, base_ang_vel_b])
        else:
            parts.extend([anchor_ori_b, base_ang_vel_b])
        parts.extend([joint_pos_rel, joint_vel_rel, last_action])
        obs[:] = np.concatenate(parts).astype(np.float32)

        # ---- policy -----------------------------------------------------------
        if args.replay:
            action = ((ref_joint_pos[t] - default_q) / scale_j).astype(np.float32)
        else:
            action = np.asarray(policy(obs), dtype=np.float32).reshape(-1)
            assert action.shape[0] == N, f"policy action dim {action.shape[0]} != {N}"
        last_action = action

        q_des = default_q + scale_j * action
        for _ in range(decimation):
            apply_pd(q_des)
            mujoco.mj_step(model, data)

        err_pos.append(np.linalg.norm(data.xpos[anchor_body_id] - ref_anchor_pos[t]))
        step += 1
        t += 1
        if t >= T:
            if args.no_loop:
                print(f"motion finished. mean anchor pos error: {np.mean(err_pos):.3f} m")
                break
            t = 0

        if viewer is not None:
            if not args.no_ghost:
                draw_ghost(viewer.user_scn, t)
            viewer.sync()

        if step % 10 == 0:
            print(
                f"step {step:4d} t={t:3d}  anchor err {err_pos[-1]:.3f} m  "
                f"z_err {data.xpos[anchor_body_id, 2] - ref_anchor_pos[t, 2]:+.3f}"
            )

        remaining = frame_dt - (time.time() - start)
        if remaining > 0:
            time.sleep(remaining)
