"""Export a tracking checkpoint to TorchScript ``policy.pt`` for C++ deploy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class ExportConfig:
  task: str = "Mjlab-Tracking-Flat-Unitree-Go2-No-State-Estimation"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  output: str | None = None
  """Output ``.pt`` path. Default: ``<checkpoint_dir>/policy.pt``."""
  device: str = "cpu"
  num_envs: int = 1


def main() -> None:
  configure_torch_backends()
  cfg = tyro.cli(ExportConfig, config=mjlab.TYRO_FLAGS)

  if cfg.task not in list_tasks():
    raise SystemExit(f"Unknown task '{cfg.task}'. Available: {list_tasks()}")
  if cfg.checkpoint_file is None:
    raise SystemExit("--checkpoint-file is required")
  ckpt = Path(cfg.checkpoint_file)
  if not ckpt.exists():
    raise SystemExit(f"Checkpoint not found: {ckpt}")

  env_cfg = load_env_cfg(cfg.task, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  agent_cfg = load_rl_cfg(cfg.task)

  if "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  ):
    motion_cmd = env_cfg.commands["motion"]
    if cfg.motion_file is not None:
      motion_cmd.motion_file = cfg.motion_file
    elif not motion_cmd.motion_file or not Path(motion_cmd.motion_file).exists():
      raise SystemExit(
        "Provide --motion-file /path/to/motion.npz "
        "(required for tracking TorchScript export)."
      )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(cfg.task) or MotionTrackingOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=cfg.device)

  if not isinstance(runner, MotionTrackingOnPolicyRunner):
    raise SystemExit(f"Task runner is {type(runner)}, expected tracking runner.")

  out = Path(cfg.output) if cfg.output else ckpt.parent / "policy.pt"
  runner.export_policy_to_torchscript(str(out.parent), out.name)
  # Quick sanity check.
  scripted = torch.jit.load(str(out), map_location="cpu")
  obs = torch.zeros(1, runner.alg.get_policy().obs_dim)
  time_step = torch.zeros(1, 1, dtype=torch.int64)
  result = scripted(obs, time_step)
  actions = result[0] if isinstance(result, tuple) else result
  print(f"[INFO] Sanity check OK: actions shape={tuple(actions.shape)}")
  print(f"[INFO] Wrote {out}")
  env.close()


if __name__ == "__main__":
  main()
