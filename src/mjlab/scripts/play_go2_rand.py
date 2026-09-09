"""Play Go2 tracking with 50 envs and full domain randomization."""

import tyro

import mjlab
from mjlab.scripts.play import PlayConfig, run_play

_TASK_ID = "Mjlab-Tracking-Flat-Unitree-Go2"


def main() -> None:
  """Play trained Go2 policy under full training-time randomization."""
  args = tyro.cli(
    PlayConfig,
    default=PlayConfig(
      num_envs=50,
      keep_randomization=True,
    ),
    config=mjlab.TYRO_FLAGS,
  )
  run_play(_TASK_ID, args)


if __name__ == "__main__":
  main()
