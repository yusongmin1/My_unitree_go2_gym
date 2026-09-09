from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_go2_hopping_flat_env_cfg
from .rl_cfg import unitree_go2_hopping_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Hopping-Flat-Unitree-Go2",
  env_cfg=unitree_go2_hopping_flat_env_cfg(),
  play_env_cfg=unitree_go2_hopping_flat_env_cfg(play=True),
  rl_cfg=unitree_go2_hopping_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
