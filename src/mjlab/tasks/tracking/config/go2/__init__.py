from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import unitree_go2_flat_tracking_env_cfg
from .rl_cfg import unitree_go2_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-Go2",
  env_cfg=unitree_go2_flat_tracking_env_cfg(),
  play_env_cfg=unitree_go2_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_go2_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-Go2-No-State-Estimation",
  env_cfg=unitree_go2_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_go2_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_go2_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
