from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.Go2_MoB.Go2_Jump.Go2_Jump import Go2_Jump_Robot
from legged_gym.envs.Go2_MoB.Go2_Jump.Go2_Jump_Config import Go2_Jump_Cfg_Yu,Go2_Jump_PPO_Yu


from legged_gym.envs.Go2_MoB.Go2_Trot.Go2_Trot import Go2_Trot_Robot
from legged_gym.envs.Go2_MoB.Go2_Trot.Go2_Trot_Config import Go2_Trot_Cfg_Yu,Go2_Trot_PPO_Yu

from legged_gym.envs.Go2_Flip.Go2_BackFlip.Go2_BackFlip import Go2_BackFlip_Robot
from legged_gym.envs.Go2_Flip.Go2_BackFlip.Go2_BackFlip_Config import Go2_BackFlip_Cfg_Yu, Go2_BackFlip_PPO_Yu

from legged_gym.envs.Go2_Flip.Go2_Spring_Jump.Go2_Spring_Jump import Go2_Spring_Jump_Robot
from legged_gym.envs.Go2_Flip.Go2_Spring_Jump.Go2_Spring_Jump_Config import Go2_Spring_Jump_Cfg_Yu, Go2_Spring_Jump_PPO_Yu

from legged_gym.envs.Go2_Stand.Go2_Handstand.Go2_Handstand import Go2_Handstand_Robot
from legged_gym.envs.Go2_Stand.Go2_Handstand.Go2_Handstand_Config import Go2_Handstand_Cfg_Yu,Go2_Handstand_PPO_Yu

from legged_gym.envs.Go2_Stand.Go2_Leggedstand.Go2_Leggedstand import Go2_Leggedstand_Robot
from legged_gym.envs.Go2_Stand.Go2_Leggedstand.Go2_Leggedstand_Config import Go2_Leggedstand_Cfg_Yu,Go2_Leggedstand_PPO_Yu

# ========== DreamwaQ 任务（纯 DreamwaQ 算法）==========
from legged_gym.envs.Go2_DreamWaQ.Go2_Stairs_DreamWaQ import Go2_Stairs_DreamWaQ_Robot
from legged_gym.envs.Go2_DreamWaQ.Go2_Stairs_DreamWaQ_Config import Go2_Stairs_DreamWaQ_Cfg_Yu, Go2_Stairs_DreamWaQ_PPO_Yu

# ========== AMP + DreamwaQ 任务（AMP 判别器 + DreamwaQ 算法）==========
from legged_gym.envs.Go2_AMP_DreamWaQ.Go2_Stairs_AMP_DreamWaQ import Go2_Stairs_AMP_DreamWaQ_Robot
from legged_gym.envs.Go2_AMP_DreamWaQ.Go2_Stairs_AMP_DreamWaQ_Config import Go2_Stairs_AMP_DreamWaQ_Cfg_Yu, Go2_Stairs_AMP_DreamWaQ_PPO_Yu

# ========== CTS 任务（Concurrent Teacher-Student 算法）==========
from legged_gym.envs.Go2_Cts.Go2_Cts import Go2_Cts_Robot
from legged_gym.envs.Go2_Cts.Go2_Cts_Config import Go2_Cts_Cfg_Yu, Go2_Cts_PPO_Yu

# ========== AMP + CTS 任务（AMP 判别器 + CTS 算法）==========
from legged_gym.envs.Go2_AMP_Cts.Go2_AMP_Cts import Go2_AMP_Cts_Robot
from legged_gym.envs.Go2_AMP_Cts.Go2_AMP_Cts_Config import Go2_AMP_Cts_Cfg_Yu, Go2_AMP_Cts_PPO_Yu

# ========== AMP Teacher-Student 任务（AMP 训练 teacher + LSTM 蒸馏 student）==========
from legged_gym.envs.base.legged_robot_amp_ts import LeggedRobotAMP_TS
from legged_gym.envs.Go2_AMP_Ts.Go2_AMP_Ts_Config import Go2_AMP_Ts_Cfg_Yu, Go2_AMP_Ts_PPO_Yu
from legged_gym.envs.Go2_AMP_Ts.Go2_AMP_Ts_Student_Config import Go2_AMP_Ts_Student_Cfg_Yu, Go2_AMP_Ts_Student_PPO_Yu

from legged_gym.utils.task_registry import task_registry


# ===== 标准 PPO 任务（main 原有特技任务）=====
task_registry.register( "go2_trot", Go2_Trot_Robot, Go2_Trot_Cfg_Yu(), Go2_Trot_PPO_Yu())
task_registry.register( "go2_jump", Go2_Jump_Robot, Go2_Jump_Cfg_Yu(), Go2_Jump_PPO_Yu())
task_registry.register( "go2_handstand", Go2_Handstand_Robot, Go2_Handstand_Cfg_Yu(), Go2_Handstand_PPO_Yu())
task_registry.register( "go2_leggedstand", Go2_Leggedstand_Robot, Go2_Leggedstand_Cfg_Yu(), Go2_Leggedstand_PPO_Yu())
task_registry.register( "go2_spring_jump", Go2_Spring_Jump_Robot, Go2_Spring_Jump_Cfg_Yu(), Go2_Spring_Jump_PPO_Yu())
task_registry.register( "go2_backflip", Go2_BackFlip_Robot, Go2_BackFlip_Cfg_Yu(), Go2_BackFlip_PPO_Yu())

# ===== DreamwaQ 任务 =====
task_registry.register( "go2_stairs_dreamwaq", Go2_Stairs_DreamWaQ_Robot, Go2_Stairs_DreamWaQ_Cfg_Yu(), Go2_Stairs_DreamWaQ_PPO_Yu())

# ===== AMP + DreamwaQ 任务 =====
task_registry.register( "go2_amp_dreamwaq", Go2_Stairs_AMP_DreamWaQ_Robot, Go2_Stairs_AMP_DreamWaQ_Cfg_Yu(), Go2_Stairs_AMP_DreamWaQ_PPO_Yu())

# ===== CTS 任务 =====
task_registry.register( "go2_cts", Go2_Cts_Robot, Go2_Cts_Cfg_Yu(), Go2_Cts_PPO_Yu())

# ===== AMP + CTS 任务 =====
task_registry.register( "go2_amp_cts", Go2_AMP_Cts_Robot, Go2_AMP_Cts_Cfg_Yu(), Go2_AMP_Cts_PPO_Yu())

# ===== AMP Teacher-Student 任务（teacher: AMP 特权策略训练；student: LSTM 蒸馏部署策略）=====
task_registry.register( "go2_amp_ts", LeggedRobotAMP_TS, Go2_AMP_Ts_Cfg_Yu(), Go2_AMP_Ts_PPO_Yu())
task_registry.register( "go2_amp_ts_student", LeggedRobotAMP_TS, Go2_AMP_Ts_Student_Cfg_Yu(), Go2_AMP_Ts_Student_PPO_Yu())

print("注册的任务:  ",task_registry.task_classes)
