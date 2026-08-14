from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.Go2_MoB.GO2_JUMP.go2_jump_env import GO2_JUMP_Robot
from legged_gym.envs.Go2_MoB.GO2_JUMP.GO2_JUMP_config import GO2_JUMP_Cfg_Yu,GO2_JUMP_PPO_Yu


from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Trot import GO2_Trot_Robot
from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Trot_config import GO2_Trot_Cfg_Yu,GO2_Trot_PPO_Yu

from legged_gym.envs.GO2_Flip.GO2_BackFlip.GO2_BackFlip_env import Go2_BackFlip
from legged_gym.envs.GO2_Flip.GO2_BackFlip.GO2_BackFlip_Config import GO2_BackFlip_Cfg_Yu, GO2_BackFlip_PPO_Yu

from legged_gym.envs.GO2_Flip.GO2_Spring_Jump.GO2_Spring_Jump_env import GO2_Spring_Jump_Robot
from legged_gym.envs.GO2_Flip.GO2_Spring_Jump.GO2_Spring_Jump_Config import GO2_Spring_Jump_Cfg_Yu, GO2_Spring_Jump_PPO_Yu

from legged_gym.envs.GO2_Stand.GO2_Handstand.Go2_handstand import Go2_stand
from legged_gym.envs.GO2_Stand.GO2_Handstand.Go2_handstand_Config import GO2Cfg_Handstand,GO2CfgPPO_Handstand

from legged_gym.envs.GO2_Stand.GO2_Leggedstand.Go2_legstand import Go2_legstand
from legged_gym.envs.GO2_Stand.GO2_Leggedstand.Go2_legstand_Config import GO2Cfg_Leggedstand,GO2CfgPPO_Leggedstand

# ========== DreamwaQ 任务（纯 DreamwaQ 算法）==========
from legged_gym.envs.GO2_DreamWaQ.GO2_Stairs_DreamWaQ import GO2_Stairs_DreamWaQ_Robot
from legged_gym.envs.GO2_DreamWaQ.GO2_Stairs_DreamWaQ_config import GO2_Stairs_DreamWaQ_Cfg_Yu, GO2_Stairs_DreamWaQ_PPO_Yu

# ========== AMP + DreamwaQ 任务（AMP 判别器 + DreamwaQ 算法）==========
from legged_gym.envs.GO2_AMP_DreamWaQ.GO2_Stairs_AMP_DreamWaQ import GO2_Stairs_AMP_DreamWaQ_Robot
from legged_gym.envs.GO2_AMP_DreamWaQ.GO2_Stairs_AMP_DreamWaQ_config import GO2_Stairs_AMP_DreamWaQ_Cfg_Yu, GO2_Stairs_AMP_DreamWaQ_PPO_Yu

# ========== CTS 任务（Concurrent Teacher-Student 算法）==========
from legged_gym.envs.GO2_CTS.GO2_Cts import GO2_Cts_Robot
from legged_gym.envs.GO2_CTS.GO2_Cts_config import GO2_Cts_Cfg_Yu, GO2_Cts_PPO_Yu

# ========== AMP + CTS 任务（AMP 判别器 + CTS 算法）==========
from legged_gym.envs.GO2_AMP_CTS.GO2_Amp_Cts import GO2_Amp_Cts_Robot
from legged_gym.envs.GO2_AMP_CTS.GO2_Amp_Cts_config import GO2_Amp_Cts_Cfg_Yu, GO2_Amp_Cts_PPO_Yu

# ========== AMP Teacher-Student 任务（AMP 训练 teacher + LSTM 蒸馏 student）==========
from legged_gym.envs.base.legged_robot_amp_ts import LeggedRobotAMP_TS
from legged_gym.envs.GO2_AMP_TS.go2_amp_ts_config import GO2_AMP_TS_Cfg, GO2_AMP_TS_CfgPPO
from legged_gym.envs.GO2_AMP_TS.go2_amp_ts_student_config import GO2_AMP_TS_Student_Cfg, GO2_AMP_TS_Student_CfgPPO

from legged_gym.utils.task_registry import task_registry


# ===== 标准 PPO 任务（main 原有特技任务）=====
task_registry.register( "go2_trot", GO2_Trot_Robot, GO2_Trot_Cfg_Yu(), GO2_Trot_PPO_Yu())
task_registry.register( "go2_jump", GO2_JUMP_Robot, GO2_JUMP_Cfg_Yu(), GO2_JUMP_PPO_Yu())
task_registry.register( "go2_handstand", Go2_stand, GO2Cfg_Handstand(), GO2CfgPPO_Handstand())
task_registry.register( "go2_leggedstand", Go2_legstand, GO2Cfg_Leggedstand(), GO2CfgPPO_Leggedstand())
task_registry.register( "go2_spring_jump", GO2_Spring_Jump_Robot, GO2_Spring_Jump_Cfg_Yu(), GO2_Spring_Jump_PPO_Yu())
task_registry.register( "go2_backflip", Go2_BackFlip, GO2_BackFlip_Cfg_Yu(), GO2_BackFlip_PPO_Yu())

# ===== DreamwaQ 任务 =====
task_registry.register( "go2_stairs_dreamwaq", GO2_Stairs_DreamWaQ_Robot, GO2_Stairs_DreamWaQ_Cfg_Yu(), GO2_Stairs_DreamWaQ_PPO_Yu())

# ===== AMP + DreamwaQ 任务 =====
task_registry.register( "go2_amp_dreamwaq", GO2_Stairs_AMP_DreamWaQ_Robot, GO2_Stairs_AMP_DreamWaQ_Cfg_Yu(), GO2_Stairs_AMP_DreamWaQ_PPO_Yu())

# ===== CTS 任务 =====
task_registry.register( "go2_cts", GO2_Cts_Robot, GO2_Cts_Cfg_Yu(), GO2_Cts_PPO_Yu())

# ===== AMP + CTS 任务 =====
task_registry.register( "go2_amp_cts", GO2_Amp_Cts_Robot, GO2_Amp_Cts_Cfg_Yu(), GO2_Amp_Cts_PPO_Yu())

# ===== AMP Teacher-Student 任务（teacher: AMP 特权策略训练；student: LSTM 蒸馏部署策略）=====
task_registry.register( "go2_amp_ts", LeggedRobotAMP_TS, GO2_AMP_TS_Cfg(), GO2_AMP_TS_CfgPPO())
task_registry.register( "go2_amp_ts_student", LeggedRobotAMP_TS, GO2_AMP_TS_Student_Cfg(), GO2_AMP_TS_Student_CfgPPO())

print("注册的任务:  ",task_registry.task_classes)
