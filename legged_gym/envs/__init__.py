from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.Go2_MoB.GO2_JUMP.go2_jump_env import GO2_JUMP_Robot
from legged_gym.envs.Go2_MoB.GO2_JUMP.GO2_JUMP_config import GO2_JUMP_Cfg_Yu,GO2_JUMP_PPO_Yu


from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Trot import GO2_Trot_Robot
from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Trot_config import GO2_Trot_Cfg_Yu,GO2_Trot_PPO_Yu

from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Stairs import GO2_Stairs_Robot
from legged_gym.envs.Go2_MoB.GO2_Trot.GO2_Stairs_config import GO2_Stairs_Cfg_Yu,GO2_Stairs_PPO_Yu

from legged_gym.envs.GO2_Flip.GO2_BackFlip.GO2_BackFlip_env import Go2_BackFlip
from legged_gym.envs.GO2_Flip.GO2_BackFlip.GO2_BackFlip_Config import GO2_BackFlip_Cfg_Yu, GO2_BackFlip_PPO_Yu

from legged_gym.envs.GO2_Flip.GO2_Spring_Jump.GO2_Spring_Jump_env import GO2_Spring_Jump_Robot
from legged_gym.envs.GO2_Flip.GO2_Spring_Jump.GO2_Spring_Jump_Config import GO2_Spring_Jump_Cfg_Yu, GO2_Spring_Jump_PPO_Yu

from legged_gym.envs.GO2_Stand.GO2_Handstand.Go2_handstand import Go2_stand
from legged_gym.envs.GO2_Stand.GO2_Handstand.Go2_handstand_Config import GO2Cfg_Handstand,GO2CfgPPO_Handstand

from legged_gym.utils.task_registry import task_registry


task_registry.register( "go2_trot", GO2_Trot_Robot, GO2_Trot_Cfg_Yu(), GO2_Trot_PPO_Yu())
task_registry.register( "go2_stairs", GO2_Stairs_Robot, GO2_Stairs_Cfg_Yu(), GO2_Stairs_PPO_Yu())
task_registry.register( "go2_jump", GO2_JUMP_Robot, GO2_JUMP_Cfg_Yu(), GO2_JUMP_PPO_Yu())
task_registry.register( "go2_handstand", Go2_stand, GO2Cfg_Handstand(), GO2CfgPPO_Handstand())

task_registry.register( "go2_spring_jump", GO2_Spring_Jump_Robot, GO2_Spring_Jump_Cfg_Yu(), GO2_Spring_Jump_PPO_Yu())
task_registry.register( "go2_backflip", Go2_BackFlip, GO2_BackFlip_Cfg_Yu(), GO2_BackFlip_PPO_Yu())

print("注册的任务:  ",task_registry.task_classes)
