from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.GO2_cannot_deploy.go2_config import GO2Cfg_Yu,GO2CfgPPO_Yu
from legged_gym.envs.GO2_Stand.GO2_Handstand.Go2_handstand_Config import GO2Cfg_Handstand,GO2CfgPPO_Handstand


from .base.legged_robot import LeggedRobot
from .GO2_cannot_deploy.Go2_env import Go2_env
from .GO2_Stand.GO2_Handstand.Go2_handstand import Go2_stand
from legged_gym.utils.task_registry import task_registry




task_registry.register( "go2", Go2_env, GO2Cfg_Yu(), GO2CfgPPO_Yu())
task_registry.register( "go2_handstand", Go2_stand, GO2Cfg_Handstand(), GO2CfgPPO_Handstand())
print("注册的任务:  ",task_registry.task_classes)