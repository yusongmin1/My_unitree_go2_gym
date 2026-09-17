# warp_sensor：从 MGDP-master 迁移的 Warp 深度相机传感器库
# 作为 legged_gym 的子包使用（无需 pip install，直接 import）
# 依赖：warp-lang（pip install warp-lang==1.10.1 已在 amp 环境装好）
from .warp_manager import WarpManager
from .config.sensor_config import *

import os

WARP_SENSOR_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
WARP_SENSOR_CFG_DIR = os.path.join(WARP_SENSOR_ROOT_DIR, 'config')
