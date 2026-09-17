from dataclasses import field
from typing import Optional
from legged_gym.warp_sensor.helpers import config, MISSING

@config 
class TransRand:
    min: list = field(default_factory=lambda: [-0.0, -0.0, -0.0])
    max: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
@config 
class RotRand:
    min: list = field(default_factory=lambda: [-0.0, -2.0, -0.0])
    max: list = field(default_factory=lambda: [0.0, 2.0, 0.0])
@config # to initialize class members
class Offset:
    translation: list = field(default_factory=lambda: [0.27, 0.0, 0.07])
    rotation: list = field(default_factory=lambda: [0, 27, 0])
    trans_rand: TransRand = TransRand()
    rot_rand: RotRand = RotRand()

@config
class TransRandLidar:
    min: list = field(default_factory=lambda: [-0.6, -0.1, -0.1])
    max: list = field(default_factory=lambda: [0.1, 0.1, 0.1])
@config
class RotRandLidar:
    min: list = field(default_factory=lambda: [-5.0, -5.0, -5.0])
    max: list = field(default_factory=lambda: [5.0, 5.0, 5.0])
@config
class OffsetLidar(Offset):
    rotation: list = field(default_factory=lambda: [0, 0, 0])
    trans_rand = TransRandLidar()
    rot_rand = RotRandLidar()
@config 
class Noise:
    gaussian: float = 0.03
    dropout: float = 0.1
@config 
class ImageProcess:
    resize: Optional[list] = None
    noise: Optional[Noise] = None
    clip: Optional[list] = field(default_factory=lambda: [0.0, 10.0])
    normalize: bool = True
@config
class LidarProcess:
    noise: Optional[Noise] = Noise()
    random_downsample: Optional[list] = field(default_factory=lambda: [1, 3])

@config 
class Pattern: #only scan sensor
    max_range: int = 10.0
    min_range: int = 0.2
    return_pointcloud: bool = MISSING
    pointcloud_in_world_frame: bool = MISSING

@config 
class LidarPattern(Pattern):
    height: int = 100   # scan lines TODO change name
    width: int = 100    # points per scan line
    horizontal_fov_deg_min: float = -180.0
    horizontal_fov_deg_max: float = 180.0
    vertical_fov_deg_min: float = -89.0
    vertical_fov_deg_max: float = 89.0
    return_pointcloud: bool = True
    pointcloud_in_world_frame: bool = True
    
    calcu_occupancy: bool = False
    calcu_wide_occupancy: bool = False
    wide_occu_resolution: float = 0.1
    occu_resolution: float = 0.1

@config 
class DepthCameraPattern(Pattern):
    height: int = 120
    width: int = 160
    horizontal_fov_deg: float = 90.0
    max_range: float = 4.0
    min_range: float = 0.0
    calculate_depth: bool = True
    return_pointcloud: bool = False
    pointcloud_in_world_frame: bool = False

@config 
class Sensor:
    num_envs: int = 0
    num_sensors: int = 1
    offset: Offset = MISSING
    pattern: Pattern = MISSING

@config 
class Camera(Sensor):    
    offset: Offset = Offset()
    pattern: DepthCameraPattern = DepthCameraPattern()
    process: Optional[ImageProcess] = None

@config 
class Lidar(Sensor):
    offset: Offset = OffsetLidar()
    pattern: LidarPattern = LidarPattern()
    process: Optional[LidarProcess] = LidarProcess()
    
@config
class Config:
    camera: Optional[Camera] = Camera()
    lidar: Optional[Lidar] = Lidar()
