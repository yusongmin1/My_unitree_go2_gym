from .warp_cam import WarpCam
from .warp_lidar import WarpLidar
from .warp_utils import *
from .config.sensor_config import Sensor, Config, Camera, Lidar
from dataclasses import dataclass
import warp as wp
import torch
import numpy as np
import os, sys

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
 
WARP_SENSOR_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
WARP_SENSOR_CFG_DIR = os.path.join(WARP_SENSOR_ROOT_DIR, 'config')

# for visualization
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

class SensorData:
    def __init__(self, sensor, device, cfg: Sensor):#TODO base sensor class
        
        self.sensor = sensor
        self.cfg = cfg
        self.device = device
        self.num_envs = cfg.num_envs
        self._init_buffers(cfg)
        self.data_processor = None
        sensor.set_image_tensors(self.tensor)
        sensor.set_pose_tensor(self.pos, self.quat)

        self.noise_gaussian = cfg.process.noise.gaussian if (cfg.process is not None and cfg.process.noise is not None) else None
        self.noise_dropout = cfg.process.noise.dropout if (cfg.process is not None and cfg.process.noise is not None) else None

    def _init_buffers(self, cfg):
        self.quat = torch.zeros((cfg.num_envs, cfg.num_sensors, 4), device=self.device, requires_grad=False, dtype=torch.float32)
        self.pos = torch.zeros((cfg.num_envs, cfg.num_sensors, 3), device=self.device, requires_grad=False, dtype=torch.float32)
        self.pixels = None
        self.clean_pix = None
        self.pro_pix = None
        self.tensor = None
        self._calcu_offset()
    
    def _post_update(self):
        pass

    def update(self):
        self.pixels[:] = self.sensor.capture(debug=False)
        if hasattr(self, "data_processor") and self.data_processor is not None:
            self.pro_pix  = self.data_processor(self.pixels) #TODO check
            # self.pro_pix = self.clean_pix.clone()

            # if self.noise_gaussian is not None:
            #     self.pro_pix.add_(torch.randn_like(self.pro_pix) * (self.noise_gaussian * self.pro_pix))
            #
            # if self.noise_dropout is not None:
            #     mask = torch.rand_like(self.pro_pix) > self.noise_dropout
            #     self.pro_pix.mul_(mask)

        self._post_update()
            
    def _calcu_offset(self):
        rand_p = self.cfg.offset.trans_rand
        if rand_p is not None: 
            print("random translation")
            self.offset_p = torch.tensor(self.cfg.offset.translation, dtype=torch.float32, device=self.device, requires_grad=False).repeat(self.num_envs, 1)
            trans_min = torch.tensor(rand_p.min, dtype=torch.float32, device=self.device, requires_grad=False)
            trans_max = torch.tensor(rand_p.max, dtype=torch.float32, device=self.device, requires_grad=False)
            self.offset_p.add_(torch.rand(self.num_envs, 3, device=self.device) * (trans_max - trans_min) + trans_min)
        else:
            self.offset_p = torch.tensor(self.cfg.offset.translation, dtype=torch.float32, device=self.device, requires_grad=False)
        
        rand_q = self.cfg.offset.rot_rand
        if rand_q is not None:
            print("random rotation")
            if len(self.cfg.offset.rotation) == 3:
                euler_base = torch.tensor(self.cfg.offset.rotation, dtype=torch.float32, device=self.device, requires_grad=False).repeat(self.num_envs, 1)
                euler_min = torch.tensor(rand_q.min, dtype=torch.float32, device=self.device, requires_grad=False)
                euler_max = torch.tensor(rand_q.max, dtype=torch.float32, device=self.device, requires_grad=False)
                euler_base.add_(torch.rand(self.num_envs, 3, device=self.device) * (euler_max - euler_min) + euler_min)
                self.offset_q = euler_to_quaternion(euler_base)
            else:
                raise ValueError("Invalid rotation format")
        else:
            if len(self.cfg.offset.rotation) == 3:
                self.offset_q = euler_to_quaternion(torch.tensor(self.cfg.offset.rotation, dtype=torch.float32, device=self.device, requires_grad=False))
            elif len(self.cfg.offset.rotation) == 4:
                self.offset_q = torch.tensor(self.cfg.offset.rotation, dtype=torch.float32, device=self.device, requires_grad=False)
            else:
                raise ValueError("Invalid rotation format")
    def _visualize(self):
        raise NotImplementedError
    @property
    def data(self):
        return self.pro_pix
    @property
    def clean_data(self):
        return self.clean_pix
    
class CameraData(SensorData):
    def __init__(self, sensor:WarpCam, device, cfg: Camera):
        super().__init__(sensor, device, cfg)
        self.data_processor = make_image_processor(self.cfg, debug=False) if cfg.process is not None else None
        
    def _init_buffers(self, cfg):
        super()._init_buffers(cfg)
        self.pixels = torch.zeros((cfg.num_envs, cfg.num_sensors, cfg.pattern.height, cfg.pattern.width), device=self.device, requires_grad=False)
        self.tensor = torch.zeros((cfg.num_envs, cfg.num_sensors, cfg.pattern.height, cfg.pattern.width), device=self.device, requires_grad=False)
        
    def _calcu_offset(self):
        super()._calcu_offset()
        self.offset_q = quaternion_multiply_batch(self.offset_q, torch.tensor([-0.5, 0.5, -0.5, 0.5], dtype=torch.float32, device=self.device, requires_grad=False))
    
class LidarData(SensorData):
    def __init__(self, sensor:WarpLidar, device, cfg: Lidar, manager):
        super().__init__(sensor, device, cfg)
        self.manager = manager
        process_cfg = self.cfg.process
        if process_cfg is not None:
            self.data_processor = make_lidar_processor(self.cfg, debug=False)
            self.random_downsample = torch.tensor(process_cfg.random_downsample, dtype=torch.int32) if process_cfg.random_downsample is not None else None # [[0, 0], [2, 2]] means min and max downsample
        else:
            self.data_processor = None
            self.random_downsample = None
    
    def _init_buffers(self, cfg):
        super()._init_buffers(cfg)
        self.tensor = torch.zeros((cfg.num_envs, cfg.num_sensors, cfg.pattern.height, cfg.pattern.width, 3), device=self.device, requires_grad=False)
        self.pixels = torch.zeros((cfg.num_envs, cfg.num_sensors, cfg.pattern.height, cfg.pattern.width, 3), device=self.device, requires_grad=False)
        self.pcl_b = None
        self.occupancy_points = None
        self.wide_occupancy_points = None
        if getattr(self.cfg.pattern, "calcu_occupancy", False):
            resolution = self.cfg.pattern.occu_resolution
            self.occupancy_range = [(-0.75, 1.25, resolution), (-0.45, 0.55, resolution), (-1.8, 1.2, 0.03)]
            shape = [int((rg[1]-rg[0])/rg[2]) for rg in self.occupancy_range]
            self.occupancy_grid = torch.zeros((self.num_envs, *shape), dtype=torch.bool, device=self.device)
            
        if getattr(self.cfg.pattern, "calcu_wide_occupancy", False):
            resolution = self.cfg.pattern.wide_occu_resolution
            self.wide_occupancy_range = [(-0.95, 3.05, resolution), (-1.95, 2.05, resolution), (-1.8, 1.2, 0.03)]
            shape = [int((rg[1]-rg[0])/rg[2]) for rg in self.wide_occupancy_range]
            self.wide_occupancy_grid = torch.zeros((self.num_envs, *shape), dtype=torch.bool, device=self.device)
            
    def _post_update(self):
        # if self.random_downsample is not None:
        #     downsample = torch.randint(self.random_downsample[0], self.random_downsample[1]+1, (2,), device=self.device)
        #     pcl = self.pixels[..., ::downsample[0], ::downsample[1],:]
        # else:
        #     pcl = self.pixels
        self.pcl_b = self.pixels.reshape(self.num_envs, -1, 3).detach() # (n*m)*3
        # pcl = pcl.reshape(self.cfg.num_envs, -1, 3).detach() #world frame #(n*m)*3
        # root_ori = self.manager.base_ori.clone()
        # root_ori[:, :2] = 0.0
        # root_ori = torch_normalize(root_ori).unsqueeze(1).repeat(1, pcl.shape[1], 1).reshape(-1, 4)
        # root_pos = self.manager.base_pos.unsqueeze(1).repeat(1, pcl.shape[1], 1).reshape(-1, 3)
            
        # ## calculate root frame pointcloud
        # if self.cfg.pattern.pointcloud_in_world_frame:
        #     self.pcl_b = quaternion_apply(quaternion_inverse(root_ori), (pcl.reshape(-1, 3) - root_pos)) # root frame
        # else: # in sensor frame
        #     # raise NotImplementedError("not implemented only apply yaw rotation")
        #     if len(self.offset_q.shape) == 2:
        #         sensor_ori = self.offset_q.unsqueeze(1).repeat(1, pcl.shape[1], 1).reshape(-1, 4)
        #     else:
        #         sensor_ori = self.offset_q.repeat(self.num_envs, pcl.shape[1], 1).reshape(-1, 4)
        #     if len(self.offset_p.shape) == 2:
        #         sensor_pos = self.offset_p.unsqueeze(1).repeat(1, pcl.shape[1], 1).reshape(-1, 3)
        #     else:
        #         sensor_pos = self.offset_p.repeat(self.num_envs, pcl.shape[1], 1).reshape(-1, 3)
        #     self.pcl_b = quaternion_apply(sensor_ori, pcl.reshape(-1, 3)) + sensor_pos # root frame
        # self.pcl_b = self.pcl_b.reshape(self.num_envs, -1, 3)
        
        ## calculate occupancy
        if getattr(self.cfg.pattern, "calcu_occupancy", False):
            self.occupancy_points = self._calcu_occupancy(self.pcl_b.clone(), self.occupancy_range, self.occupancy_grid)
            
        if getattr(self.cfg.pattern, "calcu_wide_occupancy", False):
            self.wide_occupancy_points = self._calcu_occupancy(self.pcl_b.clone(), self.wide_occupancy_range, self.wide_occupancy_grid)
            
    def _calcu_occupancy(self, p, ranges, grid):
        x_range, y_range, z_range = ranges # [(min, max, step), ...]
        valid_mask = (p[:, :, 0] >= x_range[0]) & (p[:, :, 0] < x_range[1]) & (p[:, :, 1] >= y_range[0]) & (p[:, :, 1] < y_range[1]) & (abs(p[:, :, 2])<5.0) #& (p[:, :, 2] >= z_range[0]) & (p[:, :, 2] < z_range[1]) #n*k
        p = p[valid_mask] # points*3
        # get every point's env id
        env_ids = valid_mask.nonzero(as_tuple=False)[:, 0] 
        grid_indices = [((p[:, 0] - x_range[0]) / x_range[2]).clamp(0, (x_range[1] - x_range[0]) / x_range[2]-0.01).long(),
                        ((p[:, 1] - y_range[0]) / y_range[2]).clamp(0, (y_range[1] - y_range[0]) / y_range[2]-0.01).long(),
                        ((p[:, 2] - z_range[0]) / z_range[2]).clamp(0, (z_range[1] - z_range[0]) / z_range[2]-0.01).long()]
        grid[:] = 0.0
        grid[env_ids, grid_indices[0], grid_indices[1], grid_indices[2]] = 1.0 # env_id, x, y, z
    
        ## for visualization
        # visualize occupancy grid local
        p = grid[0,...].nonzero().float()[:, :] # point num * 3 only env 0
        # p = self.occupancy_grid.nonzero().float()[:, 1:] # all env
        p = p * torch.tensor([x_range[2], y_range[2], z_range[2]], device=self.device)
        p = p + torch.tensor([x_range[0], y_range[0], z_range[0]], device=self.device)
        
        return p
     
    @property
    def data(self):
        return self.pcl_b
    
    @property
    def occupancy(self):
        return self.occupancy_grid
    
    @property
    def wide_occupancy(self):
        return self.wide_occupancy_grid

class WarpManager:
    def __init__(self, num_envs, gym_ptr, cfg=Config(), device: str="cuda:0", enable_viz=True):
        print("\033[1;32;40m ----- Initializing WarpManager -----")
        
        self.cfg = cfg
        self.gym_ptr = gym_ptr
        self.num_envs = num_envs
        self.device = device 
        self.sensors = {}
        self.visualizer = None
        
        setattr(self.cfg, 'num_env', num_envs)
        self.cfg.print_config()
        
        self.__initialize() # prepare sensor data

        if enable_viz:
            try:
                from .viz import WarpViser
                self.visualizer = WarpViser(self)
                print(" ----- Visualizer initialized -----")
            except ImportError as e:
                print(f" ----- Warning: Could not initialize visualizer: {e} -----")
        
        print(" ----- WarpManager Initialized Successed ----- \033[0m")
        
    def warp_update_frame(self):
        for name, sensor in self.sensors.items():
            self.__update_position(sensor)
            sensor.update()
        if self.visualizer is not None:
            self.visualizer.update_scene()
    
    def __initialize(self):
        wp.init()
        self.__load_mesh_from_gym(self.gym_ptr.terrain)
        
        for k in dir(self.cfg):
            if k.startswith("_"):
                continue
            v = getattr(self.cfg, k)
            if not isinstance(v, (Camera, Lidar)):
                continue
            v.num_envs = self.num_envs
            if type(v) == Camera:
                print(f" ----- Creating Camera {k} -----")
                self.sensors[k] = self.__create_camera(v)
            elif type(v) == Lidar:
                print(f" ----- Creating Lidar {k} -----")
                self.sensors[k] = self.__create_lidar(v)
                
        if len(self.sensors) == 0:
            raise ValueError("No sensor configured, but WarpManager is used")
        
    def __load_mesh_from_gym(self, gym_terrain):
        self.warp_mesh = wp.Mesh(
            points=wp.array(gym_terrain.vertices.flatten(), dtype=wp.vec3),
            indices=wp.array(gym_terrain.triangles.flatten(), dtype=wp.int32)
        )
        self.warp_mesh_id = wp.array([self.warp_mesh.id], dtype=wp.uint64)

    def __update_position(self, sensor:SensorData):      
        sensor_p = sensor.offset_p
        sensor_r = sensor.offset_q
        # 跨设备保护：sim 管线在 CPU 而 warp 在 GPU 时先搬运（同设备则无拷贝）
        base_ori = self.base_ori.to(sensor.pos.device)
        base_pos = self.base_pos.to(sensor.pos.device)
        sensor.pos[:] = (quaternion_rotate_batch(base_ori, sensor_p) + base_pos).unsqueeze(1)
        sensor.quat[:] = (quaternion_multiply_batch(base_ori, sensor_r)).unsqueeze(1)

    def __create_camera(self, cfg:Camera, mesh_id=None):
        if mesh_id is None:
            mesh_id = self.warp_mesh_id
        cam = WarpCam(self.num_envs, cfg.num_sensors, cfg.pattern, mesh_id, self.device)
        return CameraData(cam, self.device, cfg)

    def __create_lidar(self, cfg:Lidar, mesh_id=None):
        if mesh_id is None:
            mesh_id = self.warp_mesh_id
        lidar = WarpLidar(self.num_envs, cfg.num_sensors, cfg.pattern, mesh_id, self.device)            
        return LidarData(lidar, self.device, cfg, manager=self)
    
    @property
    def base_pos(self):
        return self.gym_ptr.root_states[:,:3].detach().to(torch.float32)
    
    @property
    def base_ori(self):
        return self.gym_ptr.root_states[:,3:7].detach().to(torch.float32)
    
    def __getitem__(self, key):
        return self.sensors[key]
    
    def close(self):
        if self.visualizer is not None:
            self.visualizer.close()
            self.visualizer = None
    
    def __del__(self):
        self.close()