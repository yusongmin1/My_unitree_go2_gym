import torch
import torchvision
import torch.nn.functional as F
from typing import Optional, List, Union, Callable
from .config.sensor_config import Sensor, Config, Camera, Lidar, ImageProcess, LidarProcess

import warp as wp
import numpy as np

def torch_normalize(v):
    norm = torch.norm(v, dim=-1, keepdim=True)
    return v / norm

def quaternion_conjugate(q):
    # x, y, z, w
    return torch.stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]], dim=-1)

def quaternion_inverse(q):
    return torch_normalize(quaternion_conjugate(q))

def quaternion_apply(q, v):
    # q: x, y, z, w
    # v: x, y, z
    q = torch_normalize(q)
    v = torch.cat([v, torch.zeros_like(v[:, 0:1])], dim=-1)
    return quaternion_multiply(q, quaternion_multiply(v, quaternion_conjugate(q)))[:, :3]

def euler_to_quaternion(rot_angles, quat_format='xyzw'):
    """
    rot_angles: (3,) or (num_env, 3) Euler angles in radians
    quat_format: 'xyzw' or 'wxyz'
    """
    if len(rot_angles.shape) == 1:
        rot_angles = rot_angles.expand(1, -1)
    
    roll, pitch, yaw = torch.deg2rad(rot_angles[:, 0]), torch.deg2rad(rot_angles[:, 1]), torch.deg2rad(rot_angles[:, 2])
    
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    if quat_format == 'xyzw':
        x = cy * cp * sr - sy * sp * cr
        y = cy * sp * cr + sy * cp * sr
        z = sy * cp * cr - cy * sp * sr
        w = cy * cp * cr + sy * sp * sr
    else:  # wxyz
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = cy * sp * cr + sy * cp * sr
        z = sy * cp * cr - cy * sp * sr

    quaternion = torch.stack([x, y, z, w], dim=-1)
    
    return quaternion

@torch.jit.script
def quaternion_rotate_batch(q: torch.Tensor, v: torch.Tensor, quat_format: str = 'xyzw') -> torch.Tensor:
    """
    TorchScript optimized quaternion rotation function
    Args:
        q: (num_env, 4) quaternions
        v: (3,) or (num_env, 3) vectors
        quat_format: 'xyzw' or 'wxyz'
    Returns:
        rotated vectors (num_env, 3)
    """
    if len(v.shape) == 1:
        v = v.expand(q.shape[0], -1)

    qvec = q[:, 0:3] if quat_format == 'xyzw' else q[:, 1:]
    qw = q[:, 3:4] if quat_format == 'xyzw' else q[:, 0:1]
    
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    
    return v + 2.0 * (uv * qw + uuv)

@torch.jit.script
def quaternion_multiply_batch(q1: torch.Tensor, q2: torch.Tensor, quat_format: str = 'xyzw') -> torch.Tensor:
    """
    q1: (num_env, 4) quaternions
    q2: (4,) or (num_env, 4) quaternion
    quat_format: 'xyzw' or 'wxyz'
    """
    if len(q2.shape) == 1:
        q2 = q2.expand(q1.shape[0], -1)
    
    if quat_format == 'xyzw':
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([x, y, z, w], dim=1)
    else:  # wxyz
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=1)

def quaternion_multiply(q1, q0, quat_format='xyzw'):
    """
    Multiply two n * 4 quaternion tensors.
    """
    if quat_format == 'wxyz':
        x0, y0, z0, w0 = q0[:, 1], q0[:, 2], q0[:, 3], q0[:, 0]
        x1, y1, z1, w1 = q1[:, 1], q1[:, 2], q1[:, 3], q1[:, 0]
    else:
        x0, y0, z0, w0 = q0[:, 0], q0[:, 1], q0[:, 2], q0[:, 3]
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    return torch.stack([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dim=-1)

def make_mesh(self, vertices, faces):
        return wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3),
            indices=wp.array(faces.flatten(), dtype=wp.int32)
        )

def make_cube_mesh(cube_length=2, cube_position=(0, 0, 0)):

    half_length = cube_length / 2.0
    cube_vertices = np.array([
        [-half_length, -half_length, -half_length], [half_length, -half_length, -half_length], 
        [half_length, half_length, -half_length], [-half_length, half_length, -half_length],
        [-half_length, -half_length, half_length], [half_length, -half_length, half_length], 
        [half_length, half_length, half_length], [-half_length, half_length, half_length]
    ], dtype=np.float32)

    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3],  # front
        [4, 5, 6], [4, 6, 7],  # back
        [0, 1, 5], [0, 5, 4],  # bottom
        [2, 3, 7], [2, 7, 6],  # top
        [0, 3, 7], [0, 7, 4],  # left
        [1, 2, 6], [1, 6, 5]   # right
    ], dtype=np.int32)

    cube_vertices += np.array(cube_position, dtype=np.float32)

    cube_mesh = wp.Mesh(
        points=wp.array(cube_vertices, dtype=wp.vec3),
        indices=wp.array(cube_faces.flatten(), dtype=wp.int32)
    )

    return cube_mesh

def make_lidar_processor(cfg: Lidar, debug: bool = False) -> Callable:
    process_cfg = cfg.process
    noise_guassian = process_cfg.noise.gaussian 
    noise_dropout = process_cfg.noise.dropout
    @torch.jit.script
    def processor(data: torch.Tensor, noise_guassian: Optional[float], noise_dropout: Optional[float]) -> torch.Tensor:
        if noise_guassian is not None:
            data.add_(torch.randn_like(data) * noise_guassian)
        if noise_dropout is not None:
            mask = torch.rand_like(data) > noise_dropout
            data.mul_(mask)
        return data
    return lambda data: processor(data, noise_guassian, noise_dropout)
            
def make_image_processor(cfg: Camera, debug: bool = False) -> Callable:
    process_cfg = cfg.process
    max_range = cfg.pattern.max_range
    min_range = cfg.pattern.min_range

    if debug:
        print("\033[32mImage Processor Configuration:\033[0m")
        if hasattr(process_cfg, 'resize') and process_cfg.resize is not None:
            print(f"\t\033[36mResize operation, target size: {process_cfg.resize}\033[0m")
        
        if hasattr(process_cfg, 'noise') and process_cfg.noise is not None:
            noise_cfg = process_cfg.noise
            if hasattr(noise_cfg, 'gaussian') and noise_cfg.gaussian is not None:
                print(f"\t\033[33mGaussian noise, standard deviation: {noise_cfg.gaussian}\033[0m")
            if hasattr(noise_cfg, 'dropout') and noise_cfg.dropout is not None:
                print(f"\t\033[35mDropout, probability: {noise_cfg.dropout}\033[0m")
        
        if hasattr(process_cfg, 'clip') and process_cfg.clip is not None:
            min_val, max_val = process_cfg.clip
            if max_val == 'max_range':
                max_val = max_range
            print(f"\t\033[34mClip operation, min value: {min_val}, max value: {max_val}\033[0m")
        
        if hasattr(process_cfg, 'normalize') and process_cfg.normalize:
            print("\t\033[32mNormalize operation, values will be normalized to [0, 1]\033[0m")


    noise_gaussian = (process_cfg.noise.gaussian 
                     if hasattr(process_cfg, 'noise') and process_cfg.noise is not None 
                     and hasattr(process_cfg.noise, 'gaussian') else None)
    
    noise_dropout = (process_cfg.noise.dropout 
                    if hasattr(process_cfg, 'noise') and process_cfg.noise is not None 
                    and hasattr(process_cfg.noise, 'dropout') else None)
    
    if hasattr(process_cfg, 'clip') and process_cfg.clip is not None:
        clip_min, clip_max = process_cfg.clip
        if clip_max == 'max_range':
            clip_max = max_range
    else:
        clip_min, clip_max = None, None
    
    normalize = process_cfg.normalize if hasattr(process_cfg, 'normalize') else False
    resize_shape = (process_cfg.resize
                   if hasattr(process_cfg, 'resize') and process_cfg.resize is not None 
                   else None)

    @torch.jit.script
    def process_image(image: torch.Tensor,
                     noise_gaussian: Optional[float],
                     noise_dropout: Optional[float],
                     clip_min: Optional[float],
                     clip_max: Optional[float],
                     normalize: bool,
                     resize_shape: Optional[List[int]],
                     max_range: float,
                     min_range: float) -> torch.Tensor:


        if clip_min is not None and clip_max is not None:
            image.clamp_(clip_min, clip_max)


        # if normalize:
        #     range_size = max_range - min_range
        #     image.div_(range_size)
        # if normalize:
        #     range_size = max_range - min_range
        #     image = (image - min_range) / range_size

        if resize_shape is not None:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = F.interpolate(image,
                                size=resize_shape,
                                mode='bilinear',    # bilinear,  bicubic
                                align_corners=False)
            if image.size(0) == 1:
                image = image.squeeze(0)
        # if noise_gaussian is not None:
        #     image.add_(torch.randn_like(image) * (noise_gaussian * image))
        # #
        # if noise_dropout is not None:
        #     mask = torch.rand_like(image) > noise_dropout
        #     image.mul_(mask)

        # print(image)
        return image

    return lambda x: process_image(x, noise_gaussian, noise_dropout,
                                   clip_min, clip_max, normalize, 
                                   resize_shape, max_range, min_range)