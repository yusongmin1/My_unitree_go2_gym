import sys
import numpy as np
import torch
import warp as wp
import os
from warp_sensor.warp_lidar import WarpLidar
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from config.sensor_config import Sensor, Config, Lidar
from scipy.spatial.transform import Rotation as R

def quaternion_multiply(q1, q0):
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([
        w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
        w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
        w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
        w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0
    ])
    
# pyQT seems to conflict with plt 
# from warp_sensor.gl_vis import create_visualizer ,VisualizerWrapper
def test_lidar():
    wp.init()

    # Create test mesh models

    # Cube
    cube_vertices = np.array([
        [-2, -2, -2], [2, -2, -2], [1, 1, -1], [-1, 1, -1],
        [-2, -2, 2], [2, -2, 2], [1, 1, 1], [-1, 1, 1]
    ], dtype=np.float32)

    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [0, 3, 7], [0, 7, 4],  # left
        [1, 2, 6], [1, 6, 5]   # right
    ], dtype=np.int32)

    # Ground
    ground_size = 10
    ground_vertices = np.array([
        [-ground_size, -1, -ground_size],
        [ground_size, -1, -ground_size],
        [ground_size, -1, ground_size],
        [-ground_size, -1, ground_size]
    ], dtype=np.float32)

    ground_faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.int32)


    cube_mesh = wp.Mesh(
        points=wp.array(cube_vertices, dtype=wp.vec3),
        indices=wp.array(cube_faces.flatten(), dtype=wp.int32)
    )

    ground_mesh = wp.Mesh(
        points=wp.array(ground_vertices, dtype=wp.vec3),
        indices=wp.array(ground_faces.flatten(), dtype=wp.int32)
    )

    lidar_config = Lidar()
    config = lidar_config.pattern

    num_envs = 10
    mesh_ids_array = wp.array([cube_mesh.id, ground_mesh.id], dtype=wp.uint64)
    device = 'cuda:0'

    lidar = WarpLidar(num_envs, lidar_config.num_sensors, config, mesh_ids_array, device)
    lidar_position = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32, device=device)

    rot = R.from_euler('y', 0, degrees=True)
    q1 = rot.as_quat()  # [x, y, z, w]
    q2 = quaternion_multiply(q1, R.from_euler('y', 90, degrees=True).as_quat())
    q = quaternion_multiply(q2, R.from_euler('z', 90, degrees=True).as_quat())
    lidar_orientation = torch.tensor([[[q[0], q[1], q[2], q[3]]]], dtype=torch.float32, device=device)

    pixels = torch.zeros((num_envs, lidar_config.num_sensors, config.height, config.width, 3),
                         dtype=torch.float32, device=device)
    lidar.set_image_tensors(pixels)
    lidar.set_pose_tensor(lidar_position, lidar_orientation)

    pointcloud = lidar.capture()

    pointcloud_np = pointcloud.cpu().numpy().reshape(-1, 3)

    # Filter
    valid_points = pointcloud_np[np.any(pointcloud_np != 0, axis=1)]

    # Points within max distance
    # max_distance = 8.0
    lidar_pos_np = lidar_position.cpu().numpy().reshape(-1, 3)
    # distances = np.linalg.norm(valid_points - lidar_pos_np, axis=1)
    # valid_points = valid_points[distances <= max_distance]

    valid_points = valid_points.astype(np.float32)

    # Visualize
    vis = create_visualizer()
    vis.add_trimesh(cube_vertices, cube_faces)
    # vis.add_mesh(ground_mesh)
    vis.add_points(valid_points)
    vis.add_points(lidar_pos_np)
    # Create LiDAR FOV/range mesh
    fov_vertices = np.array([
        [0, 0, 0],  # origin
        [config.max_range * np.tan(np.radians(config.horizontal_fov_deg_min)), config.max_range * np.tan(np.radians(config.vertical_fov_deg_min)), config.max_range],
        [config.max_range * np.tan(np.radians(config.horizontal_fov_deg_max)), config.max_range * np.tan(np.radians(config.vertical_fov_deg_min)), config.max_range],
        [config.max_range * np.tan(np.radians(config.horizontal_fov_deg_max)), config.max_range * np.tan(np.radians(config.vertical_fov_deg_max)), config.max_range],
        [config.max_range * np.tan(np.radians(config.horizontal_fov_deg_min)), config.max_range * np.tan(np.radians(config.vertical_fov_deg_max)), config.max_range]
    ], dtype=np.float32)

    fov_faces = np.array([
        [0, 1, 2],  # bottom
        [0, 2, 3],  # bottom
        [0, 3, 4],  # bottom
        [0, 4, 1],  # bottom
        [1, 2, 3],  # top
        [1, 3, 4]   # top
    ], dtype=np.int32)

    fov_vertices_homogeneous = np.hstack((fov_vertices, np.ones((fov_vertices.shape[0], 1))))
    lidar_rot_matrix = rot.as_matrix()
    lidar_transform = np.eye(4)
    lidar_transform[:3, :3] = lidar_rot_matrix
    lidar_transform[:3, 3] = lidar_position.cpu().numpy().reshape(-1)

    fov_vertices_transformed = (lidar_transform @ fov_vertices_homogeneous.T).T[:, :3]

    vis.add_trimesh(fov_vertices_transformed, fov_faces)
    VisualizerWrapper.run()
    wp.shutdown()

def create_axis(size=5):
    from pyqtgraph import Vector
    from pyqtgraph.opengl import GLLinePlotItem
    lines = np.array([
        [0, 0, 0], [size, 0, 0],  # X axis
        [0, 0, 0], [0, size, 0],  # Y axis
        [0, 0, 0], [0, 0, size],  # Z axis
    ], dtype=np.float32)

    colors = np.array([
        [1, 0, 0, 1], [1, 0, 0, 1],  # X axis (red)
        [0, 1, 0, 1], [0, 1, 0, 1],  # Y axis (green)
        [0, 0, 1, 1], [0, 0, 1, 1],  # Z axis (blue)
    ], dtype=np.float32)

    axis = GLLinePlotItem(pos=lines, color=colors, width=2, antialias=True, mode='lines')
    return axis

if __name__ == '__main__':
    test_lidar()