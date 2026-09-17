import warp as wp
import pygame
import numpy as np
import torch
import math
import sys, os

# from warp_kernels.cam_kernel import DepthCameraWarpKernels
from warp_sensor.warp_kernels.cam_kernel import DepthCameraWarpKernels
# from warp_cam import WarpCam
from warp_sensor.warp_cam import WarpCam
from scipy.spatial.transform import Rotation
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from config.sensor_config import Sensor, Config, Lidar, Camera

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 3], q1[..., 0], q1[..., 1], q1[..., 2]
    w2, x2, y2, z2 = q2[..., 3], q2[..., 0], q2[..., 1], q2[..., 2]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((x, y, z, w), dim=-1)

def normalize_quaternion(q):
    return q / torch.norm(q, dim=-1, keepdim=True)

def test_cam():
    wp.init()
    
    # Create a larger cube mesh
    cube_vertices = np.array([
        [-2, -2, -2], [2, -2, -2], [2, 2, -2], [-2, 2, -2],
        [-2, -2, 2], [2, -2, 2], [2, 2, 2], [-2, 2, 2]
    ], dtype=np.float32)

    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3],  # front
        [4, 5, 6], [4, 6, 7],  # back
        [0, 1, 5], [0, 5, 4],  # bottom
        [2, 3, 7], [2, 7, 6],  # top
        [0, 3, 7], [0, 7, 4],  # left
        [1, 2, 6], [1, 6, 5]   # right
    ], dtype=np.int32)

    # Add ground mesh
    ground_size = 10
    ground_vertices = np.array([
        [-ground_size, -2, -ground_size],
        [ground_size, -2, -ground_size],
        [ground_size, -2, ground_size],
        [-ground_size, -2, ground_size]
    ], dtype=np.float32)

    ground_faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.int32)

    # Create meshes
    cube_mesh = wp.Mesh(
        points=wp.array(cube_vertices, dtype=wp.vec3),
        indices=wp.array(cube_faces.flatten(), dtype=wp.int32)
    )

    ground_mesh = wp.Mesh(
        points=wp.array(ground_vertices, dtype=wp.vec3),
        indices=wp.array(ground_faces.flatten(), dtype=wp.int32)
    )

    camera_config = Camera()
    config = camera_config.pattern

    # Create cameras
    num_envs = 1
    mesh_ids_array = wp.array([cube_mesh.id, ground_mesh.id], dtype=wp.uint64)
    device = 'cuda'

    # First-person camera
    cam = WarpCam(num_envs, camera_config.num_sensors, config, mesh_ids_array, device=device)
    camera_position = torch.tensor([[[0.0, 0.0, -5.0],[0.0, 0.0, -5.0]]], dtype=torch.float32, device=device)
    camera_orientation = torch.tensor([[[0.0, 0.0, 0.0, 1.0],[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    pixels = torch.zeros((num_envs, camera_config.num_sensors, config.height, config.width), dtype=torch.float32, device=device)
    cam.set_image_tensors(pixels)
    cam.set_pose_tensor(camera_position, camera_orientation)

    # Third-person camera
    third_person_cam = WarpCam(num_envs, camera_config.num_sensors, config, mesh_ids_array, device=device)
    third_person_pos = torch.tensor([[[0.0, 5.0, -10.0]]], dtype=torch.float32, device=device)
    third_person_orientation = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    third_person_pixels = torch.zeros_like(pixels)
    third_person_cam.set_image_tensors(third_person_pixels)
    third_person_cam.set_pose_tensor(third_person_pos, third_person_orientation)

    # Init Pygame
    pygame.init()
    screen = pygame.display.set_mode((config.width * 2, config.height))
    pygame.display.set_caption('Dual View Depth Renderer')
    clock = pygame.time.Clock()

    # View control parameters
    third_person_target = np.array([0.0, 0.0, 0.0])
    third_person_pitch = -0.3
    third_person_yaw = 0.0
    rotate_speed = 0.003
    move_speed = 0.1

    is_rotating_first = False
    is_rotating_third = False
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # left click
                    is_rotating_first = True
                elif event.button == 3:  # right click
                    is_rotating_third = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    is_rotating_first = False
                elif event.button == 3:
                    is_rotating_third = False

        # Keyboard control
        keys = pygame.key.get_pressed()
        # Get current orientation
        rot = Rotation.from_quat(camera_orientation[0, 0].cpu().numpy())
        forward = rot.apply([0, 0, 1])
        right = rot.apply([1, 0, 0])
        up = rot.apply([0, 1, 0])

        if keys[pygame.K_w]:
            camera_position[0, 0] += torch.tensor(forward * move_speed, device=device)
        if keys[pygame.K_s]:
            camera_position[0, 0] -= torch.tensor(forward * move_speed, device=device)
        if keys[pygame.K_a]:
            camera_position[0, 0] -= torch.tensor(right * move_speed, device=device)
        if keys[pygame.K_d]:
            camera_position[0, 0] += torch.tensor(right * move_speed, device=device)
        if keys[pygame.K_q]:
            camera_position[0, 0] += torch.tensor(up * move_speed, device=device)
        if keys[pygame.K_e]:
            camera_position[0, 0] -= torch.tensor(up * move_speed, device=device)

        # Mouse control
        if is_rotating_first:
            mouse_dx, mouse_dy = pygame.mouse.get_rel()
            if mouse_dx != 0 or mouse_dy != 0:
                yaw = -mouse_dx * rotate_speed
                pitch = -mouse_dy * rotate_speed

                quat_yaw = torch.tensor([0.0, np.sin(yaw/2), 0.0, np.cos(yaw/2)], dtype=torch.float32, device=device)
                quat_pitch = torch.tensor([np.sin(pitch/2), 0.0, 0.0, np.cos(pitch/2)], dtype=torch.float32, device=device)

                camera_orientation = quaternion_multiply(
                    camera_orientation.squeeze(0), 
                    quat_yaw.unsqueeze(0)
                )
                camera_orientation = quaternion_multiply(
                    camera_orientation, 
                    quat_pitch.unsqueeze(0)
                )
                camera_orientation = normalize_quaternion(camera_orientation).unsqueeze(0)

        elif is_rotating_third:
            mouse_dx, mouse_dy = pygame.mouse.get_rel()
            if mouse_dx != 0 or mouse_dy != 0:
                third_person_yaw -= mouse_dx * rotate_speed
                third_person_pitch = np.clip(
                    third_person_pitch - mouse_dy * rotate_speed,
                    -np.pi/2 + 0.1,
                    np.pi/2 - 0.1
                )
        else:
            pygame.mouse.get_rel()

        # Update camera pose
        cam.set_pose_tensor(camera_position, camera_orientation)

        # Update third-person view
        radius = 15.0
        x = radius * np.cos(third_person_pitch) * np.sin(third_person_yaw)
        y = radius * np.sin(third_person_pitch)
        z = radius * np.cos(third_person_pitch) * np.cos(third_person_yaw)
        
        third_person_pos_np = third_person_target + np.array([x, y, z])
        third_person_pos[0, 0] = torch.tensor(third_person_pos_np, dtype=torch.float32, device=device)

        # Compute third-person camera orientation
        direction = third_person_target - third_person_pos_np
        direction = direction / np.linalg.norm(direction)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(up, direction)
        right = right / np.linalg.norm(right)
        up = np.cross(direction, right)
        rot_matrix = np.column_stack((right, up, direction))
        quat = Rotation.from_matrix(rot_matrix).as_quat()
        third_person_orientation[0, 0] = torch.tensor(quat, dtype=torch.float32, device=device)

        third_person_cam.set_pose_tensor(third_person_pos, third_person_orientation)

        # Render
        depth_image = cam.capture()
        third_person_depth = third_person_cam.capture()

        # Process and display depth images
        def process_depth(depth):
            depth_np = depth[0, 0].cpu().numpy()
            depth_np[depth_np > config.max_range] = config.max_range
            depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
            return (depth_norm * 255).astype(np.uint8)

        depth_uint8 = process_depth(depth_image)
        third_depth_uint8 = process_depth(third_person_depth)

        depth_surface = pygame.surfarray.make_surface(np.flip(depth_uint8, axis=0))
        third_depth_surface = pygame.surfarray.make_surface(np.flip(third_depth_uint8, axis=0))

        screen.blit(depth_surface, (0, 0))
        screen.blit(third_depth_surface, (config.width, 0))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    wp.shutdown()

if __name__ == '__main__':
    test_cam()