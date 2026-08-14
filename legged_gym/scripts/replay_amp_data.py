import cv2
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

from isaacgym import gymapi, gymtorch
from legged_gym.envs import *
from legged_gym.utils import get_args

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    for k in env_cfg.control.stiffness.keys():
        env_cfg.control.stiffness[k] = 0.0
    for k in env_cfg.control.damping.keys():
        env_cfg.control.damping[k] = 0.0
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.mesh_type= 'plane'
    # env_cfg.noise.add_noise = False
    # env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.push_robots = False
    # env_cfg.domain_rand.randomize_gains = False
    # env_cfg.domain_rand.randomize_base_mass = False
    train_cfg.runner.amp_num_preload_transitions = 1

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()


    camera_rot = 0
    camera_rot_per_sec = np.pi / 6
    img_idx = 0

    video_duration = 4
    num_frames = int(video_duration / env.dt)
    print(f"gathering {num_frames} frames")
    video = None

    t = 0.0
    traj_idx = 0

    while traj_idx < len(env.amp_loader.trajectory_lens):
        actions = torch.zeros(
            (env_cfg.env.num_envs, env.num_actions), device=env.sim_device
        )

        if (
            t + env.amp_loader.time_between_frames + env.dt
        ) >= env.amp_loader.trajectory_lens[traj_idx]:
            traj_idx += 1
            t = 0
        else:
            t += env.dt
        print(f"traj_idx: {traj_idx}, t: {t}",env.amp_loader.trajectory_lens)
        env_ids = torch.tensor([0], device=env.device)
        root_pos = env.amp_loader.get_root_pos_batch(
            env.amp_loader.get_full_frame_at_time_batch(
                np.array([traj_idx]), np.array([t])
            )
        )
        env.root_states[env_ids, :3] = root_pos
        root_orn = env.amp_loader.get_root_rot_batch(
            env.amp_loader.get_full_frame_at_time_batch(
                np.array([traj_idx]), np.array([t])
            )
        )
        env.root_states[env_ids, 3:7] = root_orn
        env.root_states[env_ids, 7:10] = quat_rotate(
            root_orn,
            env.amp_loader.get_linear_vel_batch(
                env.amp_loader.get_full_frame_at_time_batch(
                    np.array([traj_idx]), np.array([t])
                )
            ),
        )
        env.root_states[env_ids, 10:13] = quat_rotate(
            root_orn,
            env.amp_loader.get_angular_vel_batch(
                env.amp_loader.get_full_frame_at_time_batch(
                    np.array([traj_idx]), np.array([t])
                )
            ),
        )

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        env.dof_pos[env_ids] = env.amp_loader.get_joint_pose_batch(
            env.amp_loader.get_full_frame_at_time_batch(
                np.array([traj_idx]), np.array([t])
            )
        )
        env.dof_vel[env_ids] = env.amp_loader.get_joint_vel_batch(
            env.amp_loader.get_full_frame_at_time_batch(
                np.array([traj_idx]), np.array([t])
            )
        )
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        env.gym.set_dof_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
        # print("---")
        foot_pos_amp = env.amp_loader.get_tar_toe_pos_local_batch(
            env.amp_loader.get_full_frame_at_time_batch(
                np.array([traj_idx]), np.array([t])
            )
        )
        # print(env.get_amp_observations()[0, 12:24])
        # print(foot_pos_amp[0])

        env.step(actions.detach())

        # Reset camera position.
        look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
        camera_rot = (camera_rot + camera_rot_per_sec * env.dt) % (2 * np.pi)
        camera_relative_position = 2.0 * np.array(
            [np.cos(camera_rot), np.sin(camera_rot), 0.45]
        )
        env.set_camera(look_at + camera_relative_position, look_at)

        if RECORD_FRAMES:
            frames_path = os.path.join(
                LEGGED_GYM_ROOT_DIR,
                "logs",
                train_cfg.runner.experiment_name,
                "exported",
                "frames",
            )
            if not os.path.isdir(frames_path):
                os.mkdir(frames_path)
            filename = os.path.join(
                "logs",
                train_cfg.runner.experiment_name,
                "exported",
                "frames",
                f"{img_idx}.png",
            )
            env.gym.write_viewer_image_to_file(env.viewer, filename)
            img = cv2.imread(filename)
            if video is None:
                video = cv2.VideoWriter(
                    "record.mp4",
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    int(1 / env.dt),
                    (img.shape[1], img.shape[0]),
                )
            video.write(img)
            img_idx += 1

    video.release()


if __name__ == "__main__":
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    args = get_args()
    play(args)