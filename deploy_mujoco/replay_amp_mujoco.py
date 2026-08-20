"""在 MuJoCo 中回放 AMP 动捕数据（参考 sim2sim 脚本写法，不依赖 isaacgym）。

用法：
    python deploy_mujoco/replay_amp_mujoco.py                          # 回放 datasets/mocap_motions_go2 全部动作
    python deploy_mujoco/replay_amp_mujoco.py --motion datasets/mocap_motions_go2/forward.txt
    python deploy_mujoco/replay_amp_mujoco.py --speed 0.5             # 慢放
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import mujoco
import mujoco.viewer

from legged_gym import LEGGED_GYM_ROOT_DIR
from rsl_rl.datasets.motion_loader import AMPLoader


def quat_xyzw_to_wxyz(q):
    """动捕数据四元数为 isaac 的 [x,y,z,w]，MuJoCo qpos 需要 [w,x,y,z]。"""
    x, y, z, w = q
    return np.array([w, x, y, z])


def replay(args):
    # ---------- 加载动捕数据（CPU 上的 AMPLoader，脱离 isaacgym） ----------
    dt = args.dt  # 回放步长，与训练 env.dt 一致（插值步长）
    amp_loader = AMPLoader(
        device='cpu',
        time_between_frames=dt,
        motion_files=args.motion_files,
    )
    print(f"共 {len(amp_loader.trajectory_lens)} 条轨迹，总时长 "
          f"{sum(amp_loader.trajectory_lens):.1f}s，回放步长 {dt}s")

    # ---------- MuJoCo 模型 ----------
    model = mujoco.MjModel.from_xml_path(args.mjcf)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    num_joints = 12
    data.qpos[7:7 + num_joints] = amp_loader.get_joint_pose_batch(
        amp_loader.get_full_frame_at_time_batch(np.array([0]), np.array([0.0])))[0].numpy()
    mujoco.mj_forward(model, data)

    camera_rot_per_sec = 0.0  # 固定视角，不环绕旋转

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -30

        for traj_idx in range(len(amp_loader.trajectory_lens)):
            traj_len = amp_loader.trajectory_lens[traj_idx]
            print(f"[{traj_idx + 1}/{len(amp_loader.trajectory_lens)}] "
                  f"{amp_loader.trajectory_names[traj_idx]}: {traj_len:.2f}s")

            t = 0.0
            # 与原 isaacgym 版一致：t + 2*dt 触及轨迹末尾时切下一条，避免插值越界
            while t + 2 * dt < traj_len:
                step_start = time.time()
                frame = amp_loader.get_full_frame_at_time_batch(
                    np.array([traj_idx]), np.array([t]))[0]

                root_pos = amp_loader.get_root_pos_batch(frame.unsqueeze(0))[0].numpy()
                root_orn = amp_loader.get_root_rot_batch(frame.unsqueeze(0))[0].numpy()      # xyzw
                joint_pos = amp_loader.get_joint_pose_batch(frame.unsqueeze(0))[0].numpy()   # FL,FR,RL,RR
                joint_vel = amp_loader.get_joint_vel_batch(frame.unsqueeze(0))[0].numpy()
                lin_vel = amp_loader.get_linear_vel_batch(frame.unsqueeze(0))[0].numpy()     # 基座系
                ang_vel = amp_loader.get_angular_vel_batch(frame.unsqueeze(0))[0].numpy()    # 基座系

                # 写入 MuJoCo 状态（运动学回放：只设状态并 mj_forward，不跑动力学）
                data.qpos[0:3] = root_pos
                data.qpos[3:7] = quat_xyzw_to_wxyz(root_orn)
                data.qpos[7:7 + num_joints] = joint_pos
                # 线速度转世界系；MuJoCo free joint 角速度为基座系，直接用
                w, x, y, z = data.qpos[3:7]
                Rwb = np.array([
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ])
                data.qvel[0:3] = Rwb @ lin_vel
                data.qvel[3:6] = ang_vel
                data.qvel[6:6 + num_joints] = joint_vel
                data.ctrl[:] = 0

                mujoco.mj_forward(model, data)

                # 固定视角，仅 lookat 跟随机器人
                viewer.cam.lookat[:] = np.array([root_pos[0], root_pos[1], root_pos[2]])

                viewer.sync()

                # 实时同步
                time_until_next = dt / args.speed - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
                t += dt

    print("回放结束")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AMP 动捕数据 MuJoCo 回放')
    parser.add_argument('--motion', type=str, default=None,
                        help='动捕文件或目录（默认 datasets/mocap_motions_go2）')
    parser.add_argument('--mjcf', type=str,
                        default=f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/go2/scene.xml',
                        help='MuJoCo 模型路径')
    parser.add_argument('--dt', type=float, default=0.02, help='回放步长 [s]（与训练 env.dt 一致）')
    parser.add_argument('--speed', type=float, default=1.0, help='回放倍速')
    args = parser.parse_args()

    if args.motion is None:
        args.motion_files = sorted(glob.glob(os.path.join(
            LEGGED_GYM_ROOT_DIR, 'datasets/mocap_motions_go2', '*.txt')))
    elif os.path.isdir(args.motion):
        args.motion_files = sorted(glob.glob(os.path.join(args.motion, '*.txt')))
    else:
        args.motion_files = [args.motion]
    assert len(args.motion_files) > 0, '没有找到动捕文件'
    for f in args.motion_files:
        print('motion file:', f)

    replay(args)
