from sim2sim_config_loader import load_cfg
Go2_AMP_Cts_Cfg_Yu = load_cfg(
    "legged_gym/envs/Go2_AMP_Cts/Go2_AMP_Cts_Config.py",
    "Go2_AMP_Cts_Cfg_Yu",
)
import math
import numpy as np
import mujoco, mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import time
from viewer_utils import VelocityArrowViewer

x_vel_max, y_vel_max, yaw_vel_max = 1.5, 1.0, 1.5
    
def get_obs(data,model):
    '''Extracts an observation from the mujoco data structure
    '''
    # print(data.qpos.astype(np.double).shape,data.qvel.astype(np.double).shape)
    q = data.qpos[7:19].astype(np.double)
    dq = data.qvel[6:].astype(np.double)
    quat = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]]
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.qvel[3:6].astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    base_pos = data.qpos[0:3].astype(np.double)
    foot_positions = []
    # foot_forces = data.cfrc_ext[0][2].copy().astype(np.double)
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        # print(body_name)
        if 'foot' in body_name: 
            # print(body_name)
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces = data.cfrc_ext[i][2].copy().astype(np.double)
    return (q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces)

def pd_control(target_q, q, kp, target_dq, dq, kd, cfg):
    '''Calculates torques from position commands
    '''
    torque_out = (target_q + cfg.robot_config.default_dof_pos - q ) * kp + (target_dq - dq)* kd
    return torque_out


def run_mujoco(policy, cfg):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.

    Returns:
        None
    """
    # 初始化手柄/键盘输入 + 箭头绘制工具（手柄优先，无手柄回退键盘）
    # 楼梯任务四档速度：0.5 / 1.0 / 1.5 / 2.0 m/s（L2 降档 / R2 升档）
    vel_arrow = VelocityArrowViewer(max_cmd=(x_vel_max, y_vel_max, yaw_vel_max),
                                    gears=[0.5, 1.0, 1.5, 2.0])
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    
    model.opt.timestep = cfg.sim_config.dt
    
    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions  # This should match the number of actuated joints in your model
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos

    mujoco.mj_step(model, data)
    
    # 使用官方 mujoco.viewer（支持 user_scn 绘制箭头）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -45
        viewer.cam.lookat[:] =np.array([0.0,-0.25,0.824])

        target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
   
        action = np.zeros((cfg.env.num_actions), dtype=np.double)

        hist_obs = deque()
        for _ in range(6):
            hist_obs.append(np.zeros([1, 45], dtype=np.double))

        count_lowlevel = 1
    

        np.set_printoptions(formatter={'float': '{:0.4f}'.format})
        policy_input = np.zeros([1, cfg.env.num_observations*6], dtype=np.float32)
        sim_step = 0
        while True:  # 无限循环，直到手动关闭程序
            # 每帧更新命令（手柄优先，无手柄则用键盘累积值）
            step_start = time.time()  # 实时同步：记录本步开始时间
            cmd = vel_arrow.update_cmd()
            x_vel_cmd, y_vel_cmd, yaw_vel_cmd = cmd[0], cmd[1], cmd[2]

            # Obtain an observation
            q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data,model)
            # q = q[-cfg.env.num_actions:]
            # dq = dq[-cfg.env.num_actions:]
        
            # 1000hz -> 100hz
            if count_lowlevel % cfg.sim_config.decimation == 0:

                obs = np.zeros([1, 45], dtype=np.float32)

                obs[0, 0] = x_vel_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 1] = y_vel_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 2] = yaw_vel_cmd * cfg.normalization.obs_scales.ang_vel
                obs[0, 3:6] =omega*cfg.normalization.obs_scales.ang_vel 
                obs[0, 6:9] = gvec

                obs[0, 9:21] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = action
                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                hist_obs.append(obs)
                hist_obs.popleft()
            
                for i in range(6):

                    policy_input[0, i * 45 : (i + 1) * 45] = hist_obs[i][0, :]
                # 如果 policy_input 在 GPU，先移到 CPU 并转成 numpy
                # with open('policy_input.txt', 'a+') as f:
                #     for row in policy_input:
                #         # 默认用空格分隔列，如需逗号分隔改成 ','.join(...)
                #         f.write(' '.join(map(str, row)) + '\n')
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)

                target_q = action * cfg.control.action_scale

            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)

            # Generate PD control
            if sim_step <100:
                tau = pd_control(np.zeros((cfg.env.num_actions)), q, cfg.robot_config.kps,
                                target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques
                tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            else:
                tau = pd_control(target_q, q, cfg.robot_config.kps,
                                target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques
                tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        
            data.ctrl = tau
            applied_tau = data.actuator_force

            mujoco.mj_step(model, data)

            vel_arrow.draw_arrows(viewer, data)
            viewer.sync()
            # 实时同步：若本步耗时小于仿真步长则等待补足，防止加速播放
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            count_lowlevel += 1
            sim_step += 1
            idx = 5
            dof_pos_target = target_q + cfg.robot_config.default_dof_pos


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default=f'{LEGGED_GYM_ROOT_DIR}/logs/go2_amp_cts/exported/policies/policy_cts.pt',help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(Go2_AMP_Cts_Cfg_Yu):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/go2/scene.xml'
            sim_duration = 120.0
            dt = 0.005
            decimation = 4

        class robot_config:
            kps = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], dtype=np.double)
            kds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.double)
            tau_limit = 45 * np.ones(12, dtype=np.double)
            default_dof_pos = np.array( [0.1,0.8,-1.5,    # FL
                -0.1,0.8,-1.5,     # FR
                 0.1,1.0,-1.5,     # RL
                -0.1,1.0,-1.5], dtype=np.double)  # RR



    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())