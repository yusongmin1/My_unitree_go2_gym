import sys
from legged_gym.envs.GO2_Flip.GO2_BackFlip.GO2_BackFlip_Config import GO2_BackFlip_Cfg_Yu, GO2_BackFlip_PPO_Yu
import math
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import keyboard  # 导入keyboard库
from pynput import keyboard
import yaml
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 1.5, 1.0, 3.0

joystick_use = True
joystick_opened = False
def on_press(key):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
    try:
        if key.char == '6':
            x_vel_cmd =1
        elif key.char == '7':
            x_vel_cmd =0
        elif key.char == '8':
            y_vel_cmd =0
        elif key.char == '9':
            y_vel_cmd=1
        elif key.char == '-':
            yaw_vel_cmd += 0.5
        elif key.char == '=':
            yaw_vel_cmd -= 0.5
        elif key.char == '1':
            x_vel_cmd=0
            y_vel_cmd=0
            yaw_vel_cmd=0
        print(f"Updated velocities: vx={x_vel_cmd}, vy={y_vel_cmd}, dyaw={yaw_vel_cmd}")
    except AttributeError:
        pass
    
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

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
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    
    model.opt.timestep = cfg.sim_config.dt
    
    data = mujoco.MjData(model)
    num_actuated_joints = cfg.env.num_actions  # This should match the number of actuated joints in your model
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos
    count_lowlevel = 1

    mujoco.mj_step(model, data)
    import time
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        start_=time.time()
        hist_obs = deque()
        for _ in range(cfg.env.frame_stack):
                hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
        target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
        action = np.zeros((cfg.env.num_actions), dtype=np.double)
        np.set_printoptions(formatter={'float': '{:0.4f}'.format})
        while viewer.is_running() and time.time() - start < 1000000:

                # Obtain an observation
            q, dq, quat, v, omega, gvec, base_pos, foot_positions, foot_forces = get_obs(data,model)
            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            if count_lowlevel % cfg.sim_config.decimation == 0:

                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi

                obs[0, 0] = math.sin(2 * math.pi *(( count_lowlevel * cfg.sim_config.dt )  %cfg.rewards.cycle_time/ cfg.rewards.cycle_time))
                obs[0, 1] = math.cos(2 * math.pi *(( count_lowlevel * cfg.sim_config.dt )  %cfg.rewards.cycle_time/ cfg.rewards.cycle_time))

                obs[0, 2] =x_vel_cmd
                obs[0, 3] = y_vel_cmd
                obs[0, 4] = 0.6
                obs[0, 5:8] = omega*cfg.normalization.obs_scales.ang_vel
                obs[0, 8:11] = eu_ang*cfg.normalization.obs_scales.quat

                obs[0, 11:23] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 23:35] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 35:47] = action

                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                target_q = action * cfg.control.action_scale
                # Generate PD control
            if time.time() - start_<1:
                    tau = pd_control(np.zeros((cfg.env.num_actions)), q, cfg.robot_config.kps,
                                    target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques
                    tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            else:
                    tau = pd_control(target_q, q, cfg.robot_config.kps,
                                    target_dq, dq, cfg.robot_config.kds, cfg)  # Calc torques
                    tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
    
            data.ctrl = tau
            mujoco.mj_step(model, data)
            current_time = time.time()
            elapsed = current_time - start
                # 确保实时同步

            sleep_time = cfg.sim_config.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            start = current_time 
            count_lowlevel += 1
            viewer.sync()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default="./logs/go2_backflip/exported/policies/policy_4.pt",help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(GO2_BackFlip_Cfg_Yu):
        class sim_config:
            mujoco_model_path = f'./resources/robots/go2/go2/scene.xml'
            sim_duration = 120.0
            dt = 0.005
            decimation = 4

        class robot_config:
            kps = np.array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], dtype=np.double)
            kds = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.double)
            tau_limit = 45 * np.ones(12, dtype=np.double)
            default_dof_pos = np.array( [0.0,0.8,-1.5,
                -0.0,0.8,-1.5,
                 0.0,0.8,-1.5,
                -0.,0.8 ,-1.5], dtype=np.double)

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())