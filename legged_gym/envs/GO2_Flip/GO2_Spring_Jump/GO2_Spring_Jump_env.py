import os
import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from collections import deque
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.GO2_Flip.GO2_Spring_Jump.GO2_Spring_Jump_Config import GO2_Spring_Jump_Cfg_Yu

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz
class GO2_Spring_Jump_Robot(BaseTask):
    def __init__(self, cfg: GO2_Spring_Jump_Cfg_Yu, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self.height_buf=torch.zeros(self.num_envs,device=self.device)

        self._prepare_reward_function()
        self.reset()
        self.prob=0
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            action_delayed = self.update_cmd_action_latency_buffer()
            self.torques = self._compute_torques(action_delayed).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.update_obs_latency_buffer()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.cfg.env.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(
            self.cfg.env.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
    
    def update_cmd_action_latency_buffer(self):
        actions_scaled = self.actions * self.cfg.control.action_scale
        if self.cfg.domain_rand.add_cmd_action_latency:
            self.cmd_action_latency_buffer[:,:,1:] = self.cmd_action_latency_buffer[:,:,:self.cfg.domain_rand.range_cmd_action_latency[1]].clone()
            self.cmd_action_latency_buffer[:,:,0] = actions_scaled.clone()
            action_delayed = self.cmd_action_latency_buffer[torch.arange(self.num_envs),:,self.cmd_action_latency_simstep.long()]
        else:
            action_delayed = actions_scaled
        
        return action_delayed

    def update_obs_latency_buffer(self):
        if self.cfg.domain_rand.randomize_obs_motor_latency:
            q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
            dq = self.dof_vel * self.obs_scales.dof_vel
            self.obs_motor_latency_buffer[:,:,1:] = self.obs_motor_latency_buffer[:,:,:self.cfg.domain_rand.range_obs_motor_latency[1]].clone()
            self.obs_motor_latency_buffer[:,:,0] = torch.cat((q, dq), 1).clone()
        if self.cfg.domain_rand.randomize_obs_imu_latency:
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.base_quat[:] = self.root_states[:, 3:7]
            self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
            self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
            self.obs_imu_latency_buffer[:,:,1:] = self.obs_imu_latency_buffer[:,:,:self.cfg.domain_rand.range_obs_imu_latency[1]].clone()
            self.obs_imu_latency_buffer[:,:,0] = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, self.base_euler_xyz * self.obs_scales.quat), 1).clone()

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.commands[self.episode_length_buf==self.command_frame,2]=1.0
        self.check_jump()
        self.check_termination()
        self.compute_reward()
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        if self.cfg.env.test:
            self._draw_debug_goal()
        env_ids = self.not_pushed_up * ~self.has_jumped * (self.commands[:,2]==1)
        self.max_height=torch.maximum(self.max_height,self.root_states[:,2])
        if self.cfg.domain_rand.push_towards_goal and torch.any(env_ids):
            self._push_robots_upwards(env_ids)
            self.not_pushed_up[env_ids] = False        
        # env_ids =~self.not_pushed_up*(self.commands[:,2]==1)*self.not_pushed_rotot
        # if self.cfg.domain_rand.push_towards_goal and torch.any(env_ids):
        #     self._push_robots_desired(env_ids)
        #     self.not_pushed_rotot[env_ids] =False
    def _push_robots_upwards(self,env_ids):

        random_push = torch.randint(0,10,(self.num_envs,1),device=self.device).squeeze()
        self.prob=max(8-int(self.count/(24*50)),0)
        env_ids = torch.logical_and(random_push<self.prob,env_ids)

        self.root_states[env_ids,9] += torch_rand_float(1.5, 2.2, (self.num_envs, 1), device=self.device).flatten()[env_ids]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots_desired(self,env_ids):
        """ Randomly pushes some robots towards the goal just before takeoff. Emulates an impulse by setting a randomized base velocity. 
        """
        
        random_push = torch.randint(0,10,(self.num_envs,1),device=self.device).squeeze()
        env_ids = torch.logical_and(random_push<self.prob,env_ids)
        self.root_states[env_ids,7] += torch_rand_float(0.0,1.0, (self.num_envs, 1), device=self.device).flatten()[env_ids]
        # self.root_states[env_ids,11] -= 0.1
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))  

    def check_jump(self):
        """ Check if the robot has jumped
        """
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # print("contact",contact)
        self.contact_filt =torch.logical_or(contact, self.last_contacts)
        self.last_contacts=contact.clone()

        jump_filter = torch.all(~self.contact_filt , dim=1)

        was_in_flight=torch.logical_and(jump_filter,self.commands[:,2]>0)
        self.was_in_flight[was_in_flight] = True
        has_jumped = torch.logical_and(torch.any(self.contact_filt ,dim=1), self.was_in_flight) #飞起来过并且落地就是已经跳跃过了
        
        self.landing_poses[torch.logical_and(has_jumped,~self.has_jumped)] = self.root_states[torch.logical_and(has_jumped,~self.has_jumped),:2]
        # Only count the first time flight is achieved:
        self.has_jumped[has_jumped] = True 

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf[self.root_states[:,2]  <= self.cfg.env.reset_height] = True # check if the robot base is below 0.1 metres
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.was_in_flight[env_ids] = False
        self.has_jumped[env_ids] = False
        self.landing_poses[env_ids,:] = self.init_state[env_ids,:2]
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.max_height[env_ids]=0.0
        self.last_contacts[env_ids] = 0
        self.not_pushed_up[env_ids] = True
        self.not_pushed_rotot[env_ids] =True
        self.command_frame=torch.randint(50,60,(self.num_envs,),device=self.device)
        # self.height_buf[env_ids] = 0.0

        self.commands[env_ids, 0] = torch_rand_float(0.8,1.2, (1,1), device=self.device)
        self.commands[env_ids, 1] = 0.0

        self.commands[env_ids, 2] = 0

        self._reset_latency_buffer(env_ids)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        # fix reset gravity bug
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        #这里要把全局坐标系下的速度和角速度都转换到局部坐标系下，控制前进是控制向基座坐标系下的前进方向，而不是全局坐标系下的前进方向
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):


        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.privileged_obs_buf = torch.cat((
            self.commands,  # 2 + 3 控制输入 ，相位，目标速度，角速度
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12  当前关节位置与默认关节位置之差
            self.dof_pos * self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 速度乘以缩放因子 12
            self.actions,  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz *self.cfg.normalization.obs_scales.quat,  # 3
            contact_mask,  # 2    
            self.has_jumped.unsqueeze(1),
            
        ), dim=-1)#5 12 12 12 12 3 3 3 1 1 2 2 =68
        # print("self.privileged_obs_buf",self.privileged_obs_buf.shape)
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel

        if self.cfg.domain_rand.randomize_obs_motor_latency:
            self.obs_motor = self.obs_motor_latency_buffer[torch.arange(self.num_envs), :, self.obs_motor_latency_simstep.long()]#????
        else:
            self.obs_motor = torch.cat((q, dq), 1)

        if self.cfg.domain_rand.randomize_obs_imu_latency:
            self.obs_imu = self.obs_imu_latency_buffer[torch.arange(self.num_envs), :, self.obs_imu_latency_simstep.long()]
        else:              
            self.obs_imu = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, self.base_euler_xyz * self.obs_scales.quat), 1)

        obs_buf = torch.cat((
            torch.zeros((self.num_envs, 2), device=self.device),
            self.commands,
            self.obs_imu,#6 角速度，欧拉角XYZ
            self.obs_motor,#24
            self.actions,   # 12
        ), dim=-1)
        # print("obs_buf",obs_buf.shape)
        if self.add_noise:  
            obs_now = obs_buf.clone() + (2 * torch.rand_like(obs_buf) -1) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)

        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

                 # randomization of the motor zero calibration for real machine
        if self.cfg.domain_rand.randomize_motor_zero_offset:
            self.motor_zero_offsets[env_id, :] = torch_rand_float(self.cfg.domain_rand.motor_zero_offset_range[0], self.cfg.domain_rand.motor_zero_offset_range[1], (1,self.num_actions), device=self.device)
        
        # randomization of the motor pd gains
        if self.cfg.domain_rand.randomize_pd_gains:
            self.p_gains_multiplier[env_id, :] = torch_rand_float(self.cfg.domain_rand.stiffness_multiplier_range[0], self.cfg.domain_rand.stiffness_multiplier_range[1], (1,self.num_actions), device=self.device)
            self.d_gains_multiplier[env_id, :] =  torch_rand_float(self.cfg.domain_rand.damping_multiplier_range[0], self.cfg.domain_rand.damping_multiplier_range[1], (1,self.num_actions), device=self.device)   
        
        return props

        return props
    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            self.added_base_masses = torch_rand_float(self.cfg.domain_rand.added_base_mass_range[0], self.cfg.domain_rand.added_base_mass_range[1], (1, 1), device=self.device)
            props[0].mass += self.added_base_masses

        # randomize link masses
        if self.cfg.domain_rand.randomize_link_mass:
            self.multiplied_link_masses_ratio = torch_rand_float(self.cfg.domain_rand.multiplied_link_mass_range[0], self.cfg.domain_rand.multiplied_link_mass_range[1], (1, self.num_bodies-1), device=self.device)
    
            for i in range(1, len(props)):
                props[i].mass *= self.multiplied_link_masses_ratio[0,i-1]

        # randomize base com
        if self.cfg.domain_rand.randomize_base_com:
            self.added_base_com = torch_rand_float(self.cfg.domain_rand.added_base_com_range[0], self.cfg.domain_rand.added_base_com_range[1], (1, 3), device=self.device)
            props[0].com += gymapi.Vec3(self.added_base_com[0, 0], self.added_base_com[0, 1],
                                    self.added_base_com[0, 2])

        return props

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        p_gains = self.p_gains * self.p_gains_multiplier
        d_gains = self.d_gains * self.d_gains_multiplier

        torques = p_gains * (actions + self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains * self.dof_vel

        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos 
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.init_state[env_ids]=self.root_states[env_ids]

        # base velocities
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_latency_buffer(self,env_ids):
        if self.cfg.domain_rand.add_cmd_action_latency:   
            self.cmd_action_latency_buffer[env_ids, :, :] = 0.0
            if self.cfg.domain_rand.randomize_cmd_action_latency:
                self.cmd_action_latency_simstep[env_ids] = torch.randint(self.cfg.domain_rand.range_cmd_action_latency[0], 
                                                           self.cfg.domain_rand.range_cmd_action_latency[1]+1,(len(env_ids),),device=self.device) 
            else:
                self.cmd_action_latency_simstep[env_ids] = self.cfg.domain_rand.range_cmd_action_latency[1]
                               
        if self.cfg.domain_rand.add_obs_latency:
            self.obs_motor_latency_buffer[env_ids, :, :] = 0.0
            self.obs_imu_latency_buffer[env_ids, :, :] = 0.0
            if self.cfg.domain_rand.randomize_obs_motor_latency:
                self.obs_motor_latency_simstep[env_ids] = torch.randint(self.cfg.domain_rand.range_obs_motor_latency[0],
                                                        self.cfg.domain_rand.range_obs_motor_latency[1]+1, (len(env_ids),),device=self.device)
            else:
                self.obs_motor_latency_simstep[env_ids] = self.cfg.domain_rand.range_obs_motor_latency[1]

            if self.cfg.domain_rand.randomize_obs_imu_latency:
                self.obs_imu_latency_simstep[env_ids] = torch.randint(self.cfg.domain_rand.range_obs_imu_latency[0],
                                                        self.cfg.domain_rand.range_obs_imu_latency[1]+1, (len(env_ids),),device=self.device)
            else:
                self.obs_imu_latency_simstep[env_ids] = self.cfg.domain_rand.range_obs_imu_latency[1]

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self,env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if not self.init_done:
            # don't change on initial reset
            return
        success_rate = torch.sum(self.success_rate[env_ids], dim=1)/(self.success_rate.shape[-1])
        # robots that were sucessful for enough trials should be moved to harder jumps:
        # Currently if it succeeds once it's moved up
        move_up = torch.logical_and(success_rate > 0.0, torch.all(self.success_rate[env_ids] >= 0.))
        # robots that were not successful for enough trials should be moved to easier jumps:
        # if it fails in both trials it's moved down
        move_down = torch.logical_and(success_rate <= 0.0, torch.all(self.success_rate[env_ids] >= 0.)) * ~move_up
       
        self.command_dist_levels[env_ids] += 1 * move_up #- 1 * move_down
        self.command_dist_levels[env_ids] -= 1 * move_down
        

        max_command_dist_level = self.cfg.commands.num_levels - 1

        self.reset_landing_error[env_ids * (self.command_dist_levels[env_ids] >= max_command_dist_level/2)] -= 1 * move_up
        self.reset_landing_error[env_ids * (self.command_dist_levels[env_ids] >= max_command_dist_level/2)] += 1 * move_down
        self.reset_landing_error[env_ids * (self.command_dist_levels[env_ids] < max_command_dist_level/2)] = max_command_dist_level
        

        self.command_dist_levels = torch.clip(self.command_dist_levels, min=0, max=max_command_dist_level) # (the minumum level is zero)
        self.reset_landing_error = torch.clip(self.reset_landing_error, min=0, max=max_command_dist_level)
        # Reset success rate for robots that have changed difficulty levels
        self.success_rate[env_ids[torch.logical_or(move_up,move_down)]] = -1.



    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands

        noise_vec[5:8] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[8:11] = noise_scales.quat         # euler x,y
        noise_vec[11: 23] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[23: 35] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[35: 47] = 0.  # previous actions
        return noise_vec
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) 
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.gym.refresh_dof_state_tensor(self.sim)  #刷新关节状态张量，确保其包含最新的物理状
        self.gym.refresh_actor_root_state_tensor(self.sim)#刷新刚体根状态张量，确保其包含最新的物理状态。
        self.gym.refresh_net_contact_force_tensor(self.sim)#刷新净接触力张量，确保其包含最新的物理状态。
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.init_state=torch.zeros_like(self.root_states)
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) 
        self.commands[:, 0] = 0.6
        self.command_frame=torch.randint(50,100,(self.num_envs,),device=self.device)
        self.max_height=torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.was_in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.has_jumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.ori_error = torch.zeros(self.num_envs,dtype=torch.float, device=self.device, requires_grad=False)
        self.reset_idx_landing_error= torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.landing_poses = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.count=torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=False)

        self.success_rate = -torch.ones(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])

        self.not_pushed_up = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.not_pushed_rotot = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.lie_joint_pos= torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.jump_command=torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            angle_2 = self.cfg.init_state.lie_joint_angles[name]
            
            self.default_dof_pos[i] = angle
            self.lie_joint_pos[i] = angle_2
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.obs_history = deque(maxlen=self.cfg.env.frame_stack)
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history.append(torch.zeros(
                self.num_envs, self.cfg.env.num_single_obs, dtype=torch.float, device=self.device))
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, self.cfg.env.single_num_privileged_obs, dtype=torch.float, device=self.device))
        #通信延迟 cmd延迟，obs延迟 ，imu延迟
        self.cmd_action_latency_buffer = torch.zeros(self.num_envs,self.num_actions,self.cfg.domain_rand.range_cmd_action_latency[1]+1,device=self.device)
        self.obs_motor_latency_buffer = torch.zeros(self.num_envs,self.num_actions * 2,self.cfg.domain_rand.range_obs_motor_latency[1]+1,device=self.device)
        self.obs_imu_latency_buffer = torch.zeros(self.num_envs, 6, self.cfg.domain_rand.range_obs_imu_latency[1]+1,device=self.device)
        self.cmd_action_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) 
        self.obs_motor_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) 
        self.obs_imu_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  
        self._reset_latency_buffer(torch.arange(self.num_envs, device=self.device))
    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, which will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        self.motor_zero_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                         requires_grad=False) 
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains_multiplier = torch.ones(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains_multiplier = torch.ones(self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

    def _draw_debug_goal(self):
        """ Draws desired goal points for debugging.
            Default behaviour: draws goal position
        """
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        sphere_geom_start = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))

        for i in range(self.num_envs):
            goal_pos = (self.init_state[i, :3] + self.commands[i,0:3]).cpu().numpy()
            goal_pos[2] = 0
            # goal_pos = self.env_origins[i].clone()
            # Plot the xy position as a sphere on the ground plane
            sphere_pose = gymapi.Transform(gymapi.Vec3(goal_pos[0], goal_pos[1], goal_pos[2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

            sphere_pose_start = gymapi.Transform(gymapi.Vec3(self.init_state[i, 0], self.init_state[i, 1], 0.0), r=None)
            gymutil.draw_lines(sphere_geom_start, self.gym, self.viewer, self.envs[i], sphere_pose_start)
    #------------ reward functions----------------
    def _reward_before_setting(self):
        #切换到蹲姿状态之前的奖励函数
        rew = torch.exp(-torch.sum(torch.abs(self.dof_pos-self.default_dof_pos),dim=1)/2)*(self.commands[:, 2] == 0)
        return rew
    
    def _reward_line_z(self):
        #在初始化后和落地之前z轴线速度越大越好
        rew=(self.root_states[:, 9]>0)*self.root_states[:, 9] *(~self.has_jumped)*(self.commands[:, 2] == 1)
        return rew
    
    def _reward_land_pos(self):
        #xy轴角速度奖励
        land_err=self.init_state[:,:2]+self.commands[:,:2]-self.landing_poses
        # print(land_err[0])
        return torch.exp(-torch.sum(torch.abs(land_err),dim=1))*(self.has_jumped)*(self.base_euler_xyz.sum(dim=1)<0.6)*(self.max_height>0.42)


    def _reward_base_height_flight(self):
        #跳跃的高度奖励
        base_height_flight = (self.root_states[:, 2] - 0.47)
        rew= torch.exp(-torch.abs(base_height_flight)*5)*(self.was_in_flight)*~self.has_jumped*6
        return rew 
    
    def _reward_base_height_stance(self):
        #落地后的高度奖励和默认关节角度的奖励
        return torch.abs((self.root_states[:, 2] - 0.3))*self.has_jumped + 0.2*torch.abs((self.root_states[:, 2] - 0.25))*(self.commands[:,2]==0)
    
    def _reward_dof_pos(self):
        #落地后的高度奖励和默认关节角度的奖励
        return torch.abs(self.dof_pos - self.default_dof_pos).sum(dim=1)

    def _reward_dof_hip_pos(self):
        #落地后的高度奖励和默认关节角度的奖励
        rew=torch.abs(self.dof_pos - self.default_dof_pos)
        return rew[:,0]+rew[:,3]+rew[:,6]+rew[:,9]
        
    def _reward_orientation(self):
        #rewer=self.base_euler_xyz
        rew=torch.exp(-torch.abs(self.base_euler_xyz).sum(dim=1))
        return rew

    def _reward_ang_vel_xy(self):
        rew=torch.sum(torch.abs(self.base_ang_vel),dim=1)
        return rew

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)
    
    def _reward_collision(self):
        return torch.sum(1.*(torch.linalg.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits).clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits).clip(min=0.), dim=1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        self.count+=1
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_flight(self):
        # Penalize dof velocities
        return self.was_in_flight
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.square(self.commands[:, 0]*1.6 - self.base_lin_vel[:, 0])
        # print(self.commands[0, 0]*2 , self.base_lin_vel[0, 0])
        return torch.exp(-lin_vel_error)*self.was_in_flight*~self.has_jumped*5
    
    def _reward_line_vel_stance(self):
        # Penalize dof velocities
        return torch.sum(torch.abs(self.base_lin_vel[:,:2]), dim=1)*(self.has_jumped)
    

    def _reward_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        height_error = torch.abs(footpos_in_body_frame[:, :, 2] +0.20)
        return torch.sum(height_error, dim=1) *self.was_in_flight*(~self.has_jumped) *6