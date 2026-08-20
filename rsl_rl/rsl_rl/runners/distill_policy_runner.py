# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms.ppo_amp_ts import PPO_AMP_TS
from rsl_rl.modules import ActorCritic_Distill
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer
from rsl_rl.modules.actor_critic_distill import LSTMEncoder
from copy import deepcopy
class DistillPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.lstm_cfg = train_cfg["LSTMEncoder"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        self.actor_critic=ActorCritic_Distill( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        num_privileged_input=self.env.privileged_buf.shape[-1],  # 实际维度=3(线速度)+域随机参数+4(足接触)
                                                        LSTM_INPUT_SIZE=self.lstm_cfg["input_size"],
                                                        LSTM_HIDDEN_SIZE=self.lstm_cfg["hidden_size"],
                                                        LSTM_NUM_LAYERS=self.lstm_cfg["num_layers"],
                                                        **self.policy_cfg).to(self.device)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.lstm_actor=deepcopy(self.actor_critic.actor)
        self.lstm_encoder=LSTMEncoder(45, 256, 3).to(self.device)
        self.lstm_encoder_optimizer = optim.Adam(self.lstm_encoder.parameters(), lr=1e-3)
        self.lstm_actor_optimizer = optim.Adam(self.lstm_actor.parameters(), lr=1e-3)
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.env.reset()
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        obs,terrain_obs,domain_rand_obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs ,terrain_obs,domain_rand_obs= obs.to(self.device), critic_obs.to(self.device), terrain_obs.to(self.device),domain_rand_obs.to(self.device)
        self.lstm_encoder.train() # switch to train mode (for dropout for example)
        self.lstm_actor.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            teacher_encode_buffer = []
            student_encode_buffer = []
            actions_teacher_buffer = []
            actions_student_buffer = []
            for i in range(self.num_steps_per_env):
                with torch.no_grad():
                    actions_teacher,teacher_encode = self.actor_critic.act_inference(obs, terrain_obs,domain_rand_obs)
                    actions_teacher_buffer.append(actions_teacher)
                    teacher_encode_buffer.append(teacher_encode)

                obs_student = obs.clone()
                student_encode=self.lstm_encoder(obs_student)
                student_encode_buffer.append(student_encode)
                actions_student =self.lstm_actor(torch.cat([student_encode,obs],dim=-1))
                actions_student_buffer.append(actions_student)

                if it < 1:
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states,terrain_obs,domain_rand_obs = self.env.step(actions_teacher.detach())  # obs has changed to next_obs !! if done obs has been reset
                else:
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states,terrain_obs,domain_rand_obs = self.env.step(actions_student.detach())
                obs, critic_obs, rewards, dones ,terrain_obs,domain_rand_obs= obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device),terrain_obs.to(self.device),domain_rand_obs.to(self.device)
                
                # self.actor_critic.lstm_encoder.reset_hidden_states(dones)
                if self.log_dir is not None:
                    # Book keeping
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    # print(rewbuffer)
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            # Learning step
            start = stop
            teacher_encode_buffer = torch.cat(teacher_encode_buffer, dim=0)
            student_encode_buffer = torch.cat(student_encode_buffer, dim=0)
            actions_teacher_buffer = torch.cat(actions_teacher_buffer, dim=0)
            actions_student_buffer = torch.cat(actions_student_buffer, dim=0)
            lstm_encoder_loss,lstm_actor_loss =  self.update_lstm_encoder(student_encode_buffer,teacher_encode_buffer ,actions_student_buffer, actions_teacher_buffer)
            stop = time.time()
            learn_time = stop - start

            self.lstm_encoder.detach_hidden_states()


            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_student_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss_lstm/lstm_encoder', locs['lstm_actor_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                           f"""{'lstm encoder loss:':>{pad}} {locs['lstm_encoder_loss']:.4f}\n"""
                           f"""{'lstm actor loss:':>{pad}} {locs['lstm_actor_loss']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                           f"""{'lstm encoder loss:':>{pad}} {locs['lstm_encoder_loss']:.4f}\n"""
                           f"""{'lstm actor loss:':>{pad}} {locs['lstm_actor_loss']:.4f}\n""")
                        #   f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                        #   f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            "actor_state_dict": self.lstm_actor.state_dict(),
            "lstm_state_dict": self.lstm_encoder.state_dict()
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.lstm_actor.load_state_dict(self.actor_critic.actor.state_dict())
        if loaded_dict.get('lstm_state_dict', None) is not None:
            self.lstm_encoder.load_state_dict(loaded_dict['lstm_state_dict'])

    def get_inference_policy(self, device=None):
        self.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.actor_critic.to(device)
        return self.actor_critic.act_inference

    def get_inference_policy_student(self, device=None):
        self.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.actor_critic.to(device)
        return self.act_inference_student
    
    def act_inference_student(self, obs):
        with torch.no_grad():
            act=self.lstm_actor(torch.cat((self.lstm_encoder(obs),obs),dim=-1))
        return act
    
    def update_lstm_encoder(self, student_encode_buffer, teacher_encode_buffer,actions_student_buffer, actions_teacher_buffer):
            depth_encoder_loss = (teacher_encode_buffer.detach() - student_encode_buffer).norm(p=2, dim=1).mean()
            depth_actor_loss = (actions_teacher_buffer.detach() - actions_student_buffer).norm(p=2, dim=1).mean()


            depth_loss = depth_actor_loss +depth_encoder_loss #+
            self.lstm_encoder_optimizer.zero_grad()
            self.lstm_actor_optimizer.zero_grad()
            depth_loss.backward()

            nn.utils.clip_grad_norm_(self.lstm_encoder.parameters(), 1)
            nn.utils.clip_grad_norm_(self.lstm_actor.parameters(), 1)

            self.lstm_encoder_optimizer.step()
            self.lstm_actor_optimizer.step()

            return depth_encoder_loss.item(), depth_actor_loss.item()     