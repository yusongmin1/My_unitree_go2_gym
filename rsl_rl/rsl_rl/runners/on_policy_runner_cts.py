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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import CTS
from rsl_rl.modules import ActorCriticCTS
from rsl_rl.env import VecEnv


class OnPolicyRunnerCTS:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        num_critic_obs = self.env.num_critic_obs  #critic network input size
        model=ActorCriticCTS(
            self.env.num_obs,
            self.env.num_privileged_obs,#随机化加线速度加地形
            num_critic_obs,
            self.env.num_actions,
            self.env.cfg.env.num_obs_hist,
            **self.policy_cfg).to(self.device)
        self.alg=CTS(model, self.env.num_envs, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_critic_obs],[self.env.num_history_obs],[self.env.num_actions])

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.reset()
        print("地球导演到此一游")
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs, privileged_buf, obs_history, critic_obs_buf = self.env.get_observations()
        obs, privileged_buf, obs_history, critic_obs_buf = obs.to(self.device), privileged_buf.to(self.device), obs_history.to(self.device), critic_obs_buf.to(self.device)
        self.alg.model.train() # switch to train mode (for dropout for example)

        ep_infos = []
        teacher_rewbuffer = deque(maxlen=100)
        teacher_lenbuffer = deque(maxlen=100)
        student_rewbuffer = deque(maxlen=100)
        student_lenbuffer = deque(maxlen=100)

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        self.start_learning_iteration = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_buf,critic_obs_buf, obs_history)
                    obs, privileged_buf, obs_history, critic_obs_buf , rewards, dones, infos = self.env.step(actions)
                    obs, privileged_buf, obs_history, critic_obs_buf , rewards, dones, infos = obs.to(self.device), privileged_buf.to(self.device), obs_history.to(self.device), critic_obs_buf.to(self.device), rewards.to(self.device), dones.to(self.device), infos
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        if new_ids.shape[0]:
                            ti = self.alg.teacher_env_idxs
                            teacher_ids = new_ids[torch.isin(new_ids, ti)]
                            student_ids = new_ids[~torch.isin(new_ids, ti)]
                            teacher_rewbuffer.extend(cur_reward_sum[teacher_ids].cpu().numpy().tolist())
                            teacher_lenbuffer.extend(cur_episode_length[teacher_ids].cpu().numpy().tolist())
                            student_rewbuffer.extend(cur_reward_sum[student_ids].cpu().numpy().tolist())
                            student_lenbuffer.extend(cur_episode_length[student_ids].cpu().numpy().tolist())
                            cur_reward_sum[new_ids] = 0
                            cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                start = stop


                self.alg.compute_returns(privileged_buf,critic_obs_buf,obs_history)
            
            mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration += 1
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), it, False)
            ep_infos.clear()
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)), it, True)

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
                if 'terrain' in key:
                    self.writer.add_scalar('Terrain/' + key, value, locs['it'])
                else:
                    self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        if 'mcp' not in self.cfg["algorithm_class_name"].lower():
            mean_std = self.alg.model.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/entropy', locs['mean_entropy_loss'], locs['it'])
        self.writer.add_scalar('Loss/latent', locs['mean_latent_loss'], locs['it'])
        if len(locs['teacher_rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_teacher_reward', statistics.mean(locs['teacher_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_teacher_episode_length', statistics.mean(locs['teacher_lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_teacher_reward/time', statistics.mean(locs['teacher_rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_teacher_episode_length/time', statistics.mean(locs['teacher_lenbuffer']), self.tot_time)
        if len(locs['student_rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_student_reward', statistics.mean(locs['student_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_student_episode_length', statistics.mean(locs['student_lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_student_reward/time', statistics.mean(locs['student_rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_student_episode_length/time', statistics.mean(locs['student_lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {self.current_learning_iteration}/{locs['tot_iter']} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                      f"""{str.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                      'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                      f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                      f"""{'Entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                      f"""{'Latent loss:':>{pad}} {locs['mean_latent_loss']:.4f}\n""")
        if len(locs['teacher_rewbuffer']):
            log_string += (f"""{'Mean teacher reward:':>{pad}} {statistics.mean(locs['teacher_rewbuffer']):.2f}\n"""
                           f"""{'Mean teacher episode length:':>{pad}} {statistics.mean(locs['teacher_lenbuffer']):.2f}\n""")
        if len(locs['student_rewbuffer']):
            log_string += (f"""{'Mean student reward:':>{pad}} {statistics.mean(locs['student_rewbuffer']):.2f}\n"""
                           f"""{'Mean student episode length:':>{pad}} {statistics.mean(locs['student_lenbuffer']):.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (self.current_learning_iteration - self.start_learning_iteration) * (
                               locs['tot_iter'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, it, last_model, infos=None):
        torch.save({
            'model_state_dict': self.alg.model.state_dict(),
            'optimizer1_state_dict': self.alg.optimizer1.state_dict(),
            'optimizer2_state_dict': self.alg.optimizer2.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.model.load_state_dict(loaded_dict['model_state_dict'])
        # if load_optimizer:
        #     self.alg.optimizer1.load_state_dict(loaded_dict['optimizer1_state_dict'])
        #     self.alg.optimizer2.load_state_dict(loaded_dict['optimizer2_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.model.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.model.to(device)
        return self.alg.model.act_inference
