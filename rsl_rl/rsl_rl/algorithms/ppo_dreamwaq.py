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

from rsl_rl.modules import ActorCriticDreamWaQ
from rsl_rl.storage import RolloutStorageDreamWaQ

class PPO_DreamWaQ:
    actor_critic: ActorCriticDreamWaQ
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.99,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 sym_loss=False,
                 sym_coef=1.0,
                 frame_stack=None,
                 obs_permutation=None,
                 act_permutation=None,
                 privileged_permutation=None,
                 terrain_permutation=None,
                 device='cpu',
                 vae_learning_rate=1e-3,
                 vae_kl_weight=1.0,
                 num_obs=45,
                **kwargs):
        if kwargs:
            print("ActorCritic_DWAQ.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))

        self.device = device
        self.num_obs=num_obs
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.vae_learning_rate = vae_learning_rate
        self.bata=vae_kl_weight
        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None 
        # self.rl_parameters = list(self.actor_critic.actor.parameters()) + \
        #                      list(self.actor_critic.critic.parameters()) + \
        #                     [self.actor_critic.std]
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=learning_rate)
        self.vae_optimizer = optim.Adam(
            self.actor_critic.vae.parameters(), lr=vae_learning_rate)
        self.transition = RolloutStorageDreamWaQ.Transition()

        # ===== 镜像对称损失（默认关闭）=====
        self.sym_loss = sym_loss
        self.sym_coef = sym_coef
        if self.sym_loss:
            from rsl_rl.utils.utils import build_sym_perm_matrix
            self.sym_obs_P = build_sym_perm_matrix(obs_permutation, stack=1, device=self.device)
            self.sym_act_P = build_sym_perm_matrix(act_permutation, stack=1, device=self.device)
            fs = frame_stack if frame_stack else 1
            self.sym_hist_P = build_sym_perm_matrix(obs_permutation, stack=fs, device=self.device)


        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_hist_shape,estimation_shape, action_shape):
        self.storage = RolloutStorageDreamWaQ(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, obs_hist_shape, estimation_shape,action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs,  obs_history,estimation):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs,obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.obs_hist = obs_history
        self.transition.privileged_observations = critic_obs
        self.transition.estimation = estimation
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_vae_loss = 0
        mean_vel_estimation_loss=0
        mean_recon_loss=0
        mean_kl_loss=0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch,  obs_hist_batch,estimation_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, live_batch,hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, obs_hist_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy
                mean_entropy_loss += entropy_batch.mean().item()
                # 对称损失：镜像 obs/hist 经 VAE+actor 前向，与原 mu 对比
                sym_loss_val = torch.tensor(0., device=self.device)
                if self.sym_loss:
                    m_obs = obs_batch @ self.sym_obs_P
                    m_hist = obs_hist_batch @ self.sym_hist_P
                    code_m, _, _, _ = self.actor_critic.vae.cenet_forward(m_hist)
                    mu_m = self.actor_critic.actor(torch.cat((code_m, m_obs), dim=-1))
                    sym_loss_val = (mu_batch - mu_m @ self.sym_act_P).pow(2).mean()


                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.sym_coef * sym_loss_val

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                vel_target = estimation_batch
                decode_target = critic_obs_batch[:,-self.num_obs:]
                vel_target.requires_grad = False
                decode_target.requires_grad = False
                (code),(code_vel,code_latent),(decode),(mean_vel,logvar_vel,mean_latent,logvar_latent) = self.actor_critic.vae.cenet_forward(obs_hist_batch)

                vel_estimation_loss=nn.MSELoss()(code_vel*live_batch,vel_target*live_batch) 
                recon_loss= nn.MSELoss()(decode*live_batch,decode_target*live_batch)
                kl_loss= -0.5 *  torch.mean(torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp(), dim=-1)* live_batch)
                vae_loss=vel_estimation_loss+recon_loss+self.bata*kl_loss
                self.vae_optimizer.zero_grad()
                vae_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.vae.parameters(), self.max_grad_norm)
                self.vae_optimizer.step()
                
                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_vae_loss += vae_loss.item()
                mean_vel_estimation_loss += vel_estimation_loss.item()
                mean_recon_loss += recon_loss.item()
                mean_kl_loss += kl_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_vae_loss/=num_updates
        mean_vel_estimation_loss /= num_updates
        mean_recon_loss /= num_updates
        mean_kl_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_vae_loss,mean_vel_estimation_loss, \
            mean_recon_loss, mean_kl_loss, mean_entropy_loss
