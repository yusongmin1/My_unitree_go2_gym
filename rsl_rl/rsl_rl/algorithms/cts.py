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

import itertools
from rsl_rl.modules import ActorCriticCTS
from rsl_rl.storage import RolloutStorageCTS

class CTS:
    model: ActorCriticCTS
    def __init__(self,
                 model,
                 num_envs,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 student_encoder_learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 teacher_env_ratio=0.75,
                 sym_loss=False,
                 sym_coef=1.0,
                 frame_stack=None,
                 obs_permutation=None,
                 act_permutation=None,
                 privileged_permutation=None,
                 terrain_permutation=None,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # CTS components
        self.model = model
        self.model.to(self.device)
        self.storage = None # initialized later
        params1 = [
            {"params": self.model.teacher_encoder.parameters()},
            {"params": self.model.critic.parameters()},
            {"params": self.model.actor.parameters()},
            {"params": self.model.std}
        ]
        self.optimizer1 = optim.Adam(params1, lr=learning_rate)
        self.optimizer2 = optim.Adam(self.model.student_encoder.parameters(), lr=student_encoder_learning_rate)
        self.transition = RolloutStorageCTS.Transition()

        # CTS parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.teacher_num_envs = int(num_envs * teacher_env_ratio)
        self.student_num_envs = num_envs - self.teacher_num_envs
        student_env_ratio = 1 - teacher_env_ratio
        self.teacher_env_idxs = torch.tensor([i for i in range(num_envs) if i % int(1/student_env_ratio) != 0], device=self.device)
        self.student_env_idxs = torch.tensor([i for i in range(num_envs) if i % int(1/student_env_ratio) == 0], device=self.device)
        assert num_envs%4==0, f"{num_envs%4!= 0}"
        assert len(self.teacher_env_idxs) == self.teacher_num_envs, f"{len(self.teacher_env_idxs)=} != {self.teacher_num_envs=}"
        assert len(self.student_env_idxs) == self.student_num_envs, f"{len(self.student_env_idxs)=} != {self.student_num_envs=}"

        # ===== 镜像对称损失（默认关闭）=====
        self.sym_loss = sym_loss
        self.sym_coef = sym_coef
        if self.sym_loss:
            from rsl_rl.utils.utils import build_sym_perm_matrix
            self.sym_obs_P = build_sym_perm_matrix(obs_permutation, stack=1, device=self.device)
            self.sym_act_P = build_sym_perm_matrix(act_permutation, stack=1, device=self.device)
            fs = frame_stack if frame_stack else 1
            self.sym_hist_P = build_sym_perm_matrix(obs_permutation, stack=fs, device=self.device)
            self.sym_priv_P = build_sym_perm_matrix(privileged_permutation, stack=1, device=self.device)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,critic_obs_shape,history_shape, action_shape):
        self.storage = RolloutStorageCTS(num_envs, self.teacher_num_envs, num_transitions_per_env, actor_obs_shape,privileged_obs_shape,critic_obs_shape,history_shape,action_shape, self.device)

    def test_mode(self):
        self.model.test()
    
    def train_mode(self):
        self.model.train()

    def act(self, obs, privileged_obs,critic_obs, history):
        history = history.clone()
        def get_results(obs, privileged_obs, critic_obs,history, is_teacher):
            actions = self.model.act(obs, privileged_obs, history, is_teacher).detach()
            return (
                actions,
                self.model.evaluate(privileged_obs,critic_obs, history, is_teacher).detach(),
                self.model.get_actions_log_prob(actions).detach(),
                self.model.action_mean.detach(),
                self.model.action_std.detach(),
            )
        ti, si = self.teacher_env_idxs, self.student_env_idxs
        teacher_results = get_results(obs[ti], privileged_obs[ti],critic_obs[ti], history[ti], True)
        student_results = get_results(obs[si], privileged_obs[si],critic_obs[si], history[si], False)
        results = []
        for x1, x2 in zip(teacher_results, student_results):
            results.append(torch.cat([x1, x2], dim=0))
        # Compute the actions and values
        self.transition.actions = results[0] #3 1重整后的action并非原生action
        self.transition.values = results[1]
        self.transition.actions_log_prob = results[2]
        self.transition.action_mean = results[3]
        self.transition.action_sigma = results[4]
        # need to record obs and critic_obs before env.step()
        self.transition.history = torch.cat([history[ti], history[si]], dim=0)
        self.transition.observations = torch.cat([obs[ti], obs[si]], dim=0)
        self.transition.privileged_observations = torch.cat([privileged_obs[ti], privileged_obs[si]], dim=0)
        self.transition.critic_observations = torch.cat([critic_obs[ti], critic_obs[si]], dim=0)
        real_actions = torch.zeros_like(self.transition.actions)

        real_actions[ti] = self.transition.actions[:self.teacher_num_envs]
        real_actions[si] = self.transition.actions[self.teacher_num_envs:]
        return real_actions #两个action之间的区别
    
    def process_env_step(self, rewards, dones, infos):
        ti, si = self.teacher_env_idxs, self.student_env_idxs
        rewards = rewards.clone()
        self.transition.rewards = torch.cat([rewards[ti], rewards[si]], dim=0)
        self.transition.dones = torch.cat([dones[ti], dones[si]], dim=0)
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            time_outs = torch.cat([infos['time_outs'][ti], infos['time_outs'][si]], dim=0)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs.unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.model.reset(dones)
    
    def compute_returns(self, last_privileged_obs,last_critic_obs, last_history):
        ti, si = self.teacher_env_idxs, self.student_env_idxs
        last_values = torch.cat([
            self.model.evaluate(last_privileged_obs[ti],last_critic_obs[ti], last_history[ti], True).detach(),
            self.model.evaluate(last_privileged_obs[si],last_critic_obs[si], last_history[si], False).detach(),
        ], dim=0)
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_latent_loss = 0
        data = list(self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs))
        teacher_samples = self.teacher_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        student_samples = self.student_num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        for sample in data:
            (
                obs_batch, privileged_obs_batch,critic_obs_batch, actions_batch, history_batch,
                target_values_batch, advantages_batch, returns_batch,
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                hid_states_batch, masks_batch
            ) = sample
            def get_results(start, end, is_teacher):
                self.model.act(obs_batch[start:end], privileged_obs_batch[start:end], history_batch[start:end], is_teacher)
                actions_log_prob = self.model.get_actions_log_prob(actions_batch[start:end])
                value = self.model.evaluate(privileged_obs_batch[start:end], critic_obs_batch[start:end],history_batch[start:end], is_teacher)
                mu = self.model.action_mean
                sigma = self.model.action_std
                entropy = self.model.entropy
                return actions_log_prob, value, mu, sigma, entropy
            teacher_results = get_results(0, teacher_samples, True)
            student_results = get_results(teacher_samples, teacher_samples + student_samples, False)
            results = []
            for x1, x2 in zip(teacher_results, student_results):
                results.append(torch.cat([x1, x2], dim=0))
            actions_log_prob_batch = results[0]
            value_batch = results[1]
            mu_batch = results[2]
            sigma_batch = results[3]
            entropy_batch = results[4]
            # 对称损失：teacher 段镜像前向（obs/priv 镜像 -> act -> mu 映射回原动作空间）
            sym_loss_val = torch.tensor(0., device=self.device)
            if self.sym_loss:
                m_obs = obs_batch[:teacher_samples] @ self.sym_obs_P
                m_priv = privileged_obs_batch[:teacher_samples] @ self.sym_priv_P
                self.model.act(m_obs, m_priv, history_batch[:teacher_samples], True)
                sym_loss_val = (mu_batch[:teacher_samples] - self.model.action_mean @ self.sym_act_P).pow(2).mean()


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
                    
                    for param_group in self.optimizer1.param_groups:
                        param_group['lr'] = self.learning_rate


            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                            1.0 + self.clip_param)
            surrogate_losses = torch.max(surrogate, surrogate_clipped)
            teacher_surrogate_loss = surrogate_losses[:teacher_samples].mean()
            student_surrogate_loss = surrogate_losses[teacher_samples:].mean()
            surrogate_loss = teacher_surrogate_loss + student_surrogate_loss
            # surrogate_loss = surrogate_losses.mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            # teacher_value_loss = value_losses[:teacher_samples].mean()
            # student_value_loss = value_losses[teacher_samples:].mean()
            # value_loss = teacher_value_loss  # + student_value_loss

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.sym_coef * sym_loss_val

            # Gradient step
            self.optimizer1.zero_grad()
            loss.backward()
            params_to_clip = itertools.chain.from_iterable(g['params'] for g in self.optimizer1.param_groups)
            nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)
            self.optimizer1.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()
        
        for sample in data:
            (
                obs_batch, privileged_obs_batch,critic_obs_batch, actions_batch, history_batch,
                target_values_batch, advantages_batch, returns_batch,
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
                hid_states_batch, masks_batch
            ) = sample
            # Student encoder update
            student_latent = self.model.student_encoder(history_batch[teacher_samples:])
            with torch.no_grad():
                teacher_latent = self.model.teacher_encoder(privileged_obs_batch[teacher_samples:])
            latent_loss = (teacher_latent - student_latent).pow(2).mean()

            self.optimizer2.zero_grad()
            latent_loss.backward()
            nn.utils.clip_grad_norm_(self.model.student_encoder.parameters(), self.max_grad_norm)
            self.optimizer2.step()

            mean_latent_loss += latent_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_latent_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_entropy_loss, mean_latent_loss