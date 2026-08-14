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
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class RolloutStorageCTS:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.history = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, teacher_num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape,critic_obs_shape, history_shape,actions_shape, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.critic_obs_shape = critic_obs_shape
        self.actions_shape = actions_shape
        self.teacher_num_envs = teacher_num_envs
        self.student_num_envs = num_envs - teacher_num_envs

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)#线速度地形，误差补偿
        self.critic_observations = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)#critic
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.history = torch.zeros(num_transitions_per_env, num_envs,* history_shape, device=self.device)

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        self.observations[self.step].copy_(transition.observations)
        self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.critic_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.history[self.step].copy_(transition.history)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])


    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8): # 8192 24 45 obs 其中 6144 是 teacher 2048 是 student
        teacher_samples_num = self.teacher_num_envs * self.num_transitions_per_env
        student_samples_num = self.student_num_envs * self.num_transitions_per_env
        teacher_mini_batch_size = teacher_samples_num // num_mini_batches
        student_mini_batch_size = student_samples_num // num_mini_batches
        teacher_indices = torch.randperm(teacher_samples_num, requires_grad=False, device=self.device)
        student_indices = teacher_samples_num + torch.randperm(student_samples_num, requires_grad=False, device=self.device)

        obs_dims = list(range(2, self.observations.dim()))
        observations = self.observations.permute(1, 0, *obs_dims).flatten(0, 1)
        if self.privileged_observations is not None:
            critic_dims = list(range(2, self.critic_observations.dim()))
            critic_observations = self.critic_observations.permute(1, 0, *critic_dims).flatten(0, 1)
            privileged_observations = self.privileged_observations.permute(1, 0, *critic_dims).flatten(0, 1)
        else:
            critic_observations = observations

        action_dims = list(range(2, self.actions.dim()))
        actions = self.actions.permute(1, 0, *action_dims).flatten(0, 1)
        old_mu = self.mu.permute(1, 0, *action_dims).flatten(0, 1)
        old_sigma = self.sigma.permute(1, 0, *action_dims).flatten(0, 1)
        hist_dims = list(range(2, self.history.dim()))
        history = self.history.permute(1, 0, *hist_dims).flatten(0, 1)
        values = self.values.permute(1, 0, 2).flatten(0, 1)
        returns = self.returns.permute(1, 0, 2).flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.permute(1, 0, 2).flatten(0, 1)
        advantages = self.advantages.permute(1, 0, 2).flatten(0, 1)

        def get_teacher_student_samples(data, slice):
            (i1, i2), (j1, j2) = slice
            return torch.cat([data[teacher_indices[i1:i2]], data[student_indices[j1:j2]]], 0).detach()

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                slice = (
                    (i * teacher_mini_batch_size, (i+1) * teacher_mini_batch_size),
                    (i * student_mini_batch_size, (i+1) * student_mini_batch_size),
                )

                obs_batch = get_teacher_student_samples(observations, slice) #这里面的顺序问题
                critic_observations_batch = get_teacher_student_samples(critic_observations, slice)
                privileged_observations_batch = get_teacher_student_samples(privileged_observations, slice)
                actions_batch = get_teacher_student_samples(actions, slice)
                target_values_batch = get_teacher_student_samples(values, slice)
                returns_batch = get_teacher_student_samples(returns, slice)
                old_actions_log_prob_batch = get_teacher_student_samples(old_actions_log_prob, slice)
                advantages_batch = get_teacher_student_samples(advantages, slice)
                old_mu_batch = get_teacher_student_samples(old_mu, slice)
                old_sigma_batch = get_teacher_student_samples(old_sigma, slice)
                history_batch = get_teacher_student_samples(history, slice)
                yield obs_batch, privileged_observations_batch,critic_observations_batch, actions_batch, history_batch, target_values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (None, None), None
