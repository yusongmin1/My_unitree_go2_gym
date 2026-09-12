# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# DreamWaQ-main CENet+PPO with AMP discriminator (this repo).

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticDreamWaQ
from rsl_rl.storage import RolloutStorageDreamWaQ
from rsl_rl.storage.replay_buffer import ReplayBuffer


class PPO_DreamWaQ_AMP:
    actor_critic: ActorCriticDreamWaQ

    def __init__(
        self,
        actor_critic,
        discriminator,
        amp_data,
        amp_normalizer,
        min_std=None,
        amp_replay_buffer_size=100000,
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
        device="cpu",
        vae_learning_rate=None,  # ignored: joint optimizer
        vae_kl_weight=1.0,
        beta=None,
        num_obs=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "PPO_DreamWaQ_AMP.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.beta = vae_kl_weight if beta is None else beta
        self.min_std = min_std

        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorageDreamWaQ.Transition()
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device
        )
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.num_obs = num_obs if num_obs is not None else self.actor_critic.num_obs
        self.storage = None

        # Joint CENet+actor+critic + AMP discriminator (DreamWaQ-main style for policy)
        amp_trunk = {
            "params": self.discriminator.trunk.parameters(),
            "weight_decay": 1e-4,
            "name": "amp_trunk",
        }
        amp_head = {
            "params": self.discriminator.amp_linear.parameters(),
            "weight_decay": 1e-2,
            "name": "amp_head",
        }
        self.optimizer = optim.Adam(
            [
                {"params": self.actor_critic.parameters(), "name": "actor_critic"},
                amp_trunk,
                amp_head,
            ],
            lr=learning_rate,
        )
        self.transition = RolloutStorageDreamWaQ.Transition()

        self.sym_loss = sym_loss
        self.sym_coef = sym_coef
        if self.sym_loss:
            from rsl_rl.utils.utils import build_sym_perm_matrix

            self.sym_obs_P = build_sym_perm_matrix(obs_permutation, stack=1, device=self.device)
            self.sym_act_P = build_sym_perm_matrix(act_permutation, stack=1, device=self.device)
            fs = frame_stack if frame_stack else 1
            self.sym_hist_P = build_sym_perm_matrix(obs_permutation, stack=fs, device=self.device)

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_hist_shape,
        action_shape,
    ):
        self.storage = RolloutStorageDreamWaQ(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_hist_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, prev_critic_obs, obs_history, amp_obs):
        self.transition.actions = self.actor_critic.act(obs, obs_history).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.observation_history = obs_history
        self.transition.critic_observations = critic_obs
        self.transition.prev_critic_obs = prev_critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, amp_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, beta=None):
        if beta is None:
            beta = self.beta

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy_loss = 0
        mean_vae_loss = 0
        mean_vel_estimation_loss = 0
        mean_recon_loss = 0
        mean_kl_loss = 0
        mean_amp_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches,
        )

        for (
            obs_batch,
            critic_obs_batch,
            prev_critic_obs_batch,
            obs_hist_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            live_batch,
            hid_states_batch,
            masks_batch,
        ), sample_amp_policy, sample_amp_expert in zip(
            generator, amp_policy_generator, amp_expert_generator
        ):
            self.actor_critic.act(
                obs_batch, obs_hist_batch, masks=masks_batch, hidden_states=hid_states_batch[0]
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            mean_entropy_loss += entropy_batch.mean().item()

            sym_loss_val = torch.tensor(0.0, device=self.device)
            if self.sym_loss:
                m_obs = obs_batch @ self.sym_obs_P
                m_hist = obs_hist_batch @ self.sym_hist_P
                code_m, _, _, _, _, _, _ = self.actor_critic.cenet_forward(m_hist)
                mu_m = self.actor_critic.actor(torch.cat((code_m, m_obs), dim=-1))
                sym_loss_val = (mu_batch - mu_m @ self.sym_act_P).pow(2).mean()

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # CENet loss (DreamWaQ-main)
            code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent = (
                self.actor_critic.cenet_forward(obs_hist_batch)
            )
            vel_target = prev_critic_obs_batch[:, self.num_obs : self.num_obs + 3].detach()
            decode_target = obs_batch.detach()
            live = live_batch.view(-1, 1)
            vel_estimation_loss = nn.MSELoss()(code_vel * live, vel_target * live)
            recon_loss = nn.MSELoss()(decode * live, decode_target * live)
            kl_loss = beta * (
                -0.5
                * torch.mean(
                    torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp(), dim=-1)
                    * live.squeeze(-1)
                )
            )
            autoenc_loss = vel_estimation_loss + recon_loss + kl_loss

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            policy_state_unnorm = torch.clone(policy_state)
            expert_state_unnorm = torch.clone(expert_state)
            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(
                        policy_next_state, self.device
                    )
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(
                        expert_next_state, self.device
                    )
            policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
            expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
            expert_loss = torch.nn.MSELoss()(
                expert_d, torch.ones(expert_d.size(), device=self.device)
            )
            policy_loss = torch.nn.MSELoss()(
                policy_d, -1 * torch.ones(policy_d.size(), device=self.device)
            )
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(
                expert_state, expert_next_state, lambda_=5
            )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + autoenc_loss
                + self.sym_coef * sym_loss_val
                + amp_loss
                + grad_pen_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)
            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state_unnorm.cpu().numpy())
                self.amp_normalizer.update(expert_state_unnorm.cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_vae_loss += autoenc_loss.item()
            mean_vel_estimation_loss += vel_estimation_loss.item()
            mean_recon_loss += recon_loss.item()
            mean_kl_loss += kl_loss.item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_vae_loss /= num_updates
        mean_vel_estimation_loss /= num_updates
        mean_recon_loss /= num_updates
        mean_kl_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_vae_loss,
            mean_vel_estimation_loss,
            mean_recon_loss,
            mean_kl_loss,
            mean_amp_loss,
            mean_grad_pen_loss,
            mean_policy_pred,
            mean_expert_pred,
            mean_entropy_loss,
        )
