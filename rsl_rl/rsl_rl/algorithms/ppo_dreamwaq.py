# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Ported from DreamWaQ-main PPO: joint Actor-Critic + CENet training.
# Fixes vs upstream:
# - KL term uses mean over batch (sum explodes with large mini-batches)
# - live_batch mask for terminal transitions
# - optional symmetry loss retained for this repo

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticDreamWaQ
from rsl_rl.storage import RolloutStorageDreamWaQ


class PPO_DreamWaQ:
    actor_critic: ActorCriticDreamWaQ

    def __init__(
        self,
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
        device="cpu",
        vae_kl_weight=1.0,
        beta=None,
        sym_loss=False,
        sym_coef=1.0,
        frame_stack=None,
        obs_permutation=None,
        act_permutation=None,
        privileged_permutation=None,
        terrain_permutation=None,
        vae_learning_rate=None,  # ignored: joint optimizer (DreamWaQ-main style)
        num_obs=None,  # optional override; else from actor_critic.num_obs
        **kwargs,
    ):
        if kwargs:
            print(
                "PPO_DreamWaQ.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        # DreamWaQ-main uses `beta`; this repo historically used vae_kl_weight
        self.beta = vae_kl_weight if beta is None else beta

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.num_obs = num_obs if num_obs is not None else self.actor_critic.num_obs
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
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

    def act(self, obs, critic_obs, prev_critic_obs, obs_history):
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
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
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
        mean_autoenc_loss = 0
        mean_vel_loss = 0
        mean_decode_loss = 0
        mean_kl_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
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
        ) in generator:
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

            # CENet / VAE loss (DreamWaQ-main): reconstruct current obs, estimate lin_vel from prev privileged
            code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent = (
                self.actor_critic.cenet_forward(obs_hist_batch)
            )
            # privileged layout: [obs(num_obs), lin_vel(3), ...]
            vel_target = prev_critic_obs_batch[:, self.num_obs : self.num_obs + 3]
            decode_target = obs_batch
            vel_target = vel_target.detach()
            decode_target = decode_target.detach()

            # live_batch: (N,1) — mask terminals so reset frames don't poison CENet
            live = live_batch.view(-1, 1)
            vel_loss = nn.MSELoss()(code_vel * live, vel_target * live)
            decode_loss = nn.MSELoss()(decode * live, decode_target * live)
            # mean over batch (upstream sum explodes with large minibatches)
            kl_loss = beta * (
                -0.5
                * torch.mean(
                    torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp(), dim=-1)
                    * live.squeeze(-1)
                )
            )
            autoenc_loss = vel_loss + decode_loss + kl_loss

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

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + autoenc_loss
                + self.sym_coef * sym_loss_val
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_autoenc_loss += autoenc_loss.item()
            mean_vel_loss += vel_loss.item()
            mean_decode_loss += decode_loss.item()
            mean_kl_loss += kl_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_autoenc_loss /= num_updates
        mean_vel_loss /= num_updates
        mean_decode_loss /= num_updates
        mean_kl_loss /= num_updates
        self.storage.clear()

        # Keep return order compatible with existing runner logging names
        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_autoenc_loss,
            mean_vel_loss,
            mean_decode_loss,
            mean_kl_loss,
            mean_entropy_loss,
        )
