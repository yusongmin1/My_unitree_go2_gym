# PPO with PIE auxiliary estimator losses.

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCriticPIE
from rsl_rl.storage import RolloutStoragePIE


class PPO_PIE:
    actor_critic: ActorCriticPIE

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
        auxiliary_loss_coef=1.0,
        velocity_loss_coef=1.0,
        foot_clearance_loss_coef=1.0,
        height_reconstruction_loss_coef=1.0,
        successor_loss_coef=1.0,
        kl_loss_coef=4.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "PPO_PIE.__init__ ignored unexpected args: "
                + str(list(kwargs.keys()))
            )

        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStoragePIE.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.auxiliary_loss_coef = auxiliary_loss_coef
        self.auxiliary_coefficients = {
            "velocity": velocity_loss_coef,
            "foot_clearance": foot_clearance_loss_coef,
            "height_reconstruction": height_reconstruction_loss_coef,
            "successor": successor_loss_coef,
            "kl": kl_loss_coef,
        }

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_hist_shape,
        depth_shape,
        velocity_shape,
        height_shape,
        foot_shape,
        successor_shape,
        action_shape,
    ):
        self.storage = RolloutStoragePIE(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_hist_shape,
            depth_shape,
            velocity_shape,
            height_shape,
            foot_shape,
            successor_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(
        self,
        obs,
        critic_obs,
        obs_history,
        depth,
        velocity_target,
        height_target,
        foot_target,
    ):
        self.transition.hidden_states = self.actor_critic.get_hidden_states()
        self.transition.actions = self.actor_critic.act(
            obs, obs_history, depth
        ).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.observation_history = obs_history
        self.transition.depth = depth
        self.transition.velocity_target = velocity_target
        self.transition.height_target = height_target
        self.transition.foot_target = foot_target
        # Placeholder; overwritten in process_env_step with next clean proprio.
        self.transition.successor_target = obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, successor_target):
        self.transition.successor_target = successor_target
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy_loss = 0.0
        mean_auxiliary = {name: 0.0 for name in self.auxiliary_coefficients}

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (
            obs_batch,
            critic_obs_batch,
            obs_hist_batch,
            depth_batch,
            velocity_target_batch,
            height_target_batch,
            foot_target_batch,
            successor_target_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            live_batch,
            hid_batch,
        ) in generator:
            self.actor_critic.act(
                obs_batch,
                obs_hist_batch,
                depth_batch,
                hidden_states=hid_batch,
            )
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            mean_entropy_loss += entropy_batch.mean().item()

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (value_batch - returns_batch).pow(2),
                    (value_clipped - returns_batch).pow(2),
                ).mean()
            else:
                value_loss = (value_batch - returns_batch).pow(2).mean()

            auxiliary_losses = self.actor_critic.auxiliary_losses(
                obs_batch,
                obs_hist_batch,
                depth_batch,
                velocity_target_batch,
                height_target_batch,
                foot_target_batch,
                successor_target_batch,
                hidden_states=hid_batch,
                live_mask=live_batch,
            )
            weighted_auxiliary = sum(
                self.auxiliary_coefficients[name] * auxiliary_losses[name]
                for name in self.auxiliary_coefficients
            )

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.auxiliary_loss_coef * weighted_auxiliary
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            for name, value in auxiliary_losses.items():
                mean_auxiliary[name] += value.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_auxiliary = {
            name: value / num_updates for name, value in mean_auxiliary.items()
        }
        self.storage.clear()
        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_entropy_loss,
            mean_auxiliary,
        )
