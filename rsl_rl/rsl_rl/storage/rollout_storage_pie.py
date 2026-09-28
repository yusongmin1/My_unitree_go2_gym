# Rollout storage for PIE: depth + aux targets + actor GRU hidden.

import torch


class RolloutStoragePIE:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.observation_history = None
            self.depth = None
            self.velocity_target = None
            self.height_target = None
            self.foot_target = None
            self.successor_target = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape,obs_hist_shape,depth_shape,
        velocity_shape,
        height_shape,
        foot_shape,
        successor_shape,
        actions_shape,
        device="cpu",
    ):
        self.device = device
        self.num_envs = num_envs

        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=device
        )
        self.privileged_observations = torch.zeros(
            num_transitions_per_env, num_envs, *privileged_obs_shape, device=device
        )
        self.observation_history = torch.zeros(
            num_transitions_per_env, num_envs, *obs_hist_shape, device=device
        )
        self.depth = torch.zeros(
            num_transitions_per_env, num_envs, *depth_shape, device=device
        )
        self.velocity_target = torch.zeros(
            num_transitions_per_env, num_envs, *velocity_shape, device=device
        )
        self.height_target = torch.zeros(
            num_transitions_per_env, num_envs, *height_shape, device=device
        )
        self.foot_target = torch.zeros(
            num_transitions_per_env, num_envs, *foot_shape, device=device
        )
        self.successor_target = torch.zeros(
            num_transitions_per_env, num_envs, *successor_shape, device=device
        )

        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        ).byte()
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device
        )
        self.num_transitions_per_env = num_transitions_per_env

        self.saved_hidden_states_a = None
        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.observation_history[self.step].copy_(transition.observation_history)
        self.depth[self.step].copy_(transition.depth)
        self.velocity_target[self.step].copy_(transition.velocity_target)
        self.height_target[self.step].copy_(transition.height_target)
        self.foot_target[self.step].copy_(transition.foot_target)
        self.successor_target[self.step].copy_(transition.successor_target)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None:
            return
        hid_a = hidden_states[0]
        if hid_a is None:
            return
        if not isinstance(hid_a, tuple):
            hid_a = (hid_a,)
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(
                    self.observations.shape[0], *hid_a[i].shape, device=self.device
                )
                for i in range(len(hid_a))
            ]
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])

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
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        observations = self.observations.flatten(0, 1)
        critic_observations = self.privileged_observations.flatten(0, 1)
        obs_history = self.observation_history.flatten(0, 1)
        depth = self.depth.flatten(0, 1)
        velocity_target = self.velocity_target.flatten(0, 1)
        height_target = self.height_target.flatten(0, 1)
        foot_target = self.foot_target.flatten(0, 1)
        successor_target = self.successor_target.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        dones = self.dones.flatten(0, 1)

        if self.saved_hidden_states_a is not None:
            # [T, layers, N, H] -> flatten T,N : [T*N, layers, H] then transpose for GRU
            hid = self.saved_hidden_states_a[0]  # [T, layers, N, H]
            hid_flat = hid.permute(0, 2, 1, 3).flatten(0, 1)  # [T*N, layers, H]
        else:
            hid_flat = None

        for _epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                if hid_flat is not None:
                    # GRU wants [layers, batch, H]
                    hid_batch = hid_flat[batch_idx].permute(1, 0, 2).contiguous()
                else:
                    hid_batch = None

                yield (
                    observations[batch_idx],
                    critic_observations[batch_idx],
                    obs_history[batch_idx],
                    depth[batch_idx],
                    velocity_target[batch_idx],
                    height_target[batch_idx],
                    foot_target[batch_idx],
                    successor_target[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    1.0 - dones[batch_idx].float(),
                    hid_batch,
                )
