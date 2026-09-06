# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Ported from DreamWaQ-main ActorCritic_DWAQ, with inference/logvar fixes.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

# Clamp logvar only where exp() runs: wide lower bound keeps near-deterministic codes,
# tight upper bound prevents exp(logvar) overflow (main NaN source).
LOGVAR_MIN = -20.0
LOGVAR_MAX = 2.0


def clamp_logvar(logvar: torch.Tensor) -> torch.Tensor:
    return torch.clamp(logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        # PyTorch has no nn.CReLU; use ReLU (same as common rsl_rl forks)
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class ActorCriticDreamWaQ(nn.Module):
    """DreamWaQ CENet + Actor-Critic (integrated VAE, matching DreamWaQ-main)."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        cenet_in_dim,
        cenet_out_dim=19,
        activation="elu",
        init_noise_std=1.0,
        decode_dim=None,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticDreamWaQ.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.activation = get_activation(activation)
        # proprioceptive obs dim reconstructed by decoder
        self.num_obs = decode_dim if decode_dim is not None else (num_actor_obs - cenet_out_dim)
        self.cenet_out_dim = cenet_out_dim
        actor_input_dim = num_actor_obs
        critic_input_dim = num_critic_obs

        # Prefer explicit MLP dims from config when provided; else DreamWaQ-main defaults.
        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]

        actor_layers = [nn.Linear(actor_input_dim, actor_hidden_dims[0]), self.activation]
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(self.activation)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = [nn.Linear(critic_input_dim, critic_hidden_dims[0]), self.activation]
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(self.activation)
        self.critic = nn.Sequential(*critic_layers)

        # CENet / VAE
        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim, 128),
            self.activation,
            nn.Linear(128, 64),
            self.activation,
        )
        self.encode_mean_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_logvar_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_mean_vel = nn.Linear(64, 3)
        self.encode_logvar_vel = nn.Linear(64, 3)
        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim, 64),
            self.activation,
            nn.Linear(64, 128),
            self.activation,
            nn.Linear(128, self.num_obs),
        )

        # NOTE: do NOT set self.vae = self — that creates a Module cycle and
        # RecursionError in .to() / state_dict(). CENet lives on this module.

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"CENet encoder/decoder: in={cenet_in_dim}, code={cenet_out_dim}, decode={self.num_obs}")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def reparameterise(self, mean, logvar):
        logvar = clamp_logvar(logvar)
        std = torch.exp(0.5 * logvar)
        code = mean + std * torch.randn_like(std)
        return code

    def cenet_forward(self, obs_history):
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = clamp_logvar(self.encode_logvar_latent(distribution))
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = clamp_logvar(self.encode_logvar_vel(distribution))
        code_latent = self.reparameterise(mean_latent, logvar_latent)
        code_vel = self.reparameterise(mean_vel, logvar_vel)
        code = torch.cat((code_vel, code_latent), dim=-1)
        decode = self.decoder(code)
        return code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, obs_history, **kwargs):
        code, _, _, _, _, _, _ = self.cenet_forward(obs_history)
        observations = torch.cat((code, observations), dim=-1)
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, obs_history):
        # Deterministic: use mean of latent/vel (not reparameterized sample)
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        mean_vel = self.encode_mean_vel(distribution)
        code = torch.cat((mean_vel, mean_latent), dim=-1)
        observations = torch.cat((code, observations), dim=-1)
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)
