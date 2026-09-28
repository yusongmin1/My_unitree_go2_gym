# PIE actor-critic ported from parkour_mjlab, adapted to this repo's flat-tensor API.

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name: str):
    if act_name == "elu":
        return nn.ELU()
    if act_name == "selu":
        return nn.SELU()
    if act_name == "relu":
        return nn.ReLU()
    if act_name == "lrelu":
        return nn.LeakyReLU()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    if act_name == "gelu":
        return nn.GELU()
    raise ValueError(f"invalid activation: {act_name}")


def _mlp(in_dim, out_dim, hidden_dims, activation: nn.Module) -> nn.Sequential:
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(last, h), activation]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class ActorCriticPIE(nn.Module):
    """PIE estimator + actor MLP + critic MLP.

    Actor path (mjlab PIE):
      history MLP + depth CNN -> transformer fuse -> GRU memory
      -> (vel, map_latent, foot, vae_z) latent concat with proprio -> actor
    Aux heads: successor decoder, height decoder; losses computed in PPO_PIE.
    """

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        history_dim,
        depth_shape=(2, 60, 86),
        velocity_dim=3,
        height_map_dim=198,
        foot_dim=4,
        token_dim=64,
        map_latent_dim=16,
        vae_latent_dim=16,
        memory_hidden_dim=128,
        memory_num_layers=1,
        history_hidden_dims=(256, 128),
        successor_decoder_dims=(64, 128),
        height_decoder_dims=(64, 128),
        output_channels=(32, 64, 64),
        kernel_size=(8, 4, 3),
        stride=(4, 2, 1),
        padding=0,
        attention_heads=1,
        transformer_layers=2,
        transformer_ff_dim=256,
        transformer_dropout=0.0,
        cnn_activation="elu",
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticPIE.__init__ ignored unexpected args: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        if actor_hidden_dims is None:
            actor_hidden_dims = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]

        self.proprio_dim = int(num_actor_obs)
        self.history_dim = int(history_dim)
        self.velocity_dim = int(velocity_dim)
        self.height_map_dim = int(height_map_dim)
        self.foot_dim = int(foot_dim)
        self.successor_dim = self.proprio_dim

        self.depth_shape = tuple(int(v) for v in depth_shape)
        self.depth_channels, self.depth_height, self.depth_width = self.depth_shape
        self.depth_flat_dim = (
            self.depth_channels * self.depth_height * self.depth_width
        )

        self.token_dim = int(token_dim)
        self.map_latent_dim = int(map_latent_dim)
        self.vae_latent_dim = int(vae_latent_dim)
        self.memory_hidden_dim = int(memory_hidden_dim)
        self.memory_num_layers = int(memory_num_layers)

        act = get_activation(activation)
        cnn_act_name = cnn_activation

        self.history_encoder = _mlp(
            self.history_dim, self.token_dim, list(history_hidden_dims), act
        )
        self.depth_encoder = self._make_depth_encoder(
            list(output_channels),
            list(kernel_size),
            list(stride),
            padding,
            cnn_act_name,
        )
        self.depth_token_norm = nn.LayerNorm(self.token_dim)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(attention_heads),
            dim_feedforward=int(transformer_ff_dim),
            dropout=float(transformer_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.cross_modal_transformer = nn.TransformerEncoder(
            transformer_layer, num_layers=int(transformer_layers)
        )
        self.memory = nn.GRU(
            input_size=2 * self.token_dim,
            hidden_size=self.memory_hidden_dim,
            num_layers=self.memory_num_layers,
        )

        self.velocity_head = nn.Linear(self.memory_hidden_dim, self.velocity_dim)
        self.map_latent_head = nn.Linear(self.memory_hidden_dim, self.map_latent_dim)
        self.foot_clearance_head = nn.Linear(self.memory_hidden_dim, self.foot_dim)
        self.vae_mu_head = nn.Linear(self.memory_hidden_dim, self.vae_latent_dim)
        self.vae_logvar_head = nn.Linear(self.memory_hidden_dim, self.vae_latent_dim)

        estimator_output_dim = (
            self.velocity_dim
            + self.map_latent_dim
            + self.foot_dim
            + self.vae_latent_dim
        )
        self.actor_input_dim = self.proprio_dim + estimator_output_dim
        self.successor_decoder = _mlp(
            estimator_output_dim,
            self.successor_dim,
            list(successor_decoder_dims),
            act,
        )
        self.height_decoder = _mlp(
            self.map_latent_dim,
            self.height_map_dim,
            list(height_decoder_dims),
            act,
        )

        self.actor = _mlp(
            self.actor_input_dim, num_actions, list(actor_hidden_dims), act
        )
        self.critic = _mlp(
            int(num_critic_obs), 1, list(critic_hidden_dims), act
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        self.memory_hidden: torch.Tensor | None = None
        # Cache last estimate tensors for aux loss (set during act/update)
        self._last_estimates: dict | None = None

        print(
            f"PIE actor_in={self.actor_input_dim}, critic_in={num_critic_obs}, "
            f"depth={self.depth_shape}, hist={self.history_dim}, "
            f"memory={self.memory_hidden_dim}"
        )

    def _make_depth_encoder(self, channels, kernels, strides, padding, act_name):
        channels = list(channels)
        channels[-1] = self.token_dim
        layers: list[nn.Module] = []
        in_ch = self.depth_channels
        for i, out_ch in enumerate(channels):
            k = kernels[min(i, len(kernels) - 1)]
            s = strides[min(i, len(strides) - 1)]
            if isinstance(padding, (tuple, list)):
                pad = int(padding[min(i, len(padding) - 1)])
            else:
                pad = int(padding)
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, k, stride=s, padding=pad),
                    get_activation(act_name),
                ]
            )
            in_ch = out_ch
        return nn.Sequential(*layers)

    def _encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        leading = depth.shape[:-1]
        image = depth.reshape(-1, *self.depth_shape) - 0.5
        feature = self.depth_encoder(image)
        tokens = feature.flatten(start_dim=2).transpose(1, 2)
        tokens = self.depth_token_norm(tokens)
        return tokens.reshape(*leading, tokens.shape[-2], self.token_dim)

    def _estimate_tensors(
        self,
        proprio: torch.Tensor,
        history: torch.Tensor,
        depth: torch.Tensor,
        *,
        hidden_state: torch.Tensor | None,
        update_internal: bool,
        sample_latent: bool,
    ):
        history_token = self.history_encoder(history)
        depth_tokens = self._encode_depth(depth)
        leading = history_token.shape[:-1]
        tokens = torch.cat([history_token.unsqueeze(-2), depth_tokens], dim=-2)
        token_count = tokens.shape[-2]
        fused = self.cross_modal_transformer(
            tokens.reshape(-1, token_count, self.token_dim)
        )
        fused = fused.reshape(*leading, token_count, self.token_dim)
        memory_input = torch.cat(
            [fused[..., 0, :], fused[..., 1:, :].mean(dim=-2)], dim=-1
        )

        # Truncated BPTT length-1 when hidden_state is provided (training mini-batch).
        # Rollout path: update_internal=True updates self.memory_hidden.
        h0 = hidden_state if hidden_state is not None else self.memory_hidden
        memory_output, new_hidden = self.memory(memory_input.unsqueeze(0), h0)
        memory_output = memory_output.squeeze(0)
        if update_internal:
            self.memory_hidden = new_hidden

        velocity = self.velocity_head(memory_output)
        map_latent = self.map_latent_head(memory_output)
        foot_clearance = self.foot_clearance_head(memory_output)
        mu = self.vae_mu_head(memory_output)
        logvar = self.vae_logvar_head(memory_output).clamp(-10.0, 5.0)
        if sample_latent:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        latent = torch.cat([velocity, map_latent, foot_clearance, z], dim=-1)
        estimates = {
            "velocity": velocity,
            "map_latent": map_latent,
            "foot_clearance": foot_clearance,
            "mu": mu,
            "logvar": logvar,
            "latent": latent,
        }
        return estimates, new_hidden

    def reset(self, dones=None):
        if self.memory_hidden is None:
            return
        if dones is None:
            self.memory_hidden = None
            return
        self.memory_hidden[:, dones.bool(), :] = 0.0

    def get_hidden_states(self):
        # Critic is feed-forward; only actor GRU state is tracked.
        return self.memory_hidden, None

    def init_hidden(self, num_envs: int, device):
        self.memory_hidden = torch.zeros(
            self.memory_num_layers,
            num_envs,
            self.memory_hidden_dim,
            device=device,
        )

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_input: torch.Tensor):
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, obs_history, depth, hidden_states=None, **kwargs):
        estimates, _ = self._estimate_tensors(
            observations,
            obs_history,
            depth,
            hidden_state=hidden_states,
            update_internal=hidden_states is None,
            sample_latent=self.training,
        )
        self._last_estimates = estimates
        actor_input = torch.cat([observations, estimates["latent"]], dim=-1)
        self.update_distribution(actor_input)
        return self.distribution.sample()

    def act_inference(self, observations, obs_history, depth, hidden_states=None):
        estimates, new_hidden = self._estimate_tensors(
            observations,
            obs_history,
            depth,
            hidden_state=hidden_states,
            update_internal=hidden_states is None,
            sample_latent=False,
        )
        actor_input = torch.cat([observations, estimates["latent"]], dim=-1)
        actions = self.actor(actor_input)
        return actions, new_hidden

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def auxiliary_losses(
        self,
        observations,
        obs_history,
        depth,
        velocity_target,
        height_target,
        foot_target,
        successor_target,
        hidden_states=None,
        live_mask=None,
    ):
        estimates, _ = self._estimate_tensors(
            observations,
            obs_history,
            depth,
            hidden_state=hidden_states,
            update_internal=False,
            sample_latent=True,
        )
        successor_hat = self.successor_decoder(estimates["latent"])
        height_hat = self.height_decoder(estimates["map_latent"])
        mu = estimates["mu"]
        logvar = estimates["logvar"]
        kl_per = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp(), dim=-1)

        def _mse(pred, target):
            err = (pred - target).square().mean(dim=-1)
            if live_mask is None:
                return err.mean()
            live = live_mask.view(-1)
            return (err * live).sum() / live.sum().clamp_min(1.0)

        def _mean(values):
            if live_mask is None:
                return values.mean()
            live = live_mask.view(-1)
            return (values * live).sum() / live.sum().clamp_min(1.0)

        return {
            "velocity": _mse(estimates["velocity"], velocity_target),
            "foot_clearance": _mse(estimates["foot_clearance"], foot_target),
            "height_reconstruction": _mse(height_hat, height_target),
            "successor": _mse(successor_hat, successor_target),
            "kl": _mean(kl_per),
        }
