# -*- coding: utf-8 -*-
'''
@File    : actor_critic_cts.py
@Time    : 2025/12/30 21:06:08
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : Concurrent Teacher Student Network
@Refer   : CTS https://arxiv.org/abs/2405.10830
'''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCriticCTS(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_privileged_obs,
                        num_critic_obs,
                        num_actions,
                        history_length,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        teacher_encoder_hidden_dims=[512, 256],
                        student_encoder_hidden_dims=[512, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        latent_dim=32,
                        norm_type='l2norm',
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        assert norm_type in ['l2norm', 'simnorm'], f"Normalization type {norm_type} not supported!"
        super(ActorCriticCTS, self).__init__()
        self.num_actions = num_actions

        activation = get_activation(activation)

        mlp_input_dim_t = num_privileged_obs
        mlp_input_dim_s = num_actor_obs * history_length
        mlp_input_dim_a = latent_dim + num_actor_obs
        mlp_input_dim_c = latent_dim + num_critic_obs


        # Teacher encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(mlp_input_dim_t, teacher_encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(teacher_encoder_hidden_dims)):
            if l == len(teacher_encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(teacher_encoder_hidden_dims[l], latent_dim))
                if norm_type == 'l2norm':
                    encoder_layers.append(L2Norm())
                elif norm_type == 'simnorm':
                    encoder_layers.append(SimNorm())
            else:
                encoder_layers.append(nn.Linear(teacher_encoder_hidden_dims[l], teacher_encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.teacher_encoder = nn.Sequential(*encoder_layers)

        # Student encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(mlp_input_dim_s, student_encoder_hidden_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(student_encoder_hidden_dims)):
            if l == len(student_encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(student_encoder_hidden_dims[l], latent_dim))
                if norm_type == 'l2norm':
                    encoder_layers.append(L2Norm())
                elif norm_type == 'simnorm':
                    encoder_layers.append(SimNorm())
            else:
                encoder_layers.append(nn.Linear(student_encoder_hidden_dims[l], student_encoder_hidden_dims[l + 1]))
                encoder_layers.append(activation)
        self.student_encoder = nn.Sequential(*encoder_layers)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Teacher Encoder MLP: {self.teacher_encoder}")
        print(f"Student Encoder MLP: {self.student_encoder}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, latent_and_obs):
        mean = self.actor(latent_and_obs)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, obs, privileged_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent = self.student_encoder(history).detach()
        x = torch.cat([latent, obs], dim=1)
        self.update_distribution(x)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs,history):
        latent = self.student_encoder(history)
        x = torch.cat([latent, obs], dim=1)
        actions_mean = self.actor(x)
        return actions_mean

    def evaluate(self, privileged_obs, critic_obs, history, is_teacher, **kwargs):
        if is_teacher:
            latent = self.teacher_encoder(privileged_obs)
        else:
            latent = self.student_encoder(history)
        x = torch.cat([latent.detach(), critic_obs], dim=1)
        value = self.critic(x)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
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

class L2Norm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=-1)

class SimNorm(nn.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self):
        super().__init__()
        self.dim = 8  # for latent dim 512

    def forward(self, x):
        shp = x.shape
        target_shape = list(shp[:-1]) + [-1, self.dim]
        x = x.view(target_shape)
        x = F.softmax(x, dim=-1)
        return x.view(shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"
