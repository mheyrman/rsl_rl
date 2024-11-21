#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

import rsl_rl.modules.utility_modules as utils

class ImitationAgent(nn.Module):
    def __init__(
        self,
        num_obs,
        num_actions,
        hidden_dims,
        activation,
        num_state_obs=45,                   # number of observations from sim, not reference motions
        encoder_hidden_dims=[32, 24, 64],
    ):
        super().__init__()

        self.num_reference_obs = num_obs - num_state_obs
        activation = get_activation(activation)

        # num_obs = 85, num_reference_obs = 40 encode into 16
        mlp_input_dim = num_state_obs + encoder_hidden_dims[2]


        # self.obs_enc = nn.Sequential(
        #     nn.Linear(self.num_reference_obs, encoder_hidden_dims[0]),
        #     nn.ReLU(),
        #     nn.Linear(encoder_hidden_dims[0], encoder_hidden_dims[1]),
        #     nn.ReLU(),
        #     nn.Linear(encoder_hidden_dims[1], encoder_hidden_dims[2]),
        # )
        self.obs_enc = utils.VAEBlock(self.num_reference_obs, encoder_hidden_dims[2])
        # self.obs_enc = utils.GRUBlock(self.num_reference_obs, encoder_hidden_dims[2], 128)
        # self.obs_enc = utils.TransformerEncoder(self.num_reference_obs, encoder_hidden_dims[2])
        # self.obs_enc = utils.PAE(self.num_reference_obs, encoder_hidden_dims[2])

        # original
        layers = []
        layers.append(nn.Linear(mlp_input_dim, hidden_dims[0]))
        layers.append(activation)
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_index], num_actions))
            else:
                layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                layers.append(activation)
        self.policy = nn.Sequential(*layers)

        # self.policy = VQVAEBlock(mlp_input_dim, num_actions)
        # self.policy = VAEBlock(mlp_input_dim, num_actions)

        # # attempt at transformer decoder
        # self.input_embedding = nn.Sequential(
        #     nn.Linear(mlp_input_dim, 32),
        #     nn.Dropout(0.1)
        # )
        # self.pos_encoding = nn.Parameter(torch.zeros(32))
        # self.decoder_blocks = nn.Sequential(
        #     TransformerBlock(32, 2, 0.1),
        #     TransformerBlock(32, 2, 0.1),
        # )
        # self.output_layer = nn.Linear(32, num_actions)


    def forward(self, x):
        # split reference obs and obs, pass reference obs through encoder, concatenate with obs and pass through policy
        ref_obs, obs = torch.split(x, [self.num_reference_obs, x.size(1) - self.num_reference_obs], dim=1)
        ref_obs = self.obs_enc(ref_obs)
        x = torch.cat([ref_obs, obs], dim=1)

        # x = self.input_embedding(x)
        # x = x + self.pos_encoding
        # x = self.decoder_blocks(x)
        # return self.output_layer(x)
        return self.policy(x)

class ActorCriticImitation(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.actor = ImitationAgent(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.critic = ImitationAgent(num_critic_obs, 1, critic_hidden_dims, activation)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

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
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

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

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
