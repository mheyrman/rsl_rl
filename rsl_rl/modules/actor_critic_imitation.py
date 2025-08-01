#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

import rsl_rl.modules.utility_modules as utils
from pytorch_wavelets import DWTForward

class ImitationAgent(nn.Module):
    def __init__(
        self,
        num_obs,
        num_actions,
        hidden_dims,
        activation,
        # num_state_obs=66,                   # number of observations from sim, not reference motions, 45 anymal, 69 h1
        encoder_hidden_dims=[32, 24, 16],
        state_obs_dim=45,
        reference_obs_dim=16,
        latent_channels=6,
        band_outputs=3,
        wavelet_type="db3",
        additional_dim=1,
    ):
        super().__init__()

        self.ref_obs_per_step = reference_obs_dim
        print(num_obs) # 791 -> 725 ref

        self.num_reference_obs = num_obs - state_obs_dim
        self.horizon = self.num_reference_obs // self.ref_obs_per_step
        self.latent_channels = latent_channels

        activation = get_activation(activation)

        # num_obs = 85, num_reference_obs = 40 encode into 16
        # mlp_input_dim = state_obs_dim + encoder_hidden_dims[-1] + 16
        # mlp_input_dim = state_obs_dim + encoder_hidden_dims[-1] + 4 * self.latent_channels + 16

        # check shape
        dwt_test = DWTForward(J=band_outputs, wave=wavelet_type, mode='zero')
        test_in = torch.randn(4096, 1, self.latent_channels, self.horizon)
        test_out_l, test_out_h = dwt_test(test_in)

        # dwt_size = test_out_l.squeeze(1).reshape(4096, -1).shape[1]
        # for i in range(len(test_out_h)):
        #     dwt_size += test_out_h[i].squeeze(1).reshape(4096, -1).shape[1]

        ##############

        dwt_size = 1
        for i in range(len(test_out_h)):
            for j in range(test_out_h[i].shape[2]):
                dwt_size += 1
        
        print("[INFO] DWT size: ", dwt_size)
        mlp_input_dim = state_obs_dim + dwt_size + encoder_hidden_dims[-1] + self.ref_obs_per_step
        # mlp_input_dim = state_obs_dim + dwt_size + encoder_hidden_dims[-1]
        # mlp_input_dim = state_obs_dim + dwt_size + self.ref_obs_per_step

        # mlp_input_dim = state_obs_dim + encoder_hidden_dims[-1] + self.ref_obs_per_step

        print("ADDITIONAL DIM:" + str(additional_dim))

        ##########################
        ########## MDME ##########
        ##########################

        self.phase_enc = utils.WaveletEncoder(
            self.num_reference_obs,
            latent_channels=self.latent_channels,
            horizon=self.horizon,
            band_outputs=band_outputs,
            wavelet_type=wavelet_type,
            # wavelet_output_dim=dwt_size,
            additional_dim=additional_dim
        )
        # self.phase_enc = utils.PeriodicEncoder(
        #     self.num_reference_obs,
        #     latent_channels=self.latent_channels,
        #     horizon=self.horizon,
        # )
        # mlp_input_dim = state_obs_dim + 4 * self.latent_channels + encoder_hidden_dims[-1] + self.ref_obs_per_step
        self.obs_enc = utils.GaussianEncoderBlock(
            self.num_reference_obs,
            encoder_hidden_dims[-1],
            encoder_dims=encoder_hidden_dims
        )
        # self.obs_enc = utils.GRUBlock(self.num_reference_obs, encoder_hidden_dims[2], 128)
        # self.obs_enc = utils.TransformerEncoder(self.num_reference_obs, encoder_hidden_dims[-1])

        ##########################
        ########## PAE ###########
        ##########################
        # self.pae_enc = utils.PAE(self.num_reference_obs, encoder_hidden_dims[-1], seq_len=25)
        # mlp_input_dim = encoder_hidden_dims[-1] + state_obs_dim

        #########################
        ########## VMP ##########
        #########################
        # self.vmp_enc = utils.GaussianEncoderBlock(
        #     self.num_reference_obs,
        #     encoder_hidden_dims[-1],
        #     encoder_dims=encoder_hidden_dims
        # )
        # mlp_input_dim = encoder_hidden_dims[-1] + state_obs_dim + self.ref_obs_per_step

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
        # check for NaN and replace with 1e5
        if torch.isnan(x).any():
            print("[INFO] NaN detected in input, replacing with 1e5")
            x[torch.isnan(x)] = 1e5
        
        if torch.isinf(x).any():
            print("[INFO] Inf detected in input, replacing with 1e5")
            x[torch.isinf(x)] = 1e5

        ##########################
        ########## MDME ##########
        ##########################

        # split reference obs and obs, pass reference obs through encoder, concatenate with obs and pass through policy
        ref_obs, obs = torch.split(x, [self.num_reference_obs, x.size(1) - self.num_reference_obs], dim=1)
        periodic_out = self.phase_enc(ref_obs)
        enc_out = self.obs_enc(ref_obs)

        # last ref = last 16 items in ref_obs
        last_ref = ref_obs[:, -self.ref_obs_per_step:]
        # concat encodings
        enc_obs = torch.cat([periodic_out, enc_out], dim=1)
        x = torch.cat([enc_obs, last_ref, obs], dim=1)
        # enc_obs = torch.cat([periodic_out], dim=1)
        # x = torch.cat([enc_obs, last_ref, obs], dim=1)

        ##########################
        ########## PAE ###########
        ##########################

        # ref_obs, obs = torch.split(x, [self.num_reference_obs, x.size(1) - self.num_reference_obs], dim=1)
        # pae_out = self.pae_enc(ref_obs)
        # pae_out = pae_out.view(pae_out.size(0), -1)
        # x = torch.cat([pae_out, obs], dim=1)

        #########################
        ########## VMP ##########
        #########################

        # ref_obs, obs = torch.split(x, [self.num_reference_obs, x.size(1) - self.num_reference_obs], dim=1)
        # vmp_out = self.vmp_enc(ref_obs)
        # last_ref = ref_obs[:, -self.ref_obs_per_step:]
        # enc_obs = torch.cat([vmp_out, last_ref], dim=1)
        # x = torch.cat([enc_obs, obs], dim=1)

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
        encoder_hidden_dims=[256, 128, 64],
        state_obs_dim=45,
        reference_obs_dim=16,
        activation="elu",
        latent_channels=6,
        band_outputs=3,
        wavelet_type="db2",
        additional_dim=0.99,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.actor = ImitationAgent(
            num_actor_obs,
            num_actions,
            actor_hidden_dims,
            activation,
            encoder_hidden_dims=encoder_hidden_dims,
            state_obs_dim=state_obs_dim,
            reference_obs_dim=reference_obs_dim,
            latent_channels=latent_channels,
            band_outputs=band_outputs,
            wavelet_type=wavelet_type,
            additional_dim=additional_dim,
        )
        self.critic = ImitationAgent(
            num_critic_obs,
            1,
            critic_hidden_dims,
            activation,
            encoder_hidden_dims=encoder_hidden_dims,
            state_obs_dim=state_obs_dim,
            reference_obs_dim=reference_obs_dim,
            latent_channels=latent_channels,
            band_outputs=band_outputs,
            wavelet_type=wavelet_type,
            additional_dim=additional_dim,
        )

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
        # replace any NaN with 0 to prevent propagation into a crash (I hope)
        observations[torch.isnan(observations)] = 0
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        # self.distribution.sample() causes the following error message:
        # RuntimeError: normal expects all elements of std >= 0.0
        # print values from self.distribution to debug
        # if torch.isnan(self.distribution.stddev).any() or (self.distribution.stddev <= 0).any():
        #     print("[Trace] Invalid values detected in std, replacing with 1e-6")
        #     self.distribution.stddev[torch.isnan(self.distribution.stddev) | (self.distribution.stddev <= 0)] = 0.5
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
