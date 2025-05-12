import torch
import torch.nn as nn

import math

# class TransformerBlock(nn.Module):
#     def __init__(self, latent_dim, num_head, dropout_rate):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.ln_1 = nn.LayerNorm(latent_dim)
#         self.attention_block = nn.MultiheadAttention(latent_dim, num_head, dropout=dropout_rate, batch_first=True)
#         self.ln_2 = nn.LayerNorm(latent_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(latent_dim, latent_dim * 4),
#             nn.ReLU(),
#             nn.Linear(latent_dim * 4, latent_dim),
#         )

#     def forward(self, x):
#         attn_output, _ = self.attention_block(x, x, x)
#         x = x + self.dropout(attn_output)
#         x = self.ln_1(x)
#         x = self.mlp(x)
#         x = x + self.dropout(x)
#         x = self.ln_2(x)
#         return x

class VQVAEBlock(nn.Module):
    def __init__(
            self,
            input_dims,
            num_actions,
            codebook_dims=64,
            num_embeddings=512
        ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, codebook_dims),
        )

        self.codebook = nn.Embedding(num_embeddings, codebook_dims)
        self.codebook.weight.data.normal_()

        self.decoder = nn.Sequential(
            nn.Linear(codebook_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        # Encoder pass
        z_e = self.encoder(x)  # Shape: [num_envs, codebook_dims]

        # Compute distances between encoded vectors and codebook embeddings
        codebook_expanded = self.codebook.weight.unsqueeze(0)  # Shape: [1, num_embeddings, codebook_dims]
        distances = torch.norm(z_e.unsqueeze(1) - codebook_expanded, dim=-1)  # Shape: [num_envs, num_embeddings]
        encoding_indices = torch.argmin(distances, dim=1)  # Shape: [num_envs]
        
        # Retrieve closest embeddings
        z_q = self.codebook(encoding_indices)  # Quantized vector, Shape: [num_envs, codebook_dims]

        # Stop gradients for the quantized vector
        z_q = z_e + (z_q - z_e).detach()
        
        # Decoder pass
        x_recon = self.decoder(z_q)  # Shape: [num_envs, num_actions]
        
        # Calculate VQ-VAE loss
        # commitment_loss = self.commitment_cost * F.mse_loss(z_e.detach(), z_q)
        # codebook_loss = F.mse_loss(z_e, z_q.detach())
        # vqvae_loss = commitment_loss + codebook_loss
        
        return x_recon
    
class GRUBlock(nn.Module):
    def __init__(
            self,
            input_dims,
            output_dims,
            hidden_dim,
            num_layers=2
    ):
        super().__init__()

        self.encoder = nn.GRU(
            input_dims,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # self.decoder = nn.GRU(
        #     hidden_dim,
        #     output_dims,
        #     num_layers=num_layers,
        #     batch_first=True
        # )

        self.fc = nn.Linear(hidden_dim, output_dims)
    
    def forward(self, x):
        x, hidden = self.encoder(x)
        # x, _ = self.decoder(x, hidden)
        x = self.fc(x)
        return x

class GaussianEncoderBlock(nn.Module):
    def __init__(
            self,
            input_dims,
            output_dims,
            encoder_dims=[256, 256, 128, 128, 64, 64],
            decoder_dims=[64, 128, 128],
            latent_dims=128
        ):
        super().__init__()

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dims, encoder_dims[0]))
        encoder_layers.append(nn.ReLU())
        for layer_index in range(len(encoder_dims)):
            if layer_index == len(encoder_dims) - 2:
                encoder_layers.append(nn.Linear(encoder_dims[layer_index], encoder_dims[-1]))
                break
            else:
                encoder_layers.append(nn.Linear(encoder_dims[layer_index], encoder_dims[layer_index + 1]))
                encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        latent_dims = output_dims
        # Mean and log variance for latent space
        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dims)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dims)
        
        # Decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dims, decoder_dims[0]),
        #     nn.ReLU(),
        #     nn.Linear(decoder_dims[0], decoder_dims[1]),
        #     nn.ReLU(),
        #     nn.Linear(decoder_dims[1], decoder_dims[2]),
        #     nn.ReLU(),
        #     nn.Linear(decoder_dims[2], output_dims)
        # )
    
    def encode(self, x):
        """Encodes the input into latent space, producing mu and logvar."""
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, var) using N(0, 1)."""
        logvar = torch.clamp(logvar, -10, 10)  # Clip log variance for numerical stability
        std = torch.exp(0.5 * logvar)  # Calculate standard deviation
        eps = torch.randn_like(std)    # Sample epsilon from standard normal
        return mu + eps * std          # Return reparameterized sample
    
    def decode(self, z):
        """Decodes the latent variable z back to the original input dimensions."""
        return self.decoder(z)
    
    def forward(self, x):
        # Encode the input to get mu and logvar
        mu, logvar = self.encode(x)
        
        # Sample z from latent space using reparameterization trick
        z = self.reparameterize(mu, logvar)

        return z

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        x = x + self.encoding[:, :x.size(1), :].to(x.device)
        return x

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            num_layers=2,
            num_heads=4,
            sequence_length=25,
            d_model=256):
        super().__init__()

        self.seq_len = sequence_length
        input_len = input_dim // sequence_length

        self.input_embedding = nn.Linear(input_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=sequence_length)
        encoder = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], self.seq_len, -1)
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        return self.output_layer(x)[:, -1, :]
    
import numpy as np


class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        print(y.shape)
        print(self.alpha.shape)
        y = y * self.alpha + self.beta
        return y
    
from pytorch_wavelets import DWTForward
class WaveletEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            horizon=15,
            latent_channels=2,
            band_outputs=3,
            wavelet_type='db3',
            dt=0.02,
            # wavelet_output_dim=1,
            additional_dim=2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.latent_channels = latent_channels
        self.dt = dt
        self.additional_dim = additional_dim

        self.args = torch.linspace(-(horizon - 1) * self.dt / 2, (horizon - 1) * self.dt / 2, self.horizon, dtype=torch.float, device='cuda')
        self.freqs = torch.fft.rfftfreq(horizon, device='cuda')[1:] * horizon
        self.encoder_shape = int(self.input_dim / self.horizon) # self.horizon or 4 (?)

        enc_layers = []

        enc_layers.append(nn.Conv1d(
            self.input_dim // self.horizon,
            self.encoder_shape,
            self.horizon,
            stride=1,
            padding=int((self.horizon - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
        ))
        enc_layers.append(nn.BatchNorm1d(num_features=self.encoder_shape))
        enc_layers.append(nn.ELU())
        enc_layers.append(nn.Conv1d(
            self.encoder_shape,
            2 * self.latent_channels,
            self.horizon,
            stride=1,
            padding=int((self.horizon - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
        ))
        enc_layers.append(nn.BatchNorm1d(num_features=2 * self.latent_channels))
        enc_layers.append(nn.ELU())
        enc_layers.append(nn.Conv1d(
            2 * self.latent_channels,
            self.latent_channels,
            self.horizon,
            stride=1,
            padding=int((self.horizon - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
        ))

        self.encoder = nn.Sequential(*enc_layers)
        print("WAVELET_TYPE: ", wavelet_type)
        self.dwt = DWTForward(J=band_outputs, wave=wavelet_type, mode='zero')

        # self.output_enc = nn.Linear(
        #     wavelet_output_dim,
        #     desired_output_dim
        # )

        # self.whatever = torch.zeros((1000, self.latent_channels, self.horizon), dtype=torch.float32, device='cuda')
        # self.count = 0

    def forward(self, x):
        x = x.reshape(x.shape[0], self.input_dim // self.horizon, self.horizon)
        x = self.encoder(x)

        # self.whatever[self.count] = x
        # self.count += 1

        # if self.count == 1000:
        #     print(self.whatever.shape)
        #     torch.save(self.whatever, 'wavelet_encoded.pt')
        #     self.count = 0

        yl, yh = self.dwt(x.unsqueeze(1))
        yl = yl.reshape(x.shape[0], -1)
        y = yl

        # for i in range(len(yh)):            # iterate through decomposition levels
        #     y_temp = yh[i].squeeze(1).reshape(x.shape[0], -1)
        #     y = torch.cat([y, y_temp], dim=-1)

        yl_norm = (yl ** 2) / torch.sum(yl ** 2, dim=-1, keepdim=True)
        if self.additional_dim == 1.0:
            y = -torch.sum(yl_norm * torch.log2(yl_norm + 1e-10), dim=-1, keepdim=True)
        else:
            y = 1 / (self.additional_dim - 1) * (1 - torch.sum(yl_norm ** self.additional_dim, dim=-1, keepdim=True)) # Tsallis Entropy
        for i in range(len(yh)):            # iterate through decomposition levels
            for j in range(yh[i].shape[2]):    # iterate through LH, HL, HH
                y_temp = yh[i][:, : j, ...].reshape(x.shape[0], -1)
                y_temp_norm = (y_temp ** 2) / torch.sum(y_temp ** 2, dim=-1, keepdim=True)
                if self.additional_dim == 1.0:  # Shannon Entropy
                    y_temp = -torch.sum(y_temp_norm * torch.log2(y_temp_norm + 1e-10), dim=-1, keepdim=True) # Shannon Entropy
                else:                           # Tsallis Entropy   
                    q = self.additional_dim
                    y_temp = 1 / (q - 1) * (1 - torch.sum(y_temp_norm ** q, dim=-1, keepdim=True)) # Tsallis Entropy
                y = torch.cat([y, y_temp], dim=-1)
            
            # y_temp = yh[i].squeeze(1).reshape(x.shape[0], -1)
            # # # newish
            # # y_temp_norm = (y_temp ** 2) / torch.sum(y_temp ** 2, dim=-1, keepdim=True)
            # # y_temp = -torch.sum(y_temp_norm * torch.log2(y_temp_norm + 1e-10), dim=-1, keepdim=True)
            # # y = torch.cat([y, y_temp], dim=-1)

            # y_temp_norm = (y_temp ** 2) / torch.sum(y_temp ** 2, dim=-1, keepdim=True)
            # y_temp = 1 / (self.additional_dim - 1) * (1 - torch.sum(y_temp_norm ** self.additional_dim, dim=-1, keepdim=True)) # Tsallis Entropy
            # y = torch.cat([y, y_temp], dim=-1)

            # old:
            # y_temp = yh[i].squeeze(1).reshape(x.shape[0], -1)
            # y = torch.cat([y, y_temp], dim=-1)
        return y


class PeriodicEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            horizon=15,
            latent_channels=6,
            dt=0.02
    ):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.latent_channels = latent_channels
        self.dt = dt

        self.args = torch.linspace(-(horizon - 1) * self.dt / 2, (horizon - 1) * self.dt / 2, self.horizon, dtype=torch.float, device='cuda')
        self.freqs = torch.fft.rfftfreq(horizon, device='cuda')[1:] * horizon
        self.encoder_shape = int(self.input_dim / self.horizon)

        enc_layers = []

        enc_layers.append(nn.Conv1d(
            self.input_dim // self.horizon,
            self.encoder_shape,
            self.horizon,
            stride=1,
            padding=int((self.horizon - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
        ))
        enc_layers.append(nn.BatchNorm1d(num_features=self.encoder_shape))
        enc_layers.append(nn.ELU())
        enc_layers.append(nn.Conv1d(
            self.encoder_shape,
            2 * self.latent_channels,
            self.horizon,
            stride=1,
            padding=int((self.horizon - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
        ))
        enc_layers.append(nn.BatchNorm1d(num_features=2 * self.latent_channels))
        enc_layers.append(nn.ELU())
        enc_layers.append(nn.Conv1d(
            2 * self.latent_channels,
            self.latent_channels,
            self.horizon,
            stride=1,
            padding=int((self.horizon - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
        ))

        self.encoder = nn.Sequential(*enc_layers)

        self.phase_encoder = nn.ModuleList()
        for _ in range(latent_channels):
            phase_enc_layers = []
            phase_enc_layers.append(nn.Linear(self.horizon, 2))
            phase_enc_layers.append(nn.BatchNorm1d(num_features=2))
            self.phase_encoder.append(nn.Sequential(*phase_enc_layers))
        self.phase_encoder.train()

    def FFT(self, function, dim=2):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:]
        power = torch.square(spectrum)
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.horizon
        offset = rfft.real[:, :, 0] / self.horizon
        
        return freq, amp, offset

    def forward(self, x):
        x = x.reshape(x.shape[0], self.input_dim // self.horizon, self.horizon)
        x = self.encoder(x)
        f, a, b = self.FFT(x, dim=2)
        p = torch.empty((x.shape[0], self.latent_channels), dtype=torch.float32, device=x.device)
        for i in range(self.latent_channels):
            v = self.phase_encoder[i](x[:, i, :])
            p[:, i] = torch.atan2(v[:, 1], v[:, 0]) / (2 * np.pi)

        y = torch.cat([p, f, a, b], dim=-1)

        return y
        

class PAE(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            num_embedding=6,
            seq_len=15
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_embedding = num_embedding
        self.output_dim = output_dim * seq_len
        self.seq_len = seq_len
        self.window = self.seq_len / 50

        self.tpi = nn.Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = nn.Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.seq_len, dtype=np.float32)), requires_grad=False)
        self.freqs = nn.Parameter(torch.fft.rfftfreq(seq_len)[1:] * seq_len / self.window, requires_grad=False)     # take [1:] as the first element is DC

        self.intermediate_dims = int(input_dim/3)

        self.conv1 = nn.Conv1d(
            input_dim // seq_len,
            self.intermediate_dims,
            seq_len,
            stride=1,
            padding=int((seq_len - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
        )
        # self.norm1 = LN_v2(seq_len)
        self.norm1 = nn.LayerNorm(self.intermediate_dims * seq_len )
        self.elu1 = nn.ELU()
        self.conv2 = nn.Conv1d(
            self.intermediate_dims,
            num_embedding,
            seq_len,
            stride=1,
            padding=int((seq_len - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
        )

        self.fc = torch.nn.ModuleList()
        for i in range(num_embedding):
            self.fc.append(nn.Linear(seq_len, 2))

        # self.rnn_output = nn.GRU(
        #     num_embedding,
        #     num_embedding,
        #     num_layers=1,
        #     batch_first=True
        # )

        self.deconv1 = nn.Conv1d(
            num_embedding,
            self.intermediate_dims,
            seq_len,
            stride=1,
            padding=int((seq_len - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
        )
        # self.denorm1 = LN_v2(seq_len)
        self.denorm1 = nn.LayerNorm(self.intermediate_dims * seq_len)

        self.elu2 = nn.ELU()
        self.deconv2 = nn.Conv1d(
            self.intermediate_dims,
            output_dim,
            seq_len,
            stride=1,
            padding=int((seq_len - 1) / 2),
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
        )
    
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:, :, 1:]
        power = spectrum.pow(2)
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.seq_len
        offset = rfft.real[:, :, 0] / self.seq_len
        
        return freq, amp, offset
    
    def forward(self, x):
        x = x.reshape(x.shape[0], self.input_dim // self.seq_len, self.seq_len)

        x = self.conv1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.norm1(x)
        x = x.reshape(x.shape[0], self.intermediate_dims, self.seq_len)
        x = self.elu1(x)
        x = self.conv2(x)

        # latent = x

        f, a, b = self.FFT(x, dim=2)

        p = torch.empty((x.shape[0], self.num_embedding), dtype=torch.float32, device=x.device)
        for i in range(self.num_embedding):
            v = self.fc[i](x[:, i, :])
            p[:, i] = torch.atan2(v[:, 1], v[:, 0]) / self.tpi
        
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        # params = [p, f, a, b]

        x = a * torch.sin(self.tpi * (f * self.args + p)) + b

        # signal = x
        # x = x.permute(0, 2, 1)
        # x, _ = self.rnn_output(x)
        # x = x.permute(0, 2, 1)

        x = self.deconv1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.denorm1(x)
        x = x.reshape(x.shape[0], self.intermediate_dims, self.seq_len)
        x = self.elu2(x)
        x = self.deconv2(x)

        x = x[:, :, -1].reshape(x.shape[0], -1)

        return x


