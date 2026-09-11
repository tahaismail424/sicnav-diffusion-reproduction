import math

import torch
import torch.nn as nn

    
class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int = 32, max_period: float = 10_000.0):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for sine/cosine pairs.")

        self.embedding_dim = embedding_dim
        self.max_period = max_period
        
        half_dim = embedding_dim // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("frequencies", frequencies)
    
    def forward(self, timesteps):
        timesteps = timesteps.to(torch.float32)
        args = timesteps[:, None] * self.frequencies[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

class Denoiser(nn.Module):
    def __init__(self,
                 obs_len: int = 50,
                 pred_len: int = 10,
                 coordinate_dim: int = 2,
                 context_dim: int = 32,
                 timestep_embedding_dim: int = 32,
                 max_period: float = 10_000.0,
                 ):
        super().__init__()
        if obs_len < 1 or pred_len < 1 or coordinate_dim < 1:
            raise ValueError("Trajectory dimensions must be positive.")

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.coordinate_dim = coordinate_dim
        self.past_dim = obs_len * coordinate_dim
        self.future_dim = pred_len * coordinate_dim

        encoder_hidden_dim = max(64, 2 * self.past_dim)
        # Keep the original toy model's capacity while allowing other horizons.
        denoiser_hidden_dim = 150
        self.past_encoder = nn.Sequential(
            nn.Linear(self.past_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, self.past_dim),
            nn.ReLU(),
            nn.Linear(self.past_dim, 64),
            nn.ReLU(),
            nn.Linear(64, context_dim)
        )
        self.timestep_encoder = SinusoidalEmbedding(
            timestep_embedding_dim, max_period
        )
        self.ffn = nn.Sequential(
            nn.Linear(
                context_dim + timestep_embedding_dim + self.future_dim,
                denoiser_hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(denoiser_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.future_dim)
        )
        
    def forward(self, x_t, past_trajectory, timesteps):
        past_encoding = self.past_encoder(past_trajectory)
        timestep_embedding = self.timestep_encoder(timesteps)
        denoiser_input = torch.cat([past_encoding, timestep_embedding, x_t], dim=-1)
        return self.ffn(denoiser_input)
