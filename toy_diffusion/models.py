import math
from typing import Optional

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
                 embedding_dim: int = 32,
                 max_period: float = 10_000.0,
                 num_diffusion_steps: int = 100,
                 beta_schedule: Optional[torch.Tensor] = None
                 ):
        super().__init__()
        self.past_encoder = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.timestep_encoder = SinusoidalEmbedding(embedding_dim, max_period)
        self.ffn = nn.Sequential(
            nn.Linear(84, 150),
            nn.ReLU(),
            nn.Linear(150, 64),
            nn.ReLU(),
            nn.Linear(64, 20)
        )
        
        self.num_diffusion_steps = num_diffusion_steps
        if beta_schedule is not None:
            betas = beta_schedule[:num_diffusion_steps]
        else:
            betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alpha_bars = torch.cumprod(1 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

    def forward(self, x_t, past_trajectory, timesteps):
        past_encoding = self.past_encoder(past_trajectory)
        timestep_embedding = self.timestep_encoder(timesteps)
        denoiser_input = torch.cat([past_encoding, timestep_embedding, x_t], dim=-1)
        return self.ffn(denoiser_input)
