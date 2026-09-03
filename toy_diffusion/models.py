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
    def __init__(self, embedding_dim: int = 32, max_period: float = 10_000.0):
        super().__init__()
        self.ccoder = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.tcoder = SinusoidalEmbedding(embedding_dim, max_period)
        self.ffn = nn.Sequential(
            nn.Linear(84, 150),
            nn.ReLU(),
            nn.Linear(150, 64),
            nn.ReLU(),
            nn.Linear(64, 20)
        )
        
    def forward(self, x_t, context, timesteps):
        c_encoding = self.ccoder(context)
        t_embedding = self.tcoder(timesteps)
        input = torch.cat([c_encoding, t_embedding, x_t], dim=-1)
        return self.ffn(input)
