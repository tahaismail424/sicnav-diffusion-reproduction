import torch

def noise(
    timesteps: torch.Tensor,
    alpha_bars: torch.Tensor,
    x_0: torch.Tensor,
    epsilon: torch.Tensor,
    ) -> torch.Tensor:
    
    alpha_bar_t =  alpha_bars[timesteps]
    x_t = torch.sqrt(alpha_bar_t)[:, None] * x_0 + torch.sqrt(1 - alpha_bar_t)[:, None] * epsilon
    return x_t

def denoise(
    timestep: int,
    betas: torch.Tensor,
    alpha_bars: torch.Tensor,
    x_t: torch.Tensor,
    epsilon_pred: torch.Tensor,
    ) -> torch.Tensor:
    beta = betas[timestep]
    alpha_bar_t = alpha_bars[timestep]
    
    mean = (x_t - (beta / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred) / torch.sqrt(1 - beta)
    
    if timestep > 0:
        sigma_t = torch.sqrt(((1 - alpha_bars[timestep - 1]) / (1 - alpha_bar_t)) * beta)
        x_tm = mean + sigma_t * torch.randn_like(x_t)
    else:
        x_tm = mean
    return x_tm
