import torch

def noise(
    timestep: int, 
    alpha_bars: torch.tensor,
    x_0: torch.tensor, 
    epsilon: torch.tensor
    ) -> torch.tensor:
    
    alpha_bar_t =  alpha_bars[timestep]
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon
    return x_t

def denoise(
    timestep: int,
    betas: torch.tensor,
    alpha_bars: torch.tensor,
    x_t: torch.tensor,
    epsilon_pred: torch.tensor
    ) -> torch.tensor:
    beta = betas[timestep]
    alpha_bar_t = alpha_bars[timestep]
    
    mean = (x_t - (beta / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred) / torch.sqrt(1 - beta)
    
    if timestep > 0:
        sigma_t = torch.sqrt(((1 - alpha_bars[timestep - 1]) / (1 - alpha_bar_t)) * beta)
        x_tm = mean + sigma_t * torch.randn_like(x_t)
    else:
        x_tm = mean
    return x_tm
    