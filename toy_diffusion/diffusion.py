from typing import Optional

import torch
import torch.nn as nn


class Diffuser(nn.Module):
    """Owns a DDPM schedule and the architecture-independent diffusion math."""

    def __init__(
        self,
        num_diffusion_steps: int = 100,
        beta_schedule: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if num_diffusion_steps < 1:
            raise ValueError("num_diffusion_steps must be positive.")

        if beta_schedule is None:
            betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        else:
            if beta_schedule.ndim != 1:
                raise ValueError("beta_schedule must be one-dimensional.")
            if beta_schedule.numel() < num_diffusion_steps:
                raise ValueError("beta_schedule is shorter than num_diffusion_steps.")
            betas = beta_schedule[:num_diffusion_steps]

        if not torch.all((betas > 0) & (betas < 1)):
            raise ValueError("All beta values must lie strictly between zero and one.")

        self.num_diffusion_steps = num_diffusion_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alpha_bars", torch.cumprod(1 - betas, dim=0))

    @staticmethod
    def _batch_coefficients(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Reshapes [B] schedule values to broadcast across any trajectory shape."""
        return values.view(values.shape[0], *([1] * (reference.ndim - 1)))

    def add_noise(
        self,
        x_0: torch.Tensor,
        timesteps: torch.Tensor,
        epsilon: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples x_t for one independently chosen timestep per batch item."""
        if x_0.ndim < 2:
            raise ValueError("x_0 must include a batch dimension and trajectory dimensions.")
        if timesteps.shape != (x_0.shape[0],):
            raise ValueError("timesteps must have shape [batch_size].")
        if epsilon is None:
            epsilon = torch.randn_like(x_0)
        if epsilon.shape != x_0.shape:
            raise ValueError("epsilon must have the same shape as x_0.")

        alpha_bar_t = self._batch_coefficients(self.alpha_bars[timesteps], x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon
        return x_t, epsilon

    def predict_x0(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        epsilon_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Recovers an x_0 estimate from a noisy trajectory and predicted noise."""
        if timesteps.shape != (x_t.shape[0],):
            raise ValueError("timesteps must have shape [batch_size].")
        if epsilon_pred.shape != x_t.shape:
            raise ValueError("epsilon_pred must have the same shape as x_t.")

        alpha_bar_t = self._batch_coefficients(self.alpha_bars[timesteps], x_t)
        return (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)

    def denoise_step(
        self,
        x_t: torch.Tensor,
        timestep: int,
        epsilon_pred: torch.Tensor,
        epsilon: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Performs one ancestral DDPM reverse step shared by the whole batch."""
        if not 0 <= timestep < self.num_diffusion_steps:
            raise ValueError("timestep is outside the configured diffusion schedule.")
        if epsilon_pred.shape != x_t.shape:
            raise ValueError("epsilon_pred must have the same shape as x_t.")

        beta_t = self.betas[timestep]
        alpha_t = self.alphas[timestep]
        alpha_bar_t = self.alpha_bars[timestep]
        mean = (x_t - beta_t * epsilon_pred / torch.sqrt(1 - alpha_bar_t)) / torch.sqrt(alpha_t)

        if timestep == 0:
            return mean

        if epsilon is None:
            epsilon = torch.randn_like(x_t)
        if epsilon.shape != x_t.shape:
            raise ValueError("epsilon must have the same shape as x_t.")
        posterior_variance = beta_t * (1 - self.alpha_bars[timestep - 1]) / (1 - alpha_bar_t)
        return mean + torch.sqrt(posterior_variance) * epsilon
