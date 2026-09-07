import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam

try:
    from .data import TrajectoryDataset
    from .diffusion import noise
    from .models import Denoiser
except ImportError:
    # Allow both `python -m toy_diffusion.train` and direct script execution.
    from data import TrajectoryDataset
    from diffusion import noise
    from models import Denoiser


def train(epochs: int, seed: int, batch_size: int, num_workers: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    model = Denoiser().to(device)
    train_set = TrajectoryDataset(partition="train", seed=seed)
    val_set = TrajectoryDataset(partition="val", seed=seed)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        )
    loss_fn = nn.MSELoss()
    adam = Adam(model.parameters(), lr=1e-3)
    epoch_losses_train = []
    epoch_losses_val = []
    denoising_ades = []
    denoising_fdes = []

    for epoch in range(epochs):
        print("starting epoch training: epoch", epoch)
        model.train()
        total_loss = 0.0
        n_batches = 0
        for past, future in train_loader:
            batch_size, _, _ = past.shape
            past_flat = torch.flatten(past, start_dim=1).to(device)
            future_flat = torch.flatten(future, start_dim=1).to(device)
            epsilon = torch.randn_like(future_flat)
            timesteps = torch.randint(
                low=0,
                high=model.num_diffusion_steps,
                size=(batch_size,),
                device=device,
            )

            adam.zero_grad()
            x_t = noise(timesteps, model.alpha_bars, future_flat, epsilon)
            epsilon_pred = model(x_t, past_flat, timesteps)
            loss = loss_fn(epsilon_pred, epsilon)
            loss.backward()
            adam.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        epoch_losses_train.append(avg_loss)
        print("epoch training epsilon MSE:", avg_loss)

        print("starting epoch validation: epoch", epoch)
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            n_val_batches = 0
            ade_sum = 0.0
            fde_sum = 0.0
            for past, future in val_loader:
                batch_size, _, _ = past.shape
                past_flat = torch.flatten(past, start_dim=1).to(device)
                future_flat = torch.flatten(future, start_dim=1).to(device)
                epsilon = torch.randn_like(future_flat)
                timesteps = torch.randint(
                    low=0,
                    high=model.num_diffusion_steps,
                    size=(batch_size,),
                    device=device,
                )

                x_t = noise(timesteps, model.alpha_bars, future_flat, epsilon)
                epsilon_pred = model(x_t, past_flat, timesteps)
                loss = loss_fn(epsilon_pred, epsilon)
                total_val_loss += loss.item()

                # Reconstruct the known clean future from its corrupted version.
                alpha_bar_t = model.alpha_bars[timesteps].unsqueeze(-1)
                future_estimate = (
                    x_t - torch.sqrt(1 - alpha_bar_t) * epsilon_pred
                ) / torch.sqrt(alpha_bar_t)
                future_points = future_flat.view(batch_size, -1, 2)
                estimated_points = future_estimate.view(batch_size, -1, 2)
                point_errors = torch.linalg.vector_norm(
                    estimated_points - future_points, dim=-1
                )
                ade_sum += point_errors.mean().item()
                fde_sum += point_errors[:, -1].mean().item()
                n_val_batches += 1

            avg_ade = ade_sum / n_val_batches
            avg_fde = fde_sum / n_val_batches
            denoising_ades.append(avg_ade)
            denoising_fdes.append(avg_fde)
            avg_val_loss = total_val_loss / n_val_batches
            epoch_losses_val.append(avg_val_loss)
            print("epoch validation epsilon MSE:", avg_val_loss)
            print("epoch denoising ADE:", avg_ade)
            print("epoch denoising FDE:", avg_fde)

    model.eval()
    return model, epoch_losses_train, epoch_losses_val, denoising_ades, denoising_fdes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs training loop for toy diffusion model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    train(args.epochs, args.seed, args.batch_size, args.num_workers)
