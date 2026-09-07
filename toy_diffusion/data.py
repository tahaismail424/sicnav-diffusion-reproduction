import math
from pathlib import Path
from typing import Literal, Tuple

import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Cached synthetic 2D trajectories with multimodal future turns."""

    cache_version = "v2"

    def __init__(
        self,
        partition: Literal["train", "val"],
        seed: int = 32,
        train_size: int = 10_000,
        val_size: int = 2_000,
        start_span: Tuple[float, float] = (16.0, 16.0),
        avg_speed: float = 1.0,
        future_turn_angle: float = math.pi / 2,
        past_seconds: float = 5.0,
        future_seconds: float = 1.0,
        hz: float = 10.0,
        speed_variance: float = 0.1,
        turn_variance: float = 0.1,
    ):
        super().__init__()
        if partition not in {"train", "val"}:
            raise ValueError("partition must be 'train' or 'val'.")

        self.partition = partition
        self.seed = seed
        self.size = train_size if partition == "train" else val_size
        self.start_span = start_span
        self.avg_speed = avg_speed
        self.future_turn_angle = future_turn_angle
        self.past_seconds = past_seconds
        self.future_seconds = future_seconds
        self.hz = hz
        self.speed_variance = speed_variance
        self.turn_variance = turn_variance
        self.obs_len = int(round(past_seconds * hz))
        self.pred_len = int(round(future_seconds * hz))
        self.total_len = self.obs_len + self.pred_len
        if self.obs_len < 2 or self.pred_len < 1:
            raise ValueError("The dataset needs at least two past points and one future point.")

        self.actual_seed = seed if partition == "train" else seed + 10_000
        self.cache_dir = Path(__file__).parent / "cache"
        cache_path = self.cache_path()
        if cache_path.exists():
            cached = self.load_data(cache_path)
        else:
            cached = self.generate_data()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(cached, cache_path)

        self.positions = cached["positions"]
        self.modes = cached["modes"]

    def cache_path(self) -> Path:
        filename = (
            f"synthetic_{self.cache_version}_{self.partition}_seed{self.actual_seed}"
            f"_n{self.size}_obs{self.obs_len}_pred{self.pred_len}"
            f"_speed{self.avg_speed}_turn{self.future_turn_angle}"
            f"_speedvar{self.speed_variance}_turnvar{self.turn_variance}.pt"
        )
        return self.cache_dir / filename

    def generate_data(self):
        generator = torch.Generator().manual_seed(self.actual_seed)
        dt = 1.0 / self.hz

        initial_position = 2 * torch.rand(
            self.size, 2, generator=generator
        ) - 1
        initial_position[:, 0] *= self.start_span[0]
        initial_position[:, 1] *= self.start_span[1]
        initial_heading = 2 * math.pi * torch.rand(self.size, generator=generator)

        speed_std = math.sqrt(self.speed_variance)
        speeds = self.avg_speed + speed_std * torch.randn(
            self.size, self.total_len, generator=generator
        )
        speeds = speeds.clamp_min(0.1)

        turn_rate_std = math.sqrt(self.turn_variance)
        past_turn_rates = turn_rate_std * torch.randn(
            self.size, self.obs_len, generator=generator
        )

        # One latent mode per trajectory; it is applied only after the past ends.
        modes = torch.randint(-1, 2, (self.size,), generator=generator)
        target_turn_rate = modes[:, None] * (
            self.future_turn_angle / self.future_seconds
        )
        future_turn_rates = target_turn_rate + turn_rate_std * torch.randn(
            self.size, self.pred_len, generator=generator
        )
        turn_rates = torch.cat([past_turn_rates, future_turn_rates], dim=1)

        positions = torch.empty(self.size, self.total_len, 2)
        positions[:, 0] = initial_position
        heading = initial_heading
        for step in range(1, self.total_len):
            heading = heading + turn_rates[:, step] * dt
            displacement = speeds[:, step] * dt
            positions[:, step, 0] = positions[:, step - 1, 0] + displacement * torch.cos(heading)
            positions[:, step, 1] = positions[:, step - 1, 1] + displacement * torch.sin(heading)

        return {
            "positions": positions,
            "modes": modes,
            "metadata": {
                "seed": self.actual_seed,
                "partition": self.partition,
                "obs_len": self.obs_len,
                "pred_len": self.pred_len,
                "hz": self.hz,
            },
        }

    @staticmethod
    def load_data(path: Path):
        return torch.load(path, map_location="cpu")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        trajectory = self.positions[index]
        past = trajectory[:self.obs_len]
        future = trajectory[self.obs_len:]
        origin = past[-1]

        # Predict motion in a local frame rather than arbitrary world coordinates.
        return past - origin, future - origin
