from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset


class NavBenchmarkDataset(Dataset):
    """Flattened single-pedestrian windows from one Social GAN benchmark fold."""

    def __init__(
        self,
        partition: Literal["train", "test", "val"],
        test_set: Literal["eth", "hotel", "univ", "zara1", "zara2"],
        obs_samples: int = 8,
        pred_samples: int = 12,
        window_stride: int = 1,
        expected_frame_delta: int = 10,
        data_root: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        if partition not in {"train", "val", "test"}:
            raise ValueError("partition must be 'train', 'val', or 'test'.")
        if test_set not in {"eth", "hotel", "univ", "zara1", "zara2"}:
            raise ValueError("test_set must be one of eth, hotel, univ, zara1, zara2.")
        if obs_samples < 1 or pred_samples < 1 or window_stride < 1:
            raise ValueError("Sample counts and window_stride must be positive.")

        self.partition = partition
        self.test_set = test_set
        self.obs_samples = obs_samples
        self.pred_samples = pred_samples
        self.window_stride = window_stride
        self.expected_frame_delta = expected_frame_delta
        self.window_size = obs_samples + pred_samples
        self.data_root = (
            Path(data_root)
            if data_root is not None
            else Path(__file__).parent / "data"
        )

        self.data, self.file_names = self.load()

    def load(self):
        split_dir = self.data_root / self.test_set / self.partition
        txt_files = sorted(split_dir.glob("*.txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in {split_dir}.")

        trajectories = []
        file_ids = []
        frame_starts = []
        pedestrian_ids = []
        file_names = [path.stem for path in txt_files]

        for file_id, path in enumerate(txt_files):
            frame = pd.read_csv(
                path,
                sep=r"\s+",
                header=None,
                names=["frame", "pid", "x", "y"],
            )
            if frame.isnull().any().any():
                raise ValueError(f"Missing values found in {path}.")

            frame["frame"] = frame["frame"].astype(int)
            frame["pid"] = frame["pid"].astype(int)
            frame = frame.sort_values(["frame", "pid"])
            if frame.duplicated(["frame", "pid"]).any():
                raise ValueError(f"Duplicate frame/pedestrian rows found in {path}.")

            frames = frame["frame"].drop_duplicates().tolist()
            last_start = len(frames) - self.window_size + 1
            for start_index in range(0, last_start, self.window_stride):
                window_frames = frames[start_index:start_index + self.window_size]
                if any(
                    later - earlier != self.expected_frame_delta
                    for earlier, later in zip(window_frames, window_frames[1:])
                ):
                    continue

                window = frame[frame["frame"].isin(window_frames)]
                for pedestrian_id, pedestrian in window.groupby("pid"):
                    pedestrian = pedestrian.set_index("frame").reindex(window_frames)
                    if pedestrian[["x", "y"]].isnull().any().any():
                        continue

                    trajectories.append(
                        torch.tensor(
                            pedestrian[["x", "y"]].to_numpy(), dtype=torch.float32
                        )
                    )
                    file_ids.append(file_id)
                    frame_starts.append(window_frames[0])
                    pedestrian_ids.append(pedestrian_id)

        if not trajectories:
            raise RuntimeError(
                f"No valid {self.window_size}-step windows found in {split_dir}."
            )

        return {
            "trajectories": torch.stack(trajectories),
            "file_ids": torch.tensor(file_ids, dtype=torch.long),
            "frame_starts": torch.tensor(frame_starts, dtype=torch.long),
            "pedestrian_ids": torch.tensor(pedestrian_ids, dtype=torch.long),
        }, file_names

    def __len__(self):
        return len(self.data["trajectories"])

    def file_name(self, file_id: int) -> str:
        """Returns the source filename stem for an item's numeric file ID."""
        return self.file_names[file_id]

    def __getitem__(self, index):
        trajectory = self.data["trajectories"][index]
        observed = trajectory[:self.obs_samples]
        future = trajectory[self.obs_samples:]
        origin = observed[-1]

        # Use the final observed position as the local coordinate origin.
        return (
            observed - origin,
            future - origin,
            self.data["file_ids"][index],
            self.data["frame_starts"][index],
            self.data["pedestrian_ids"][index],
        )
