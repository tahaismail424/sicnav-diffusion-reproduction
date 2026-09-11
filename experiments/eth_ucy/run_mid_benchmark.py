#!/usr/bin/env python3
"""Run one or more validation-selected MID ETH/UCY folds without Jupyter.

Examples:
    python experiments/eth_ucy/run_mid_benchmark.py --device cuda
    python experiments/eth_ucy/run_mid_benchmark.py --device cuda --scenes eth
    python experiments/eth_ucy/run_mid_benchmark.py \
        --device cuda:0 --data-dir /scratch/user/eth_ucy/data \
        --output-dir /scratch/user/mid_outputs --scenes hotel zara1
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from easydict import EasyDict


SCENES = ("eth", "hotel", "univ", "zara1", "zara2")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MID_ROOT = PROJECT_ROOT / "trajectory_prediction" / "MID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation-selected MID ETH/UCY benchmark folds."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=MID_ROOT / "configs" / "baseline.yaml",
        help="Base MID YAML configuration.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "eth_ucy" / "data",
        help="Directory containing <scene>_{train,val,test}.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "eth_ucy" / "outputs" / "mid",
        help="Directory for checkpoints, TensorBoard logs, and result CSVs.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=SCENES,
        default=SCENES,
        help="Held-out scene(s). Defaults to all five leave-one-scene-out folds.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device, e.g. cuda, cuda:0, mps, cpu, or auto.",
    )
    parser.add_argument("--epochs", type=int, help="Override YAML epochs.")
    parser.add_argument("--batch-size", type=int, help="Override train/eval batch size.")
    parser.add_argument("--num-samples", type=int, help="Best-of-K sample count for validation/test.")
    parser.add_argument("--sampling", choices=("ddpm", "ddim"), help="Override sampling method.")
    parser.add_argument("--sampling-step", type=int, help="Reverse-process step stride.")
    parser.add_argument("--evaluation-stride", type=int, help="Forecast-origin stride in scene timesteps.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument(
        "--run-name",
        default="mid",
        help="Checkpoint directory prefix; each fold becomes <run-name>_<scene>.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        help="Optional aggregate result CSV. Defaults inside output-dir.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip scenes already present in the result CSV.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def validate_data_dir(data_dir: Path, scenes: tuple[str, ...]) -> None:
    missing = [
        data_dir / f"{scene}_{split}.pkl"
        for scene in scenes
        for split in ("train", "val", "test")
        if not (data_dir / f"{scene}_{split}.pkl").is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing processed ETH/UCY pickle(s):\n{formatted}")


def build_config(base_config: dict, args: argparse.Namespace, scene: str) -> EasyDict:
    config = deepcopy(base_config)
    config.update(
        {
            "dataset": scene,
            "exp_name": f"{args.run_name}_{scene}",
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "device": args.device,
        }
    )
    overrides = {
        "epochs": args.epochs,
        "num_samples": args.num_samples,
        "sampling": args.sampling,
        "sampling_step": args.sampling_step,
        "evaluation_stride": args.evaluation_stride,
        "seed": args.seed,
    }
    config.update({name: value for name, value in overrides.items() if value is not None})
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
        config["eval_batch_size"] = args.batch_size
    return EasyDict(config)


def write_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).sort_values("heldout_scene").to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.results_csv is None:
        result_name = "mid_all_folds.csv" if len(args.scenes) > 1 else f"mid_{args.scenes[0]}_result.csv"
        args.results_csv = args.output_dir / result_name
    else:
        args.results_csv = args.results_csv.resolve()

    if str(MID_ROOT) not in sys.path:
        sys.path.insert(0, str(MID_ROOT))
    from mid import MID

    with args.config.open(encoding="utf-8") as config_file:
        base_config = yaml.safe_load(config_file)

    scenes = tuple(args.scenes)
    validate_data_dir(args.data_dir, scenes)
    existing_results = []
    completed_scenes = set()
    if args.skip_completed and args.results_csv.exists():
        existing_results = pd.read_csv(args.results_csv).to_dict("records")
        completed_scenes = {row["heldout_scene"] for row in existing_results}

    new_results = []
    for scene in scenes:
        if scene in completed_scenes:
            print(f"Skipping completed fold: {scene}")
            continue

        config = build_config(base_config, args, scene)
        seed_everything(config.seed)
        print(f"\n===== MID fold: hold out {scene}; device={config.device} =====")
        agent = MID(config)
        _, result = agent.train()
        new_results.append(result)
        write_results(existing_results + new_results, args.results_csv)
        print(f"Wrote {args.results_csv}")
        del agent

    if not new_results:
        print("No new folds were run.")
    elif len(existing_results) + len(new_results) > 1:
        print("\nBenchmark results:")
        print(pd.DataFrame(existing_results + new_results).sort_values("heldout_scene").to_string(index=False))


if __name__ == "__main__":
    main()
