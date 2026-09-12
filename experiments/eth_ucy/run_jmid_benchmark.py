#!/usr/bin/env python3
"""Run validation-selected faithful JMID ETH/UCY folds without Jupyter."""

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
JMID_ROOT = (
    PROJECT_ROOT
    / "trajectory_prediction"
    / "safe-interactive-crowdnav"
    / "sicnav_diffusion"
    / "JMID"
    / "MID"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation-selected JMID ETH/UCY benchmark folds."
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "experiments" / "eth_ucy" / "jmid_config.yaml")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "experiments" / "eth_ucy" / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "eth_ucy" / "outputs" / "jmid")
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, cuda:0, mps, or cpu.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--sampling", choices=("ddpm", "ddim"))
    parser.add_argument("--sampling-step", type=int, help="Number of reverse DDPM/DDIM steps.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name", default="jmid")
    parser.add_argument("--results-csv", type=Path)
    parser.add_argument("--skip-completed", action="store_true")
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
        raise FileNotFoundError("Missing processed ETH/UCY pickle(s):\n" + "\n".join(f"  - {path}" for path in missing))


def build_config(base: dict, args: argparse.Namespace, scene: str, *, evaluation: bool, epoch: int | None = None) -> EasyDict:
    config = deepcopy(base)
    config.update(
        {
            "dataset": scene,
            "inference_dataset": scene,
            "exp_name": f"{args.run_name}_{scene}",
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "device": args.device,
            "eval_mode": evaluation,
            "selection_split": "val",
            "test_split": "test",
        }
    )
    for name, value in {
        "epochs": args.epochs,
        "num_samples": args.num_samples,
        "sampling": args.sampling,
        "num_steps": args.sampling_step,
        "seed": args.seed,
    }.items():
        if value is not None:
            config[name] = value
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if epoch is not None:
        config["eval_at"] = epoch
    return EasyDict(config)


def write_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).sort_values("heldout_scene").to_csv(path, index=False)


def run_benchmark(args: argparse.Namespace) -> list[dict]:
    """Train, select, and test requested folds; usable from Python or the CLI."""
    args.config = args.config.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.results_csv = (args.results_csv or args.output_dir / ("jmid_all_folds.csv" if len(args.scenes) > 1 else f"jmid_{args.scenes[0]}_result.csv")).resolve()
    if not JMID_ROOT.is_dir():
        raise FileNotFoundError(f"Official JMID submodule is missing: {JMID_ROOT}")
    if str(JMID_ROOT) not in sys.path:
        sys.path.insert(0, str(JMID_ROOT))
    from joint_pred_mid import JointPredMID

    with args.config.open(encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)
    scenes = tuple(args.scenes)
    validate_data_dir(args.data_dir, scenes)
    existing = pd.read_csv(args.results_csv).to_dict("records") if args.skip_completed and args.results_csv.exists() else []
    completed = {row["heldout_scene"] for row in existing}
    new_results: list[dict] = []

    for scene in scenes:
        if scene in completed:
            print(f"Skipping completed fold: {scene}")
            continue
        train_config = build_config(base_config, args, scene, evaluation=False)
        seed_everything(train_config.seed)
        print(f"\n===== JMID fold: hold out {scene}; device={train_config.device} =====")
        trainer = JointPredMID(train_config, test_dataset=scene)
        selection = trainer.train(train_config.sampling, train_config.num_steps, train_config.seed)
        del trainer

        test_config = build_config(base_config, args, scene, evaluation=True, epoch=selection["best_epoch"])
        evaluator = JointPredMID(test_config)
        ade, fde, kde, _, _, sade, sfde, sade_mean, sfde_mean = evaluator.eval(
            test_config.sampling, test_config.num_steps
        )
        row = {
            "heldout_scene": scene,
            "device": str(evaluator.device),
            "best_epoch": selection["best_epoch"],
            "best_validation_ade_m": selection["best_validation_ade_m"],
            "test_ade_m": ade,
            "test_fde_m": fde,
            "test_sade_m": sade,
            "test_sfde_m": sfde,
            "test_sade_mean_m": sade_mean,
            "test_sfde_mean_m": sfde_mean,
            "test_kde_nll": kde,
            "sampling": test_config.sampling,
            "sampling_steps": test_config.num_steps,
            "num_samples": test_config.num_samples,
        }
        new_results.append(row)
        write_results(existing + new_results, args.results_csv)
        print(f"Wrote {args.results_csv}")
        del evaluator

    if new_results:
        print("\nBenchmark results:")
        print(pd.DataFrame(existing + new_results).sort_values("heldout_scene").to_string(index=False))
    else:
        print("No new folds were run.")
    return existing + new_results


def main() -> None:
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
