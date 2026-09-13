"""Shared construction and evaluation utilities for CrowdSimPlus policies."""

from __future__ import annotations

import configparser
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

try:
    from .baseline import EpisodeResult, plot_episode, run_episode, summarize
except ImportError:  # Notebook execution with this directory on sys.path.
    from baseline import EpisodeResult, plot_episode, run_episode, summarize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SICNAV_ROOT = PROJECT_ROOT / "trajectory_prediction" / "safe-interactive-crowdnav"
DEFAULT_ENV_CONFIG = SICNAV_ROOT / "sicnav_diffusion" / "configs" / "env.config"
DEFAULT_POLICY_CONFIG = SICNAV_ROOT / "sicnav_diffusion" / "configs" / "policy.config"
JMID_CHECKPOINT = (
    SICNAV_ROOT
    / "sicnav_diffusion/JMID/MID/checkpoints/sim_inference_checkpoints"
    / "sim_gen_sicnav_p_midjp_cvg_epoch121.pt"
)

BUILTIN_SCENARIOS = (
    "square_crossing",
    "circle_crossing",
    "hallway",
    "hallway_static",
    "hallway_bottleneck",
    "hallway_squeeze",
    "rectangle",
    "hallway_static_with_back",
    "left_wall",
    "no_walls",
)

METHODS = ("linear", "orca_plus", "dwa", "mpc_cvmm", "sicnav_cvg", "sicnav_jmid")

if str(SICNAV_ROOT) not in sys.path:
    sys.path.insert(0, str(SICNAV_ROOT))

from crowd_sim_plus.envs.crowd_sim_plus import CrowdSimPlus
from crowd_sim_plus.envs.utils.robot_plus import Robot


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _read_config(path: str | Path) -> configparser.RawConfigParser:
    config = configparser.RawConfigParser()
    if not config.read(Path(path)):
        raise FileNotFoundError(f"Could not read config: {path}")
    return config


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve an inference device without assuming CUDA is installed."""
    if device != "auto":
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        if requested.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false.")
        return requested
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dependency_report(method: str) -> pd.Series:
    """Report optional native and Python dependencies before a long run."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")
    return pd.Series(
        {
            "method": method,
            "rvo2": importlib.util.find_spec("rvo2") is not None,
            "casadi": importlib.util.find_spec("casadi") is not None,
            "acados_template": importlib.util.find_spec("acados_template") is not None,
            "ACADOS_SOURCE_DIR": bool(os.environ.get("ACADOS_SOURCE_DIR")),
            "checkpoint": JMID_CHECKPOINT.is_file(),
        }
    )


def require_dependencies(method: str) -> None:
    report = dependency_report(method)
    required = {"rvo2"}
    if method in {"mpc_cvmm", "sicnav_cvg", "sicnav_jmid"}:
        required.add("casadi")
    if method in {"sicnav_cvg", "sicnav_jmid"}:
        required.update({"acados_template", "ACADOS_SOURCE_DIR"})
    if method == "sicnav_jmid":
        required.add("checkpoint")
    missing = [name for name in required if not report[name]]
    if missing:
        raise RuntimeError(
            f"{method} is not ready; missing {', '.join(missing)}. "
            "Run dependency_report(method) and follow experiments/crowd_navigation/README.md."
        )


def inspect_jmid_checkpoint(path: str | Path = JMID_CHECKPOINT) -> pd.Series:
    """Load the trusted released checkpoint on CPU and summarize its contract."""
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    mid_root = SICNAV_ROOT / "sicnav_diffusion/JMID/MID"
    if str(mid_root) not in sys.path:
        sys.path.insert(0, str(mid_root))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"encoder", "ddpm"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")
    return pd.Series(
        {
            "path": str(checkpoint_path),
            "keys": ", ".join(checkpoint.keys()),
            "encoder_modules": len(checkpoint["encoder"]),
            "ddpm_tensors": len(checkpoint["ddpm"]),
            "torch_version": torch.__version__,
            "device": "cpu (validation only)",
        }
    )


def _configure_method(
    method: str,
    env_config: configparser.RawConfigParser,
    policy_config: configparser.RawConfigParser,
):
    if method == "orca_plus":
        from crowd_sim_plus.envs.policy.orca_plus import ORCAPlus

        return ORCAPlus(), "orca_plus"
    if method == "linear":
        from crowd_sim_plus.envs.policy.linear import Linear

        return Linear(), "linear"
    if method == "dwa":
        from sicnav.policy.dwa import DynamicWindowApproach

        return DynamicWindowApproach(), "dwa"
    if method == "mpc_cvmm":
        from sicnav.policy.campc import CollisionAvoidMPC

        policy_config.set("mpc_env", "hum_model", "cvmm")
        return CollisionAvoidMPC(), "campc"
    if method in {"sicnav_cvg", "sicnav_jmid"}:
        from sicnav_diffusion.policy.sicnav_acados import SICNavAcados

        policy_config.set("mpc_env", "hum_model", "orca_casadi_kkt")
        is_jmid = method == "sicnav_jmid"
        policy_config.set("campc", "human_goal_cvmm", str(not is_jmid).lower())
        policy_config.set("campc", "human_pred_MID", str(is_jmid).lower())
        # The released policy validates this selector even for CVG, where no
        # MID forecaster is constructed. Keep the official joint default.
        policy_config.set("campc", "human_pred_MID_joint", "true")
        policy_config.set("campc", "human_pred_MID_vanil_as_joint", "false")
        return SICNavAcados(), "sicnav_acados"
    raise ValueError(f"Unknown method {method!r}; choose from {METHODS}.")


def make_navigation_environment(
    method: str,
    *,
    scenario: str = "hallway_bottleneck",
    human_count: int = 3,
    human_policy: str = "orca_plus",
    starts_moving: int = 10,
    device: str = "auto",
    randomize_attributes: bool = True,
    env_config_path: str | Path = DEFAULT_ENV_CONFIG,
    policy_config_path: str | Path = DEFAULT_POLICY_CONFIG,
) -> tuple[CrowdSimPlus, Robot]:
    """Build one official controller using in-memory config overrides."""
    if scenario not in BUILTIN_SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {BUILTIN_SCENARIOS}.")
    require_dependencies(method)
    env_config = _read_config(env_config_path)
    policy_config = _read_config(policy_config_path)
    env_config.set("robot", "policy", "linear")
    env_config.set("humans", "policy", human_policy)
    env_config.set("sim", "test_sim", scenario)
    env_config.set("sim", "human_num", str(human_count))
    env_config.set("sim", "starts_moving", str(starts_moving))
    env_config.set("env", "randomize_attributes", str(randomize_attributes).lower())

    policy, policy_name = _configure_method(method, env_config, policy_config)
    env = CrowdSimPlus()
    env.configure(env_config)
    robot = Robot(env_config, "robot")
    robot.set_policy(policy)
    env.set_robot(robot)
    policy.configure(policy_config)
    policy.set_phase("test")
    policy.set_device(resolve_device(device))
    if method == "dwa":
        policy.time_step = env.time_step
        policy.configure_dwa(policy_config, env_config)
    with _working_directory(SICNAV_ROOT):
        policy.set_env(env)
    robot.policy_label = policy_name
    return env, robot


def run_navigation_suite(
    method: str,
    *,
    cases: Iterable[int] = range(10),
    output_dir: str | Path | None = None,
    resume: bool = False,
    **environment_kwargs,
) -> tuple[pd.DataFrame, dict[int, EpisodeResult]]:
    """Run deterministic cases and optionally checkpoint a tidy result CSV."""
    requested_cases = list(dict.fromkeys(int(case) for case in cases))
    scenario = environment_kwargs.get("scenario", "hallway_bottleneck")
    human_count = environment_kwargs.get("human_count", 3)
    result_path = None
    existing = pd.DataFrame()
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_path = output_path / f"{method}__{scenario}__n{human_count}.csv"
        if resume and result_path.is_file():
            existing = pd.read_csv(result_path)

    completed_cases = set(existing.get("test_case", pd.Series(dtype=int)).astype(int))
    pending_cases = [case for case in requested_cases if case not in completed_cases]
    if not pending_cases:
        print(f"[{method}] {scenario} n={human_count}: all requested cases already exist", flush=True)
        return existing[existing["test_case"].isin(requested_cases)].copy(), {}

    env, robot = make_navigation_environment(method, **environment_kwargs)
    rows: list[dict] = []
    episodes: dict[int, EpisodeResult] = {}
    for case in pending_cases:
        with _working_directory(SICNAV_ROOT):
            episode = run_episode(env, robot, test_case=case)
        episode.metrics["method"] = method
        episode.metrics["human_count"] = env.human_num
        episode.metrics["starts_moving"] = environment_kwargs.get("starts_moving", 10)
        episode.metrics["randomize_attributes"] = environment_kwargs.get("randomize_attributes", True)
        rows.append(episode.metrics)
        episodes[case] = episode
        print(
            f"[{method}] {env.test_sim} case={case} "
            f"success={episode.metrics['success']} collisions={episode.metrics['collision_steps']} "
            f"time={episode.metrics['navigation_time_s']:.2f}s",
            flush=True,
        )
        if result_path is not None:
            checkpoint = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            checkpoint = checkpoint.drop_duplicates("test_case", keep="last").sort_values("test_case")
            temporary_path = result_path.with_suffix(".tmp")
            checkpoint.to_csv(temporary_path, index=False)
            temporary_path.replace(result_path)

    results = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    results = results.drop_duplicates("test_case", keep="last").sort_values("test_case")
    return results[results["test_case"].isin(requested_cases)].copy(), episodes


def load_results(output_dir: str | Path) -> pd.DataFrame:
    """Load all policy/scenario CSVs produced by run_navigation_suite."""
    files = sorted(Path(output_dir).glob("*__*__n*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True)


__all__ = [
    "BUILTIN_SCENARIOS",
    "DEFAULT_ENV_CONFIG",
    "DEFAULT_POLICY_CONFIG",
    "JMID_CHECKPOINT",
    "METHODS",
    "dependency_report",
    "inspect_jmid_checkpoint",
    "load_results",
    "make_navigation_environment",
    "plot_episode",
    "resolve_device",
    "run_navigation_suite",
    "summarize",
]
