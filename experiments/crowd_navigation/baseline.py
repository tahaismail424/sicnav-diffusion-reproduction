"""Small, solver-free CrowdSimPlus baseline used before SICNav integration."""

from __future__ import annotations

import configparser
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SICNAV_ROOT = PROJECT_ROOT / "trajectory_prediction" / "safe-interactive-crowdnav"
DEFAULT_ENV_CONFIG = SICNAV_ROOT / "sicnav_diffusion" / "configs" / "env.config"

if str(SICNAV_ROOT) not in sys.path:
    sys.path.insert(0, str(SICNAV_ROOT))

from crowd_sim_plus.envs.crowd_sim_plus import CrowdSimPlus
from crowd_sim_plus.envs.utils.action import ActionRot, ActionXY
from crowd_sim_plus.envs.utils.robot_plus import Robot


@dataclass
class EpisodeResult:
    metrics: dict[str, float | int | bool | str]
    robot_positions: np.ndarray
    human_positions: np.ndarray
    robot_goal: np.ndarray
    static_obstacles: list


def policy_is_available(policy: str) -> bool:
    """Return whether an optional policy's external dependency is installed."""
    if policy in {"orca", "orca_plus"}:
        return importlib.util.find_spec("rvo2") is not None
    if policy == "SB3":
        return importlib.util.find_spec("stable_baselines3") is not None
    return True


def make_environment(
    *,
    robot_policy: str = "linear",
    human_policy: str = "sfm",
    scenario: str = "hallway",
    human_count: int = 3,
    starts_moving: int = 10,
    config_path: str | Path = DEFAULT_ENV_CONFIG,
) -> tuple[CrowdSimPlus, Robot]:
    """Construct CrowdSimPlus without requiring JMID, CasADi, or Acados."""
    for policy in (robot_policy, human_policy):
        if not policy_is_available(policy):
            dependency = "rvo2" if policy in {"orca", "orca_plus"} else policy
            raise ImportError(
                f"Policy {policy!r} requires the optional {dependency!r} package. "
                "Use the linear/SFM baseline or install that dependency first."
            )

    config = configparser.RawConfigParser()
    loaded = config.read(Path(config_path))
    if not loaded:
        raise FileNotFoundError(f"Could not read CrowdSimPlus config: {config_path}")

    config.set("robot", "policy", robot_policy)
    config.set("humans", "policy", human_policy)
    config.set("sim", "test_sim", scenario)
    config.set("sim", "human_num", str(human_count))
    config.set("sim", "starts_moving", str(starts_moving))

    env = CrowdSimPlus()
    env.configure(config)
    robot = Robot(config, "robot")
    robot.policy.set_phase("test")
    env.set_robot(robot)
    return env, robot


def _event(info: dict, name: str) -> bool:
    return bool(info[name].val != 0)


def _positions(env: CrowdSimPlus) -> tuple[np.ndarray, np.ndarray]:
    robot_position = np.asarray(env.robot.get_position(), dtype=float)
    human_positions = np.asarray([human.get_position() for human in env.humans], dtype=float)
    return robot_position, human_positions


def run_episode(
    env: CrowdSimPlus,
    robot: Robot,
    *,
    test_case: int,
) -> EpisodeResult:
    """Run one legacy Gym-style episode and retain enough state to inspect it."""
    observation, static_obstacles = env.reset("test", test_case, return_stat=True)
    robot_start, human_start = _positions(env)
    robot_goal = np.asarray(robot.get_goal_position(), dtype=float)

    robot_trace = [robot_start]
    human_trace = [human_start]
    cumulative_reward = 0.0
    collision_steps = 0
    wall_collision_steps = 0
    danger_steps = 0
    frozen_steps = 0
    policy_times = []
    linear_speeds = []
    angular_rates = []
    minimum_center_distance = np.inf
    minimum_clearance = np.inf
    done = False
    info = None

    while not done:
        started = time.perf_counter()
        action = robot.act(observation, static_obstacles)
        policy_times.append(time.perf_counter() - started)
        if isinstance(action, ActionXY):
            linear_speeds.append(float(np.hypot(action.vx, action.vy)))
            if len(linear_speeds) == 1:
                angular_rates.append(0.0)
            else:
                previous_heading = np.arctan2(
                    robot_trace[-1][1] - robot_trace[-2][1],
                    robot_trace[-1][0] - robot_trace[-2][0],
                ) if len(robot_trace) > 1 else np.arctan2(action.vy, action.vx)
                heading = np.arctan2(action.vy, action.vx)
                heading_delta = np.arctan2(
                    np.sin(heading - previous_heading),
                    np.cos(heading - previous_heading),
                )
                angular_rates.append(float(heading_delta / env.time_step))
        else:
            linear_speeds.append(float(action.v))
            angular_rates.append(float(action.r / env.time_step))
        observation, reward, done, info = env.step(action)
        cumulative_reward += float(reward)

        robot_position, human_positions = _positions(env)
        robot_trace.append(robot_position)
        human_trace.append(human_positions)

        for human in env.humans:
            center_distance = np.linalg.norm(robot_position - np.asarray(human.get_position()))
            clearance = center_distance - robot.radius - human.radius
            minimum_center_distance = min(minimum_center_distance, center_distance)
            minimum_clearance = min(minimum_clearance, clearance)

        collision_steps += int(_event(info, "Collision"))
        wall_collision_steps += int(_event(info, "WallCollision"))
        danger_steps += int(_event(info, "Danger"))
        frozen_steps += int(_event(info, "Frozen"))

    assert info is not None
    robot_positions = np.asarray(robot_trace)
    segment_lengths = np.linalg.norm(np.diff(robot_positions, axis=0), axis=1)
    path_length = float(segment_lengths.sum())
    endpoint_displacement = float(np.linalg.norm(robot_positions[-1] - robot_start))
    success = _event(info, "ReachGoal")
    linear_accelerations = np.diff(linear_speeds) / env.time_step
    angular_accelerations = np.diff(angular_rates) / env.time_step

    metrics: dict[str, float | int | bool | str] = {
        "test_case": test_case,
        "robot_policy": robot.policy.__class__.__name__,
        "human_policy": env.humans[0].policy.__class__.__name__,
        "scenario": env.test_sim,
        "success": success,
        "timeout": _event(info, "Timeout"),
        "navigation_time_s": float(env.global_time),
        "steps": len(robot_trace) - 1,
        "collision_steps": collision_steps,
        "wall_collision_steps": wall_collision_steps,
        "danger_steps": danger_steps,
        "frozen_steps": frozen_steps,
        "minimum_center_distance_m": float(minimum_center_distance),
        "minimum_clearance_m": float(minimum_clearance),
        "path_length_m": path_length,
        "path_efficiency": endpoint_displacement / path_length if path_length > 0 else 0.0,
        "cumulative_reward": cumulative_reward,
        "mean_policy_time_ms": 1e3 * float(np.mean(policy_times)),
        "max_policy_time_ms": 1e3 * float(np.max(policy_times)),
        "mean_abs_linear_accel_mps2": float(np.mean(np.abs(linear_accelerations)))
        if linear_accelerations.size
        else 0.0,
        "mean_abs_angular_rate_radps": float(np.mean(np.abs(angular_rates))),
        "mean_abs_angular_accel_radps2": float(np.mean(np.abs(angular_accelerations)))
        if angular_accelerations.size
        else 0.0,
    }
    return EpisodeResult(
        metrics=metrics,
        robot_positions=robot_positions,
        human_positions=np.asarray(human_trace),
        robot_goal=robot_goal,
        static_obstacles=static_obstacles,
    )


def run_suite(*, cases=range(20), **environment_kwargs) -> tuple[pd.DataFrame, dict[int, EpisodeResult]]:
    """Run deterministic test cases and return per-episode results."""
    rows = []
    episodes = {}
    for test_case in cases:
        env, robot = make_environment(**environment_kwargs)
        episode = run_episode(env, robot, test_case=int(test_case))
        episodes[int(test_case)] = episode
        rows.append(episode.metrics)
    return pd.DataFrame(rows), episodes


def summarize(results: pd.DataFrame) -> pd.Series:
    """Aggregate the navigation outcomes used for baseline comparisons."""
    return pd.Series(
        {
            "episodes": len(results),
            "success_rate": results["success"].mean(),
            "timeout_rate": results["timeout"].mean(),
            "episodes_with_collision_rate": (results["collision_steps"] > 0).mean(),
            "episodes_with_wall_collision_rate": (results["wall_collision_steps"] > 0).mean(),
            "mean_navigation_time_s": results["navigation_time_s"].mean(),
            "mean_minimum_clearance_m": results["minimum_clearance_m"].mean(),
            "mean_path_efficiency": results["path_efficiency"].mean(),
            "mean_policy_time_ms": results["mean_policy_time_ms"].mean(),
            "mean_abs_linear_accel_mps2": results["mean_abs_linear_accel_mps2"].mean(),
            "mean_abs_angular_rate_radps": results["mean_abs_angular_rate_radps"].mean(),
            "mean_abs_angular_accel_radps2": results["mean_abs_angular_accel_radps2"].mean(),
        }
    )


def plot_episode(episode: EpisodeResult, *, title: str | None = None):
    """Plot the executed robot and human trajectories in world coordinates."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(
        episode.robot_positions[:, 0],
        episode.robot_positions[:, 1],
        color="black",
        linewidth=2.5,
        label="robot",
    )
    ax.scatter(*episode.robot_positions[0], color="black", marker="o", zorder=3)
    ax.scatter(*episode.robot_goal, color="goldenrod", marker="*", s=180, label="robot goal", zorder=3)

    for human_index in range(episode.human_positions.shape[1]):
        trace = episode.human_positions[:, human_index]
        ax.plot(trace[:, 0], trace[:, 1], linewidth=1.5, alpha=0.8, label=f"human {human_index}")
        ax.scatter(*trace[0], s=25, zorder=3)

    for obstacle_index, obstacle in enumerate(episode.static_obstacles):
        points = np.asarray(obstacle)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color="firebrick",
            linewidth=4,
            label="static obstacle" if obstacle_index == 0 else None,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title(title or f"CrowdSimPlus test case {episode.metrics['test_case']}")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    return fig, ax
