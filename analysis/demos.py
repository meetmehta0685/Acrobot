"""Render the best completed checkpoint for each algorithm as an animated GIF."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from agents import DQNAgent, PPOAgent, QLearningAgent
from config import DQNConfig, PPOConfig, QLearningConfig
from experiments.results import RESULTS_ROOT, load_group
from utils.seed import seed_everything


def _params(data: dict) -> dict:
    converted = dict(data)
    for key in ("bins", "hidden_sizes"):
        if key in converted:
            converted[key] = tuple(converted[key])
    return converted


def _load_policy(result, root: Path):
    spec = result.spec
    method = result.method
    seed_everything(result.seed)
    checkpoint = root / "models" / result.key
    if method == "q_learning":
        agent = QLearningAgent(
            replace(QLearningConfig(), **_params(spec["params"])), 3, result.seed
        )
        agent.load(checkpoint.with_suffix(".npz"))
        return lambda observation: agent.act(observation, deterministic_ties=True)
    if method == "dqn":
        agent = DQNAgent(
            replace(DQNConfig(), **_params(spec["params"])), 6, 3,
            result.seed, torch.device("cpu"),
        )
        agent.load(checkpoint.with_suffix(".pt"))
        return lambda observation: agent.act(observation)
    agent = PPOAgent(
        replace(PPOConfig(), **_params(spec["params"])), 6, 3,
        result.seed, torch.device("cpu"),
    )
    agent.load(checkpoint.with_suffix(".pt"))
    return lambda observation: agent.act(observation, deterministic=True)[0]


def _best_result(method: str, root: Path):
    candidates = [result for result in load_group("main", root) if result.method == method]
    if not candidates:
        raise FileNotFoundError(
            f"No completed main-study checkpoint for {method}. Run the matrix first."
        )
    return max(candidates, key=lambda result: result.final_mean)


def show_demo(method: str, root: Path = RESULTS_ROOT) -> None:
    """Open a window and play the strongest completed seed for one method."""
    best = _best_result(method, root)
    policy = _load_policy(best, root)
    spec = best.spec
    env = gym.make(
        spec["env_id"], max_episode_steps=spec["max_episode_steps"],
        render_mode="human",
    )
    observation, _ = env.reset(seed=90_000)
    terminated = truncated = False
    while not (terminated or truncated):
        observation, _, terminated, truncated, _ = env.step(policy(observation))
    env.close()


def generate_demos(root: Path = RESULTS_ROOT) -> list[Path]:
    output = root / "demos"
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for method in ("q_learning", "dqn", "ppo"):
        try:
            best = _best_result(method, root)
        except FileNotFoundError:
            continue
        policy = _load_policy(best, root)
        spec = best.spec
        env = gym.make(
            spec["env_id"], max_episode_steps=spec["max_episode_steps"],
            render_mode="rgb_array",
        )
        observation, _ = env.reset(seed=90_000)
        frames = [env.render()]
        terminated = truncated = False
        step = 0
        while not (terminated or truncated):
            observation, _, terminated, truncated, _ = env.step(policy(observation))
            step += 1
            if step % 2 == 0 or terminated or truncated:
                frames.append(env.render())
        env.close()
        path = output / f"{method}_demo.gif"
        images = [Image.fromarray(np.asarray(frame)) for frame in frames]
        images[0].save(
            path, save_all=True, append_images=images[1:], duration=40,
            loop=0, optimize=True,
        )
        paths.append(path)
    return paths
