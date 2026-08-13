"""A common, exploration-free evaluation protocol for every algorithm."""

from typing import Callable

import gymnasium as gym

from config import CommonConfig


def evaluate_policy(
    policy: Callable, algorithm: str, common: CommonConfig, render_mode: str | None = None
) -> list[dict]:
    env = gym.make(
        common.env_id, max_episode_steps=common.max_episode_steps, render_mode=render_mode
    )
    records: list[dict] = []
    for episode in range(common.eval_episodes):
        observation, _ = env.reset(seed=10_000 + common.seed + episode)
        terminated = truncated = False
        episode_return = 0.0
        episode_length = 0
        while not (terminated or truncated):
            action = policy(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            episode_length += 1
        records.append({
            "algorithm": algorithm, "seed": common.seed, "episode": episode,
            "episode_return": episode_return, "episode_length": episode_length,
            "success": int(terminated),
        })
    env.close()
    return records

