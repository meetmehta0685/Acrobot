"""Training loop for tabular Q-Learning."""

import time

import gymnasium as gym
import numpy as np

from agents.q_learning import QLearningAgent
from config import CommonConfig, QLearningConfig
from utils.seed import seed_everything


def train_q_learning(common: CommonConfig, config: QLearningConfig):
    seed_everything(common.seed)
    env = gym.make(common.env_id, max_episode_steps=common.max_episode_steps)
    env.action_space.seed(common.seed)
    agent = QLearningAgent(config, env.action_space.n, common.seed)
    records: list[dict] = []
    steps = 0
    episode = 0
    started = time.perf_counter()
    while steps < common.interaction_budget:
        observation, _ = env.reset(seed=common.seed + episode)
        episode_return = 0.0
        episode_length = 0
        terminated = truncated = False
        td_errors: list[float] = []
        while not (terminated or truncated) and steps < common.interaction_budget:
            epsilon = agent.epsilon(steps)
            action = agent.act(observation, epsilon)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            td_errors.append(agent.update(observation, action, reward, next_observation, terminated))
            observation = next_observation
            steps += 1
            episode_length += 1
            episode_return += reward
        if terminated or truncated:
            records.append({
                "algorithm": "q_learning", "seed": common.seed, "episode": episode,
                "environment_steps": steps, "episode_return": episode_return,
                "episode_length": episode_length, "success": int(terminated),
                "epsilon": agent.epsilon(steps),
                "loss": float(np.mean(np.abs(td_errors))),
                "elapsed_seconds": time.perf_counter() - started,
            })
        episode += 1
    env.close()
    return agent, records

