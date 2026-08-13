"""Training loop for DQN."""

import time

import gymnasium as gym
import numpy as np

from agents.dqn import DQNAgent
from config import CommonConfig, DQNConfig
from utils.replay_buffer import ReplayBuffer
from utils.seed import choose_device, seed_everything


def train_dqn(common: CommonConfig, config: DQNConfig):
    seed_everything(common.seed)
    device = choose_device()
    env = gym.make(common.env_id, max_episode_steps=common.max_episode_steps)
    env.action_space.seed(common.seed)
    observation_dim = int(np.prod(env.observation_space.shape))
    agent = DQNAgent(config, observation_dim, env.action_space.n, common.seed, device)
    replay = ReplayBuffer(config.replay_capacity, observation_dim, common.seed)
    records: list[dict] = []
    steps = 0
    episode = 0
    started = time.perf_counter()
    while steps < common.interaction_budget:
        observation, _ = env.reset(seed=common.seed + episode)
        episode_return = 0.0
        episode_length = 0
        losses: list[float] = []
        terminated = truncated = False
        while not (terminated or truncated) and steps < common.interaction_budget:
            epsilon = agent.epsilon(steps)
            action = agent.act(observation, epsilon)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            replay.add(observation, action, reward, next_observation, terminated)
            observation = next_observation
            steps += 1
            episode_length += 1
            episode_return += reward
            if (
                steps >= config.learning_starts
                and steps % config.train_frequency == 0
                and len(replay) >= config.batch_size
            ):
                losses.append(agent.learn(replay.sample(config.batch_size)))
            if steps % config.target_sync_frequency == 0:
                agent.sync_target()
        if terminated or truncated:
            records.append({
                "algorithm": "dqn", "seed": common.seed, "episode": episode,
                "environment_steps": steps, "episode_return": episode_return,
                "episode_length": episode_length, "success": int(terminated),
                "epsilon": agent.epsilon(steps),
                "loss": float(np.mean(losses)) if losses else np.nan,
                "elapsed_seconds": time.perf_counter() - started,
            })
        episode += 1
    env.close()
    return agent, records

