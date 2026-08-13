"""On-policy rollout collection and PPO training loop."""

import time

import gymnasium as gym
import numpy as np

from agents.ppo import PPOAgent, compute_gae
from config import CommonConfig, PPOConfig
from utils.seed import choose_device, seed_everything


def train_ppo(common: CommonConfig, config: PPOConfig):
    seed_everything(common.seed)
    env = gym.make(common.env_id, max_episode_steps=common.max_episode_steps)
    env.action_space.seed(common.seed)
    observation_dim = int(np.prod(env.observation_space.shape))
    agent = PPOAgent(config, observation_dim, env.action_space.n, common.seed, choose_device())
    records: list[dict] = []
    steps = 0
    episode = 0
    episode_return = 0.0
    episode_length = 0
    started = time.perf_counter()
    observation, _ = env.reset(seed=common.seed)

    while steps < common.interaction_budget:
        rollout_size = min(config.rollout_steps, common.interaction_budget - steps)
        data: dict[str, list] = {key: [] for key in (
            "observations", "actions", "rewards", "log_probs", "values",
            "next_values", "terminated", "episode_done"
        )}
        completed_record_indices: list[int] = []
        for _ in range(rollout_size):
            action, log_prob, value = agent.act(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_value = 0.0 if terminated else agent.value(next_observation)
            data["observations"].append(observation)
            data["actions"].append(action)
            data["rewards"].append(reward)
            data["log_probs"].append(log_prob)
            data["values"].append(value)
            data["next_values"].append(next_value)
            data["terminated"].append(float(terminated))
            data["episode_done"].append(float(done))
            observation = next_observation
            steps += 1
            episode_return += reward
            episode_length += 1
            if done:
                records.append({
                    "algorithm": "ppo", "seed": common.seed, "episode": episode,
                    "environment_steps": steps, "episode_return": episode_return,
                    "episode_length": episode_length, "success": int(terminated),
                    "epsilon": np.nan, "loss": np.nan,
                    "elapsed_seconds": time.perf_counter() - started,
                })
                completed_record_indices.append(len(records) - 1)
                episode += 1
                episode_return = 0.0
                episode_length = 0
                observation, _ = env.reset(seed=common.seed + episode)

        arrays = {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}
        advantages, returns = compute_gae(
            arrays["rewards"], arrays["values"], arrays["next_values"],
            arrays["terminated"], arrays["episode_done"], config.gamma, config.gae_lambda
        )
        rollout = {
            "observations": arrays["observations"], "actions": arrays["actions"],
            "log_probs": arrays["log_probs"], "advantages": advantages, "returns": returns,
        }
        update_metrics = agent.update(rollout)
        for index in completed_record_indices:
            records[index]["loss"] = update_metrics["policy_loss"]
    env.close()
    return agent, records

