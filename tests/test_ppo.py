import numpy as np
import torch

from agents.ppo import PPOAgent, compute_gae
from config import PPOConfig


def test_gae_masks_terminal_and_episode_boundary() -> None:
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.5], dtype=np.float32)
    next_values = np.array([0.5, 99.0], dtype=np.float32)
    terminated = np.array([0.0, 1.0], dtype=np.float32)
    done = np.array([0.0, 1.0], dtype=np.float32)
    advantages, returns = compute_gae(rewards, values, next_values, terminated, done, 1.0, 1.0)
    np.testing.assert_allclose(advantages, [1.5, 0.5])
    np.testing.assert_allclose(returns, [2.0, 1.0])


def test_ppo_update_runs_and_is_finite() -> None:
    config = PPOConfig(hidden_sizes=(16,), optimization_epochs=2, minibatch_size=4)
    agent = PPOAgent(config, 6, 3, seed=0, device=torch.device("cpu"))
    rollout = {
        "observations": np.zeros((8, 6), dtype=np.float32),
        "actions": np.zeros(8, dtype=np.int64),
        "log_probs": np.full(8, -np.log(3), dtype=np.float32),
        "advantages": np.arange(8, dtype=np.float32),
        "returns": np.ones(8, dtype=np.float32),
    }
    metrics = agent.update(rollout)
    assert all(np.isfinite(value) for value in metrics.values())

