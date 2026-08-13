import numpy as np
import torch

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from config import DQNConfig, PPOConfig
from utils.replay_buffer import ReplayBatch


def test_dqn_without_target_network_can_update() -> None:
    agent = DQNAgent(
        DQNConfig(hidden_sizes=(8,), use_target_network=False),
        6, 3, seed=0, device=torch.device("cpu"),
    )
    batch = ReplayBatch(
        observations=np.zeros((4, 6), dtype=np.float32),
        actions=np.zeros(4, dtype=np.int64),
        rewards=np.full(4, -1.0, dtype=np.float32),
        next_observations=np.ones((4, 6), dtype=np.float32),
        terminated=np.zeros(4, dtype=np.float32),
    )
    assert np.isfinite(agent.learn(batch))


def test_ppo_without_clipping_can_update() -> None:
    agent = PPOAgent(
        PPOConfig(
            hidden_sizes=(8,), use_clipping=False,
            optimization_epochs=1, minibatch_size=4,
        ),
        6, 3, seed=0, device=torch.device("cpu"),
    )
    metrics = agent.update({
        "observations": np.zeros((4, 6), dtype=np.float32),
        "actions": np.zeros(4, dtype=np.int64),
        "log_probs": np.full(4, -np.log(3), dtype=np.float32),
        "advantages": np.arange(4, dtype=np.float32),
        "returns": np.ones(4, dtype=np.float32),
    })
    assert all(np.isfinite(value) for value in metrics.values())

