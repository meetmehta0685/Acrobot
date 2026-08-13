import numpy as np
import torch

from agents.dqn import DQNAgent
from config import DQNConfig
from utils.replay_buffer import ReplayBatch


def test_dqn_learning_step_changes_parameters() -> None:
    config = DQNConfig(hidden_sizes=(16,), batch_size=4)
    agent = DQNAgent(config, 6, 3, seed=0, device=torch.device("cpu"))
    batch = ReplayBatch(
        observations=np.zeros((4, 6), dtype=np.float32),
        actions=np.array([0, 1, 2, 0]),
        rewards=np.full(4, -1.0, dtype=np.float32),
        next_observations=np.ones((4, 6), dtype=np.float32),
        terminated=np.array([0, 0, 1, 0], dtype=np.float32),
    )
    before = [parameter.detach().clone() for parameter in agent.online.parameters()]
    loss = agent.learn(batch)
    assert loss >= 0
    assert any(not torch.equal(old, new) for old, new in zip(before, agent.online.parameters()))

