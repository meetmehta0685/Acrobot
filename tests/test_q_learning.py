import numpy as np

from agents.q_learning import QLearningAgent
from config import QLearningConfig


def test_q_learning_bellman_update_and_terminal_mask() -> None:
    config = QLearningConfig(learning_rate=0.5, gamma=0.9, bins=(2,) * 6)
    agent = QLearningAgent(config, action_count=3, seed=0)
    observation = np.zeros(6)
    next_observation = np.ones(6) * 0.5
    next_state = agent.discretizer.encode(next_observation)
    agent.q_table[next_state + (1,)] = 10.0
    agent.update(observation, 0, 2.0, next_observation, terminated=False)
    assert np.isclose(agent.q_table[agent.discretizer.encode(observation) + (0,)], 5.5)

    agent.q_table[agent.discretizer.encode(observation) + (2,)] = 0.0
    agent.update(observation, 2, 2.0, next_observation, terminated=True)
    assert np.isclose(agent.q_table[agent.discretizer.encode(observation) + (2,)], 1.0)

