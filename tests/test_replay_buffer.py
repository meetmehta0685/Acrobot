import numpy as np

from utils.replay_buffer import ReplayBuffer


def test_replay_buffer_overwrites_oldest_transition() -> None:
    buffer = ReplayBuffer(capacity=3, observation_dim=1, seed=0)
    for value in range(5):
        observation = np.array([value], dtype=np.float32)
        buffer.add(observation, 0, float(value), observation + 1, False)
    assert len(buffer) == 3
    assert set(buffer.observations[:3, 0].tolist()) == {2.0, 3.0, 4.0}

