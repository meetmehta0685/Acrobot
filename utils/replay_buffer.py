"""A fixed-size NumPy replay buffer for DQN."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, seed: int):
        self.capacity = capacity
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, observation, action, reward, next_observation, terminated) -> None:
        index = self.position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.terminated[index] = terminated
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if batch_size > self.size:
            raise ValueError("Cannot sample more transitions than are stored")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return ReplayBatch(
            self.observations[indices], self.actions[indices], self.rewards[indices],
            self.next_observations[indices], self.terminated[indices]
        )

    def __len__(self) -> int:
        return self.size

