"""Tabular Q-Learning implemented from the Bellman update."""

from pathlib import Path

import numpy as np

from config import QLearningConfig
from utils.discretizer import AcrobotDiscretizer


class QLearningAgent:
    def __init__(self, config: QLearningConfig, action_count: int, seed: int):
        self.config = config
        self.action_count = action_count
        self.discretizer = AcrobotDiscretizer(config.bins)
        self.q_table = np.zeros((*config.bins, action_count), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    def epsilon(self, environment_steps: int) -> float:
        fraction = min(environment_steps / max(self.config.epsilon_decay_steps, 1), 1.0)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_min - self.config.epsilon_start
        )

    def act(
        self, observation: np.ndarray, epsilon: float = 0.0,
        deterministic_ties: bool = False,
    ) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_count))
        state = self.discretizer.encode(observation)
        values = self.q_table[state]
        if deterministic_ties:
            return int(np.argmax(values))
        # Random tie-breaking avoids always preferring action zero initially.
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> float:
        """Apply Q(s,a) <- Q(s,a) + alpha [r + gamma max Q(s',a') - Q]."""
        state = self.discretizer.encode(observation)
        next_state = self.discretizer.encode(next_observation)
        current = float(self.q_table[state + (action,)])
        bootstrap = 0.0 if terminated else float(self.q_table[next_state].max())
        target = reward + self.config.gamma * bootstrap
        td_error = target - current
        self.q_table[state + (action,)] = current + self.config.learning_rate * td_error
        return td_error

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q_table=self.q_table, bins=np.asarray(self.config.bins))

    def load(self, path: Path) -> None:
        data = np.load(path)
        if tuple(data["bins"].tolist()) != self.config.bins:
            raise ValueError("Checkpoint bins do not match the configured discretizer")
        self.q_table = data["q_table"].astype(np.float32)
