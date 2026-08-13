"""Deep Q-Network with replay and a target network, implemented in PyTorch."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import DQNConfig
from utils.replay_buffer import ReplayBatch


def make_mlp(input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for size in hidden_sizes:
        layers.extend((nn.Linear(previous, size), nn.ReLU()))
        previous = size
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class DQNAgent:
    def __init__(
        self, config: DQNConfig, observation_dim: int, action_count: int,
        seed: int, device: torch.device,
    ):
        self.config = config
        self.observation_dim = observation_dim
        self.action_count = action_count
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.online = make_mlp(observation_dim, action_count, config.hidden_sizes).to(device)
        self.target = make_mlp(observation_dim, action_count, config.hidden_sizes).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)

    def epsilon(self, environment_steps: int) -> float:
        fraction = min(environment_steps / max(self.config.epsilon_decay_steps, 1), 1.0)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_min - self.config.epsilon_start
        )

    @torch.no_grad()
    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_count))
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(tensor).argmax(dim=1).item())

    def learn(self, batch: ReplayBatch) -> float:
        observations = torch.as_tensor(batch.observations, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        next_observations = torch.as_tensor(batch.next_observations, device=self.device)
        terminated = torch.as_tensor(batch.terminated, device=self.device)

        predicted = self.online(observations).gather(1, actions).squeeze(1)
        with torch.no_grad():
            bootstrap_network = self.target if self.config.use_target_network else self.online
            next_values = bootstrap_network(next_observations).max(dim=1).values
            targets = rewards + self.config.gamma * (1.0 - terminated) * next_values
        loss = nn.functional.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        if self.config.use_target_network:
            self.target.load_state_dict(self.online.state_dict())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.online.state_dict()}, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(checkpoint["model"])
        self.sync_target()
