"""Clipped Proximal Policy Optimization for a discrete action space."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from config import PPOConfig
from agents.dqn import make_mlp


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_count: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self.actor = make_mlp(observation_dim, action_count, hidden_sizes)
        self.critic = make_mlp(observation_dim, 1, hidden_sizes)

    def distribution(self, observations: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(observations))

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    episode_done: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE, bootstrapping truncations but not crossing episode resets."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for index in reversed(range(len(rewards))):
        delta = rewards[index] + gamma * (1.0 - terminated[index]) * next_values[index] - values[index]
        gae = delta + gamma * gae_lambda * (1.0 - episode_done[index]) * gae
        advantages[index] = gae
    return advantages, advantages + values


class PPOAgent:
    def __init__(
        self, config: PPOConfig, observation_dim: int, action_count: int,
        seed: int, device: torch.device,
    ):
        self.config = config
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.network = ActorCritic(observation_dim, action_count, config.hidden_sizes).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        distribution = self.network.distribution(tensor)
        action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(self.network.value(tensor).item()),
        )

    @torch.no_grad()
    def value(self, observation: np.ndarray) -> float:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.network.value(tensor).item())

    def update(self, rollout: dict[str, np.ndarray]) -> dict[str, float]:
        observations = torch.as_tensor(rollout["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(rollout["actions"], dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(rollout["log_probs"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0
        count = len(observations)
        for _ in range(self.config.optimization_epochs):
            permutation = self.rng.permutation(count)
            for start in range(0, count, self.config.minibatch_size):
                indices_np = permutation[start:start + self.config.minibatch_size]
                indices = torch.as_tensor(indices_np, device=self.device)
                distribution = self.network.distribution(observations[indices])
                new_log_probs = distribution.log_prob(actions[indices])
                ratio = (new_log_probs - old_log_probs[indices]).exp()
                unclipped = ratio * advantages[indices]
                if self.config.use_clipping:
                    clipped = torch.clamp(
                        ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                    ) * advantages[indices]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                else:
                    policy_loss = -unclipped.mean()
                value_loss = nn.functional.mse_loss(self.network.value(observations[indices]), returns[indices])
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                updates += 1
        return {key: value / max(updates, 1) for key, value in totals.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.network.state_dict()}, path)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint["model"])
