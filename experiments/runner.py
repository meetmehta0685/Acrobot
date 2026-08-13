"""One environment-interaction and evaluation harness for all three methods."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent, compute_gae
from agents.q_learning import QLearningAgent
from config import DQNConfig, PPOConfig, QLearningConfig
from experiments.configuration import RunSpec
from experiments.results import RESULTS_ROOT, RunResult
from utils.replay_buffer import ReplayBuffer
from utils.seed import seed_everything


EVALUATION_SEEDS = tuple(90_000 + index for index in range(100))


def _tuple_fields(params: dict[str, Any]) -> dict[str, Any]:
    converted = dict(params)
    for key in ("bins", "hidden_sizes"):
        if key in converted:
            converted[key] = tuple(converted[key])
    return converted


class QAdapter:
    def __init__(self, spec: RunSpec, action_count: int):
        config = replace(QLearningConfig(), **_tuple_fields(spec.params))
        self.agent = QLearningAgent(config, action_count, spec.seed)
        self.spec = spec
        self.update_count = 0
        self.last_loss = float("nan")

    def act(self, observation: np.ndarray, step: int, greedy: bool = False) -> int:
        return self.agent.act(
            observation,
            epsilon=0.0 if greedy else self.agent.epsilon(step),
            deterministic_ties=greedy,
        )

    def observe(self, observation, action, reward, next_observation, terminated, done, step):
        self.last_loss = abs(
            self.agent.update(observation, action, reward, next_observation, terminated)
        )
        self.update_count += 1

    def finish(self) -> None:
        pass

    def save(self, path: Path) -> None:
        self.agent.save(path.with_suffix(".npz"))

    def diagnostics(self) -> dict[str, Any]:
        return {
            "q_table_entries": int(self.agent.q_table.size),
            "q_table_megabytes": float(self.agent.q_table.nbytes / 1_000_000),
            "hyperparameter_count": 7,
        }


class DQNAdapter:
    def __init__(self, spec: RunSpec, observation_dim: int, action_count: int, device):
        config = replace(DQNConfig(), **_tuple_fields(spec.params))
        self.agent = DQNAgent(config, observation_dim, action_count, spec.seed, device)
        self.replay = ReplayBuffer(config.replay_capacity, observation_dim, spec.seed)
        self.config = config
        self.update_count = 0
        self.last_loss = float("nan")

    def act(self, observation: np.ndarray, step: int, greedy: bool = False) -> int:
        return self.agent.act(observation, epsilon=0.0 if greedy else self.agent.epsilon(step))

    def observe(self, observation, action, reward, next_observation, terminated, done, step):
        self.replay.add(observation, action, reward, next_observation, terminated)
        completed_steps = step + 1
        if (
            completed_steps >= self.config.learning_starts
            and completed_steps % self.config.train_frequency == 0
            and len(self.replay) >= self.config.batch_size
        ):
            self.last_loss = self.agent.learn(self.replay.sample(self.config.batch_size))
            self.update_count += 1
        if completed_steps % self.config.target_sync_frequency == 0:
            self.agent.sync_target()

    def finish(self) -> None:
        pass

    def save(self, path: Path) -> None:
        self.agent.save(path.with_suffix(".pt"))

    def diagnostics(self) -> dict[str, Any]:
        parameters = sum(parameter.numel() for parameter in self.agent.online.parameters())
        return {
            "network_parameters": int(parameters),
            "replay_size": len(self.replay),
            "uses_target_network": self.config.use_target_network,
            "hyperparameter_count": 13,
        }


class PPOAdapter:
    def __init__(self, spec: RunSpec, observation_dim: int, action_count: int, device):
        config = replace(PPOConfig(), **_tuple_fields(spec.params))
        self.agent = PPOAgent(config, observation_dim, action_count, spec.seed, device)
        self.config = config
        self.data: dict[str, list] = {key: [] for key in (
            "observations", "actions", "rewards", "log_probs", "values",
            "next_values", "terminated", "done",
        )}
        self.pending: tuple[int, float, float] | None = None
        self.update_count = 0
        self.last_loss = float("nan")

    def act(self, observation: np.ndarray, step: int, greedy: bool = False) -> int:
        action, log_probability, value = self.agent.act(observation, deterministic=greedy)
        if not greedy:
            self.pending = (action, log_probability, value)
        return action

    def observe(self, observation, action, reward, next_observation, terminated, done, step):
        if self.pending is None:
            raise RuntimeError("PPO observe called without a training action")
        _, log_probability, value = self.pending
        next_value = 0.0 if terminated else self.agent.value(next_observation)
        for key, item in (
            ("observations", observation), ("actions", action), ("rewards", reward),
            ("log_probs", log_probability), ("values", value),
            ("next_values", next_value), ("terminated", float(terminated)),
            ("done", float(done)),
        ):
            self.data[key].append(item)
        self.pending = None
        if len(self.data["rewards"]) >= self.config.rollout_steps:
            self._update()

    def _update(self) -> None:
        if not self.data["rewards"]:
            return
        arrays = {key: np.asarray(value, dtype=np.float32) for key, value in self.data.items()}
        advantages, returns = compute_gae(
            arrays["rewards"], arrays["values"], arrays["next_values"],
            arrays["terminated"], arrays["done"], self.config.gamma, self.config.gae_lambda,
        )
        metrics = self.agent.update({
            "observations": arrays["observations"],
            "actions": arrays["actions"],
            "log_probs": arrays["log_probs"],
            "advantages": advantages,
            "returns": returns,
        })
        minibatches = math.ceil(len(advantages) / self.config.minibatch_size)
        self.update_count += self.config.optimization_epochs * minibatches
        self.last_loss = metrics["policy_loss"]
        for value in self.data.values():
            value.clear()

    def finish(self) -> None:
        self._update()

    def save(self, path: Path) -> None:
        self.agent.save(path.with_suffix(".pt"))

    def diagnostics(self) -> dict[str, Any]:
        parameters = sum(parameter.numel() for parameter in self.agent.network.parameters())
        return {
            "network_parameters": int(parameters),
            "uses_clipping": self.config.use_clipping,
            "entropy_coefficient": self.config.entropy_coefficient,
            "hyperparameter_count": 12,
        }


def make_adapter(spec: RunSpec, observation_dim: int, action_count: int, device):
    if spec.method == "q_learning":
        return QAdapter(spec, action_count)
    if spec.method == "dqn":
        return DQNAdapter(spec, observation_dim, action_count, device)
    if spec.method == "ppo":
        return PPOAdapter(spec, observation_dim, action_count, device)
    raise ValueError(f"Unknown method: {spec.method}")


def evaluate(spec: RunSpec, adapter) -> tuple[list[float], list[int], list[int]]:
    env = gym.make(spec.env_id, max_episode_steps=spec.max_episode_steps)
    returns, lengths, successes = [], [], []
    for episode in range(spec.eval_episodes):
        observation, _ = env.reset(seed=EVALUATION_SEEDS[episode])
        terminated = truncated = False
        total, length = 0.0, 0
        while not (terminated or truncated):
            action = adapter.act(observation, step=0, greedy=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            length += 1
        returns.append(total)
        lengths.append(length)
        successes.append(int(terminated))
    env.close()
    return returns, lengths, successes


def _record_evaluation(result: RunResult, spec: RunSpec, adapter, step: int) -> None:
    returns, lengths, successes = evaluate(spec, adapter)
    mean_return = float(np.mean(returns))
    result.eval_steps.append(step)
    result.eval_returns.append(returns)
    result.eval_lengths.append(lengths)
    result.eval_successes.append(successes)
    result.eval_mean.append(mean_return)
    result.eval_success_rate.append(float(np.mean(successes)))
    if result.steps_to_solve is None and mean_return >= spec.solved_return:
        result.steps_to_solve = step


def run_experiment(spec: RunSpec, results_root: Path = RESULTS_ROOT) -> RunResult:
    """Train and periodically evaluate one algorithm under an exact step budget."""
    torch.set_num_threads(1)
    seed_everything(spec.seed)
    device = torch.device("cpu")  # parallel small-network runs are faster and more reproducible on CPU
    env = gym.make(spec.env_id, max_episode_steps=spec.max_episode_steps)
    env.action_space.seed(spec.seed)
    observation_dim = int(np.prod(env.observation_space.shape))
    action_count = int(env.action_space.n)
    adapter = make_adapter(spec, observation_dim, action_count, device)
    result = RunResult(
        spec=spec.to_dict(), key=spec.key, group=spec.group, arm=spec.arm,
        method=spec.method, seed=spec.seed,
    )
    observation, _ = env.reset(seed=spec.seed)
    episode_return = 0.0
    started = time.perf_counter()
    _record_evaluation(result, spec, adapter, 0)

    for step in range(spec.total_steps):
        action = adapter.act(observation, step)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += float(reward)
        adapter.observe(
            observation, action, float(reward), next_observation,
            terminated, done, step,
        )
        if done:
            result.train_episode_returns.append(episode_return)
            result.train_episode_steps.append(step + 1)
            episode_return = 0.0
            observation, _ = env.reset()
        else:
            observation = next_observation
        completed_steps = step + 1
        if completed_steps % spec.eval_every == 0 and completed_steps < spec.total_steps:
            _record_evaluation(result, spec, adapter, completed_steps)

    adapter.finish()
    _record_evaluation(result, spec, adapter, spec.total_steps)
    result.wall_clock_seconds = time.perf_counter() - started
    result.update_count = int(adapter.update_count)
    result.final_mean = result.eval_mean[-1]
    result.final_success_rate = result.eval_success_rate[-1]
    result.area_under_curve = float(np.trapezoid(result.eval_mean, result.eval_steps))
    result.diagnostics = adapter.diagnostics()
    env.close()

    if spec.group == "main":
        model_path = results_root / "models" / spec.key
        model_path.parent.mkdir(parents=True, exist_ok=True)
        adapter.save(model_path)
    return result

