"""Central configuration for training and evaluation."""

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODEL_DIR = RESULTS_DIR / "models"
LOG_DIR = RESULTS_DIR / "logs"
PLOT_DIR = RESULTS_DIR / "plots"


@dataclass
class CommonConfig:
    env_id: str = "Acrobot-v1"
    seed: int = 42
    interaction_budget: int = 100_000
    max_episode_steps: int = 500
    eval_episodes: int = 20


@dataclass
class QLearningConfig:
    learning_rate: float = 0.10
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 75_000
    bins: tuple[int, ...] = (8, 8, 8, 8, 10, 10)


@dataclass
class DQNConfig:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    hidden_sizes: tuple[int, ...] = (128, 128)
    replay_capacity: int = 100_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_frequency: int = 4
    target_sync_frequency: int = 1_000
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 75_000
    gradient_clip: float = 10.0
    use_target_network: bool = True


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    rollout_steps: int = 2_048
    optimization_epochs: int = 10
    minibatch_size: int = 64
    hidden_sizes: tuple[int, ...] = (64, 64)
    gradient_clip: float = 0.5
    use_clipping: bool = True


@dataclass
class ProjectConfig:
    common: CommonConfig = field(default_factory=CommonConfig)
    q_learning: QLearningConfig = field(default_factory=QLearningConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)


def ensure_result_directories() -> None:
    for directory in (MODEL_DIR, LOG_DIR, PLOT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
