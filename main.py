"""Command-line interface for training, evaluation, comparison, and demos."""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from agents import DQNAgent, PPOAgent, QLearningAgent
from config import (
    LOG_DIR, MODEL_DIR, PLOT_DIR, ProjectConfig, ensure_result_directories,
)
from evaluation.evaluate import evaluate_policy
from evaluation.metrics import evaluation_summary, experiment_summary
from training.train_dqn import train_dqn
from training.train_ppo import train_ppo
from training.train_q_learning import train_q_learning
from utils.metrics import save_records
from utils.plotting import (
    plot_evaluation, plot_seed_stability, plot_training, plot_training_time,
)
from utils.seed import choose_device, seed_everything


ALGORITHMS = ("q_learning", "dqn", "ppo")


def model_path(algorithm: str, seed: int) -> Path:
    suffix = ".npz" if algorithm == "q_learning" else ".pt"
    return MODEL_DIR / f"{algorithm}_seed{seed}{suffix}"


def train_one(algorithm: str, config: ProjectConfig):
    common = config.common
    started = time.perf_counter()
    if algorithm == "q_learning":
        agent, records = train_q_learning(common, config.q_learning)
        policy = lambda observation: agent.act(
            observation, epsilon=0.0, deterministic_ties=True
        )
    elif algorithm == "dqn":
        agent, records = train_dqn(common, config.dqn)
        policy = lambda observation: agent.act(observation, epsilon=0.0)
    elif algorithm == "ppo":
        agent, records = train_ppo(common, config.ppo)
        policy = lambda observation: agent.act(observation, deterministic=True)[0]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    training_time = time.perf_counter() - started
    agent.save(model_path(algorithm, common.seed))
    evaluations = evaluate_policy(policy, algorithm, common)
    for record in evaluations:
        record["training_time_seconds"] = training_time
        record["interaction_budget"] = common.interaction_budget
    return records, evaluations


def command_train(args) -> None:
    ensure_result_directories()
    config = ProjectConfig()
    config.common = replace(
        config.common, seed=args.seed, interaction_budget=args.budget,
        eval_episodes=args.eval_episodes,
    )
    algorithms = ALGORITHMS if args.algorithm == "all" else (args.algorithm,)
    all_training: list[dict] = []
    all_evaluation: list[dict] = []
    for algorithm in algorithms:
        print(f"Training {algorithm} for {args.budget:,} environment interactions...")
        training, evaluation = train_one(algorithm, config)
        all_training.extend(training)
        all_evaluation.extend(evaluation)
    training_frame = save_records(all_training, LOG_DIR / "latest_training.csv")
    evaluation_frame = save_records(all_evaluation, LOG_DIR / "latest_evaluation.csv")
    plot_training(training_frame, PLOT_DIR / "latest_training.png")
    plot_evaluation(evaluation_frame, PLOT_DIR / "latest_evaluation.png")
    plot_seed_stability(evaluation_frame, PLOT_DIR / "latest_seed_stability.png")
    plot_training_time(evaluation_frame, PLOT_DIR / "latest_training_time.png")
    print(evaluation_summary(evaluation_frame).to_string(index=False))


def command_compare(args) -> None:
    ensure_result_directories()
    all_training: list[dict] = []
    all_evaluation: list[dict] = []
    for seed in args.seeds:
        for algorithm in ALGORITHMS:
            config = ProjectConfig()
            config.common = replace(
                config.common, seed=seed, interaction_budget=args.budget,
                eval_episodes=args.eval_episodes,
            )
            print(f"seed={seed} algorithm={algorithm} budget={args.budget:,}")
            training, evaluation = train_one(algorithm, config)
            all_training.extend(training)
            all_evaluation.extend(evaluation)
    training_frame = save_records(all_training, LOG_DIR / "comparison_training.csv")
    evaluation_frame = save_records(all_evaluation, LOG_DIR / "comparison_evaluation.csv")
    summary = experiment_summary(training_frame, evaluation_frame)
    summary.to_csv(LOG_DIR / "comparison_summary.csv", index=False)
    plot_training(training_frame, PLOT_DIR / "comparison_training.png")
    plot_evaluation(evaluation_frame, PLOT_DIR / "comparison_evaluation.png")
    plot_seed_stability(evaluation_frame, PLOT_DIR / "comparison_seed_stability.png")
    plot_training_time(evaluation_frame, PLOT_DIR / "comparison_training_time.png")
    print(summary.to_string(index=False))


def load_agent(algorithm: str, seed: int):
    config = ProjectConfig()
    seed_everything(seed)
    env = gym.make(config.common.env_id)
    observation_dim = int(np.prod(env.observation_space.shape))
    action_count = env.action_space.n
    env.close()
    if algorithm == "q_learning":
        agent = QLearningAgent(config.q_learning, action_count, seed)
        agent.load(model_path(algorithm, seed))
        policy = lambda observation: agent.act(
            observation, epsilon=0.0, deterministic_ties=True
        )
    elif algorithm == "dqn":
        agent = DQNAgent(config.dqn, observation_dim, action_count, seed, choose_device())
        agent.load(model_path(algorithm, seed))
        policy = lambda observation: agent.act(observation, epsilon=0.0)
    else:
        agent = PPOAgent(config.ppo, observation_dim, action_count, seed, choose_device())
        agent.load(model_path(algorithm, seed))
        policy = lambda observation: agent.act(observation, deterministic=True)[0]
    return policy


def command_evaluate(args, render: bool = False) -> None:
    config = ProjectConfig()
    config.common = replace(
        config.common, seed=args.seed,
        eval_episodes=1 if render else args.eval_episodes,
    )
    path = model_path(args.algorithm, args.seed)
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}. Train the algorithm first.")
    policy = load_agent(args.algorithm, args.seed)
    records = evaluate_policy(
        policy, args.algorithm, config.common, render_mode="human" if render else None
    )
    print(evaluation_summary(pd.DataFrame(records)).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="Train one or all algorithms")
    train.add_argument("--algorithm", choices=(*ALGORITHMS, "all"), default="all")
    train.add_argument("--budget", type=int, default=100_000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--eval-episodes", type=int, default=20)
    train.set_defaults(handler=command_train)

    compare = subparsers.add_parser("compare", help="Run a fair multi-seed comparison")
    compare.add_argument("--budget", type=int, default=100_000)
    compare.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    compare.add_argument("--eval-episodes", type=int, default=20)
    compare.set_defaults(handler=command_compare)

    for command in ("evaluate", "demo"):
        sub = subparsers.add_parser(command, help=f"{command.title()} a saved agent")
        sub.add_argument("--algorithm", choices=ALGORITHMS, required=True)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--eval-episodes", type=int, default=20)
        sub.set_defaults(handler=lambda args, demo=command == "demo": command_evaluate(args, demo))
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.handler(arguments)
