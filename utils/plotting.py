"""Plots generated exclusively from stored metric records."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_training(frame: pd.DataFrame, output: Path, window: int = 20) -> None:
    if frame.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for algorithm, group in frame.groupby("algorithm"):
        ordered = group.sort_values("environment_steps")
        reward = ordered["episode_return"].rolling(window, min_periods=1).mean()
        success = ordered["success"].rolling(window, min_periods=1).mean()
        axes[0].plot(ordered["environment_steps"], reward, label=algorithm)
        axes[1].plot(ordered["environment_steps"], success, label=algorithm)
    axes[0].set_ylabel(f"Return ({window}-episode mean)")
    axes[1].set_ylabel(f"Success rate ({window}-episode mean)")
    axes[1].set_xlabel("Environment interactions")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_evaluation(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty:
        return
    summary = frame.groupby("algorithm").agg(
        mean_return=("episode_return", "mean"),
        success_rate=("success", "mean"),
        mean_steps=("episode_length", "mean"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    summary["mean_return"].plot.bar(ax=axes[0], title="Mean evaluation return")
    summary["success_rate"].plot.bar(ax=axes[1], title="Evaluation success rate")
    summary["mean_steps"].plot.bar(ax=axes[2], title="Mean episode steps")
    for axis in axes:
        axis.tick_params(axis="x", rotation=0)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_seed_stability(frame: pd.DataFrame, output: Path) -> None:
    """Show the distribution across every evaluation episode and seed."""
    if frame.empty:
        return
    algorithms = sorted(frame["algorithm"].unique())
    values = [frame.loc[frame["algorithm"] == name, "episode_return"] for name in algorithms]
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.boxplot(values, tick_labels=algorithms, showmeans=True)
    axis.set_ylabel("Evaluation return")
    axis.set_title("Evaluation stability across seeds")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_training_time(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty or "training_time_seconds" not in frame:
        return
    times = frame.groupby(["algorithm", "seed"])["training_time_seconds"].first().unstack(0)
    axis = times.plot.bar(figsize=(8, 5))
    axis.set_ylabel("Training time (seconds)")
    axis.set_xlabel("Seed")
    axis.set_title("Wall-clock training time")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    axis.figure.savefig(output, dpi=180)
    plt.close(axis.figure)
