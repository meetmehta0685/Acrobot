"""Aggregate evaluation measurements across episodes and seeds."""

import numpy as np
import pandas as pd


def evaluation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations = dict(
        mean_return=("episode_return", "mean"),
        median_return=("episode_return", "median"),
        return_std=("episode_return", "std"),
        success_rate=("success", "mean"),
        average_steps=("episode_length", "mean"),
        seeds=("seed", "nunique"),
    )
    if "training_time_seconds" in frame:
        aggregations["mean_training_seconds"] = ("training_time_seconds", "mean")
    if "interaction_budget" in frame:
        aggregations["environment_interactions"] = ("interaction_budget", "max")
    return frame.groupby("algorithm").agg(**aggregations).reset_index()


def experiment_summary(training: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    """Combine evaluation quality with convergence and stability measurements."""
    summary = evaluation_summary(evaluation)
    per_run: list[dict] = []
    for (algorithm, seed), group in training.groupby(["algorithm", "seed"]):
        ordered = group.sort_values("environment_steps")
        rolling_success = ordered["success"].rolling(20, min_periods=20).mean()
        converged = ordered.loc[rolling_success >= 0.8, "environment_steps"]
        tail = ordered.tail(20)
        per_run.append({
            "algorithm": algorithm,
            "seed": seed,
            "convergence_steps_80pct": float(converged.iloc[0]) if not converged.empty else np.nan,
            "final_training_return_std": tail["episode_return"].std(),
        })
    if per_run:
        run_frame = pd.DataFrame(per_run)
        extra = run_frame.groupby("algorithm").agg(
            mean_convergence_steps_80pct=("convergence_steps_80pct", "mean"),
            mean_final_training_return_std=("final_training_return_std", "mean"),
        ).reset_index()
        summary = summary.merge(extra, on="algorithm", how="left")
    return summary
