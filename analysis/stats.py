"""Robust cross-seed statistics for reinforcement-learning results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


BOOTSTRAP_SEED = 20260811


def iqm(values: Sequence[float] | np.ndarray) -> float:
    """Interquartile mean using fractional weights at quartile boundaries."""
    array = np.sort(np.asarray(values, dtype=np.float64))
    array = array[~np.isnan(array)]
    if array.size == 0:
        return float("nan")
    lower, upper = 0.25 * array.size, 0.75 * array.size
    weighted_sum = 0.0
    total_weight = 0.0
    for index, value in enumerate(array):
        weight = max(0.0, min(index + 1.0, upper) - max(float(index), lower))
        weighted_sum += weight * float(value)
        total_weight += weight
    return weighted_sum / total_weight if total_weight else float(array.mean())


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: Callable[[np.ndarray], float] = iqm,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples)
    for index in range(resamples):
        samples[index] = statistic(rng.choice(array, size=array.size, replace=True))
    tail = (1.0 - confidence) / 2.0
    return tuple(np.quantile(samples, [tail, 1.0 - tail]).tolist())


@dataclass
class CurveSummary:
    steps: np.ndarray
    centre: np.ndarray
    low: np.ndarray
    high: np.ndarray


def aggregate_curves(results, resamples: int = 1_000) -> CurveSummary:
    if not results:
        raise ValueError("Cannot aggregate an empty result list")
    steps = np.asarray(results[0].eval_steps)
    if any(result.eval_steps != results[0].eval_steps for result in results):
        raise ValueError("Evaluation schedules differ within an experiment arm")
    matrix = np.asarray([result.eval_mean for result in results], dtype=np.float64)
    centre = np.asarray([iqm(column) for column in matrix.T])
    intervals = np.asarray([
        bootstrap_ci(column, resamples=resamples, seed=BOOTSTRAP_SEED + index)
        for index, column in enumerate(matrix.T)
    ])
    return CurveSummary(steps, centre, intervals[:, 0], intervals[:, 1])


def group_by_arm(results) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for result in results:
        grouped.setdefault(result.arm, []).append(result)
    for items in grouped.values():
        items.sort(key=lambda item: item.seed)
    return grouped

