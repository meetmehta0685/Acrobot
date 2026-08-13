"""Metric persistence helpers."""

from pathlib import Path

import pandas as pd


TRAIN_COLUMNS = [
    "algorithm", "seed", "episode", "environment_steps", "episode_return",
    "episode_length", "success", "epsilon", "loss", "elapsed_seconds"
]


def save_records(records: list[dict], path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(path, index=False)
    return frame

