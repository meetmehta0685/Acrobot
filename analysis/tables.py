"""Generate CSV and Markdown tables from completed runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.figures import LABELS
from analysis.stats import bootstrap_ci, group_by_arm, iqm
from experiments.results import RESULTS_ROOT, load_group


def build_summary(root: Path = RESULTS_ROOT) -> pd.DataFrame:
    grouped = group_by_arm(load_group("main", root))
    rows = []
    for method, results in grouped.items():
        finals = np.asarray([result.final_mean for result in results])
        successes = np.asarray([result.final_success_rate for result in results])
        solved = [result.steps_to_solve for result in results if result.steps_to_solve is not None]
        low, high = bootstrap_ci(finals)
        rows.append({
            "algorithm": LABELS.get(method, method),
            "seeds": len(results),
            "final_return_iqm": iqm(finals),
            "return_ci_low": low,
            "return_ci_high": high,
            "success_rate_iqm": iqm(successes),
            "median_steps_to_solve": float(np.median(solved)) if solved else np.nan,
            "failed_seed_rate": 1.0 - len(solved) / len(results),
            "mean_training_seconds": float(np.mean([result.wall_clock_seconds for result in results])),
            "mean_updates": float(np.mean([result.update_count for result in results])),
        })
    return pd.DataFrame(rows)


def write_tables(root: Path = RESULTS_ROOT) -> tuple[Path, Path]:
    output = root / "tables"
    output.mkdir(parents=True, exist_ok=True)
    summary = build_summary(root)
    csv_path = output / "main_summary.csv"
    markdown_path = output / "main_summary.md"
    summary.to_csv(csv_path, index=False)
    display = summary.copy()
    for column in ("final_return_iqm", "return_ci_low", "return_ci_high", "mean_training_seconds", "mean_updates"):
        display[column] = display[column].map(lambda value: f"{value:.1f}")
    for column in ("success_rate_iqm", "failed_seed_rate"):
        display[column] = display[column].map(lambda value: f"{100 * value:.0f}%")
    display["median_steps_to_solve"] = display["median_steps_to_solve"].map(
        lambda value: "Not solved" if pd.isna(value) else f"{value:,.0f}"
    )
    markdown_path.write_text(display.to_markdown(index=False) + "\n", encoding="utf-8")
    return csv_path, markdown_path

