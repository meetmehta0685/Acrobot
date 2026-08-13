"""Equal-budget random hyperparameter search."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.configuration import RunSpec, base_run_settings
from experiments.matrix import run_all
from experiments.results import RESULTS_ROOT, load


def _sample_trials(space: dict[str, list], count: int, seed: int) -> list[dict[str, Any]]:
    keys = list(space)
    combinations = [dict(zip(keys, values)) for values in itertools.product(*(space[key] for key in keys))]
    rng = np.random.default_rng(seed)
    if count >= len(combinations):
        return combinations
    indices = rng.choice(len(combinations), size=count, replace=False)
    return [combinations[int(index)] for index in indices]


def build_tuning_specs(study: dict) -> tuple[list[RunSpec], dict[str, list[dict]]]:
    tuning = study["tuning"]
    base = base_run_settings(study, int(tuning["total_steps"]))
    trials_by_method: dict[str, list[dict]] = {}
    specs: list[RunSpec] = []
    for method_index, method in enumerate(study["methods"]):
        trials = _sample_trials(
            tuning["spaces"][method], int(tuning["trials"]),
            int(tuning["search_seed"]) + method_index,
        )
        trials_by_method[method] = trials
        for trial_index, params in enumerate(trials):
            for seed in study["seeds"]["tuning"]:
                specs.append(RunSpec(
                    method=method, seed=seed, group="tuning",
                    variant=f"trial{trial_index:02d}", params=params, **base,
                ))
    return specs, trials_by_method


def tune(
    study: dict, output: Path, workers: int = 1,
    root: Path = RESULTS_ROOT, force: bool = False,
) -> dict[str, dict]:
    specs, trials_by_method = build_tuning_specs(study)
    run_all(specs, workers=workers, root=root, force=force)
    selected: dict[str, dict] = {}
    report: dict[str, Any] = {}
    for method, trials in trials_by_method.items():
        candidates = []
        for trial_index, params in enumerate(trials):
            matching = [
                spec for spec in specs
                if spec.method == method and spec.variant == f"trial{trial_index:02d}"
            ]
            results = [load(spec, root) for spec in matching]
            final_score = float(np.mean([result.final_mean for result in results]))
            auc_score = float(np.mean([result.area_under_curve for result in results]))
            candidates.append({
                "trial": trial_index, "params": params,
                "mean_final_return": final_score, "mean_auc": auc_score,
            })
        candidates.sort(key=lambda item: (item["mean_final_return"], item["mean_auc"]), reverse=True)
        selected[method] = candidates[0]["params"]
        report[method] = candidates
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(selected, handle, sort_keys=False)
    report_path = root / "tuning_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected hyperparameters written to {output}")
    return selected

