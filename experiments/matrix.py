"""Build and execute resumable experiment matrices."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from experiments.configuration import RunSpec, base_run_settings
from experiments.results import RESULTS_ROOT, exists, save
from experiments.runner import run_experiment


def _format(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "x".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def build_main_specs(study: dict, tuned: dict) -> list[RunSpec]:
    base = base_run_settings(study)
    return [
        RunSpec(method=method, seed=seed, group="main", params=dict(tuned[method]), **base)
        for method in study["methods"] for seed in study["seeds"]["evaluation"]
    ]


def build_sweep_specs(study: dict, tuned: dict) -> list[RunSpec]:
    total_steps = int(study["sweeps"].get("total_steps", study["run"]["total_steps"]))
    base = base_run_settings(study, total_steps)
    specs: list[RunSpec] = []
    for method in study["methods"]:
        definition = study["sweeps"].get(method)
        if not definition:
            continue
        parameter = definition["parameter"]
        for value in definition["values"]:
            params = dict(tuned[method])
            params[parameter] = value
            for seed in study["seeds"]["sweep"]:
                specs.append(RunSpec(
                    method=method, seed=seed, group="sweep",
                    variant=f"{parameter}={_format(value)}", params=params, **base,
                ))
    return specs


def build_ablation_specs(study: dict, tuned: dict) -> list[RunSpec]:
    total_steps = int(study["ablations"].get("total_steps", study["run"]["total_steps"]))
    base = base_run_settings(study, total_steps)
    specs: list[RunSpec] = []
    for name, definition in study["ablations"].items():
        if name == "total_steps":
            continue
        method = definition["method"]
        params = dict(tuned[method])
        params.update(definition["params"])
        for seed in study["seeds"]["sweep"]:
            specs.append(RunSpec(
                method=method, seed=seed, group="ablation", variant=name,
                params=params, **base,
            ))
    return specs


def _worker(payload: tuple[dict, str]) -> tuple[str, float, float]:
    spec_data, root_text = payload
    spec = RunSpec.from_dict(spec_data)
    root = Path(root_text)
    result = run_experiment(spec, root)
    save(result, spec, root)
    return spec.key, result.final_mean, result.wall_clock_seconds


def run_all(
    specs: Iterable[RunSpec], workers: int = 1,
    root: Path = RESULTS_ROOT, force: bool = False,
) -> int:
    specs = list(specs)
    pending = [spec for spec in specs if force or not exists(spec, root)]
    skipped = len(specs) - len(pending)
    if skipped:
        print(f"Skipping {skipped} completed run(s).")
    if not pending:
        return 0
    workers = max(1, min(workers, os.cpu_count() or 1, len(pending)))
    print(f"Executing {len(pending)} run(s) with {workers} worker(s).", flush=True)
    if workers == 1:
        for index, spec in enumerate(pending, 1):
            key, final, seconds = _worker((spec.to_dict(), str(root)))
            print(f"[{index}/{len(pending)}] {key}: final={final:.1f}, {seconds:.1f}s", flush=True)
        return len(pending)
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker, (spec.to_dict(), str(root))): spec for spec in pending
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                key, final, seconds = future.result()
            except Exception as error:
                for item in futures:
                    item.cancel()
                raise RuntimeError(f"Run failed: {spec.key}") from error
            completed += 1
            print(
                f"[{completed}/{len(pending)}] {key}: final={final:.1f}, {seconds:.1f}s",
                flush=True,
            )
    return completed

