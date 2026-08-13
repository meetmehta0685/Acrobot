"""Raw run records and resume-safe persistence."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from experiments.configuration import RunSpec


RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results" / "study"


@dataclass
class RunResult:
    spec: dict[str, Any]
    key: str
    group: str
    arm: str
    method: str
    seed: int
    eval_steps: list[int] = field(default_factory=list)
    eval_returns: list[list[float]] = field(default_factory=list)
    eval_lengths: list[list[int]] = field(default_factory=list)
    eval_successes: list[list[int]] = field(default_factory=list)
    eval_mean: list[float] = field(default_factory=list)
    eval_success_rate: list[float] = field(default_factory=list)
    train_episode_returns: list[float] = field(default_factory=list)
    train_episode_steps: list[int] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    update_count: int = 0
    steps_to_solve: int | None = None
    final_mean: float = 0.0
    final_success_rate: float = 0.0
    area_under_curve: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunResult":
        return cls(**data)


def group_directory(group: str, root: Path = RESULTS_ROOT) -> Path:
    return root / group


def paths(spec: RunSpec, root: Path = RESULTS_ROOT) -> tuple[Path, Path]:
    directory = group_directory(spec.group, root)
    return directory / f"{spec.key}.json", directory / f"{spec.key}.csv"


def exists(spec: RunSpec, root: Path = RESULTS_ROOT) -> bool:
    return paths(spec, root)[0].is_file()


def save(result: RunResult, spec: RunSpec, root: Path = RESULTS_ROOT) -> None:
    json_path, csv_path = paths(spec, root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)
    temporary.replace(json_path)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "mean_return", "success_rate", "episode_returns"])
        for step, mean, success, returns in zip(
            result.eval_steps, result.eval_mean, result.eval_success_rate, result.eval_returns
        ):
            writer.writerow([step, mean, success, json.dumps(returns)])


def load(spec: RunSpec, root: Path = RESULTS_ROOT) -> RunResult:
    with paths(spec, root)[0].open(encoding="utf-8") as handle:
        return RunResult.from_dict(json.load(handle))


def load_group(group: str, root: Path = RESULTS_ROOT) -> list[RunResult]:
    directory = group_directory(group, root)
    if not directory.exists():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            records.append(RunResult.from_dict(json.load(handle)))
    return sorted(records, key=lambda item: (item.arm, item.seed))

