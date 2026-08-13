"""Validated study and run configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_PATH = ROOT / "configs" / "study.yaml"
DEFAULT_TUNED_PATH = ROOT / "configs" / "tuned.yaml"


def load_yaml(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int
    group: str
    params: dict[str, Any] = field(default_factory=dict)
    variant: str = ""
    env_id: str = "Acrobot-v1"
    total_steps: int = 100_000
    eval_every: int = 5_000
    eval_episodes: int = 10
    solved_return: float = -100.0
    max_episode_steps: int = 500

    @property
    def arm(self) -> str:
        return f"{self.method}:{self.variant}" if self.variant else self.method

    @property
    def key(self) -> str:
        digest = hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:10]
        safe_arm = self.arm.replace("=", "-").replace(":", "-").replace(" ", "_")
        return f"{safe_arm}-seed{self.seed}-{digest}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunSpec":
        return cls(**data)


def base_run_settings(study: dict[str, Any], total_steps: int | None = None) -> dict[str, Any]:
    settings = dict(study["run"])
    if total_steps is not None:
        settings["total_steps"] = int(total_steps)
        settings["eval_every"] = min(int(settings["eval_every"]), int(total_steps))
    return settings

