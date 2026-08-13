"""Publication-quality figures generated only from raw run records."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.stats import aggregate_curves, group_by_arm, iqm
from experiments.results import RESULTS_ROOT, load_group


LABELS = {"q_learning": "Q-Learning", "dqn": "DQN", "ppo": "PPO"}
COLOURS = {"q_learning": "#3B82F6", "dqn": "#F97316", "ppo": "#10B981"}
ABLATION_COLOUR = "#DC2626"


def _style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22,
        "figure.dpi": 160,
    })


def _save(fig, output: Path, name: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{name}.png", output / f"{name}.pdf"]
    fig.tight_layout()
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _curve(axis, results, method: str, label: str | None = None, colour: str | None = None):
    summary = aggregate_curves(results)
    colour = COLOURS[method] if colour is None else colour
    axis.plot(summary.steps, summary.centre, color=colour, label=label or LABELS[method])
    axis.fill_between(summary.steps, summary.low, summary.high, color=colour, alpha=0.16)


def learning_curves(study: dict, root: Path, output: Path) -> list[Path]:
    _style()
    grouped = group_by_arm(load_group("main", root))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in study["methods"]:
        if method not in grouped:
            continue
        _curve(axes[0], grouped[method], method)
        successes = np.asarray([run.eval_success_rate for run in grouped[method]])
        centre = np.asarray([iqm(column) for column in successes.T])
        axes[1].plot(grouped[method][0].eval_steps, centre, color=COLOURS[method], label=LABELS[method])
    axes[0].axhline(study["run"]["solved_return"], linestyle="--", color="0.35", label="Solved threshold")
    axes[0].set(title="Held-out evaluation return", xlabel="Environment interactions", ylabel="Return")
    axes[1].set(title="Held-out success rate", xlabel="Environment interactions", ylabel="Success rate", ylim=(-0.03, 1.03))
    for axis in axes:
        axis.legend()
    return _save(fig, output, "learning_curves")


def seed_scatter(study: dict, root: Path, output: Path) -> list[Path]:
    _style()
    grouped = group_by_arm(load_group("main", root))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    rng = np.random.default_rng(0)
    methods = [method for method in study["methods"] if method in grouped]
    for position, method in enumerate(methods):
        finals = np.asarray([run.final_mean for run in grouped[method]])
        success = np.asarray([run.final_success_rate for run in grouped[method]])
        jitter = rng.uniform(-0.12, 0.12, len(finals))
        axes[0].scatter(position + jitter, finals, color=COLOURS[method], s=36, alpha=0.8)
        axes[1].scatter(position + jitter, success, color=COLOURS[method], s=36, alpha=0.8)
        axes[0].plot([position - 0.25, position + 0.25], [iqm(finals)] * 2, color="black")
        axes[1].plot([position - 0.25, position + 0.25], [iqm(success)] * 2, color="black")
    for axis in axes:
        axis.set_xticks(range(len(methods)), [LABELS[method] for method in methods])
    axes[0].axhline(study["run"]["solved_return"], linestyle="--", color="0.35")
    axes[0].set(title="Final return across seeds", ylabel="Evaluation return")
    axes[1].set(title="Reliability across seeds", ylabel="Success rate", ylim=(-0.03, 1.03))
    return _save(fig, output, "seed_stability")


def sweep_curves(study: dict, root: Path, output: Path) -> list[Path]:
    _style()
    grouped = group_by_arm(load_group("sweep", root))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for axis, method in zip(axes, study["methods"]):
        definition = study["sweeps"][method]
        parameter = definition["parameter"]
        colours = plt.cm.viridis(np.linspace(0.05, 0.9, len(definition["values"])))
        for colour, value in zip(colours, definition["values"]):
            text = "x".join(map(str, value)) if isinstance(value, list) else f"{value:g}"
            arm = f"{method}:{parameter}={text}"
            if arm in grouped:
                _curve(axis, grouped[arm], method, label=text, colour=colour)
        axis.axhline(study["run"]["solved_return"], linestyle="--", color="0.4", linewidth=0.8)
        axis.set(title=f"{LABELS[method]}\n{definition['label']}", xlabel="Environment interactions")
        axis.legend(title=parameter, fontsize=7)
    axes[0].set_ylabel("Evaluation return")
    return _save(fig, output, "hyperparameter_sweeps")


def ablation_curves(study: dict, root: Path, output: Path) -> list[Path]:
    _style()
    main = group_by_arm(load_group("main", root))
    ablations = group_by_arm(load_group("ablation", root))
    definitions = [(name, value) for name, value in study["ablations"].items() if name != "total_steps"]
    fig, axes = plt.subplots(1, len(definitions), figsize=(5 * len(definitions), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, (name, definition) in zip(axes, definitions):
        method = definition["method"]
        if method in main:
            control_steps = int(study["ablations"]["total_steps"])
            control = []
            for run in main[method]:
                limit = max(index for index, step in enumerate(run.eval_steps) if step <= control_steps) + 1
                clone = type(run).from_dict(run.to_dict())
                clone.eval_steps = clone.eval_steps[:limit]
                clone.eval_mean = clone.eval_mean[:limit]
                control.append(clone)
            _curve(axis, control, method, label="Full method")
        arm = f"{method}:{name}"
        if arm in ablations:
            _curve(axis, ablations[arm], method, label="Component removed", colour=ABLATION_COLOUR)
        axis.axhline(study["run"]["solved_return"], linestyle="--", color="0.4", linewidth=0.8)
        axis.set(title=definition["label"], xlabel="Environment interactions")
        axis.legend()
    axes[0].set_ylabel("Evaluation return")
    return _save(fig, output, "ablations")


def efficiency(study: dict, root: Path, output: Path) -> list[Path]:
    _style()
    grouped = group_by_arm(load_group("main", root))
    methods = [method for method in study["methods"] if method in grouped]
    times = [np.mean([run.wall_clock_seconds for run in grouped[method]]) for method in methods]
    updates = [np.mean([run.update_count for run in grouped[method]]) for method in methods]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar([LABELS[item] for item in methods], times, color=[COLOURS[item] for item in methods])
    axes[1].bar([LABELS[item] for item in methods], updates, color=[COLOURS[item] for item in methods])
    axes[0].set(title="Mean training time", ylabel="Seconds")
    axes[1].set(title="Mean optimizer / value updates", ylabel="Updates")
    return _save(fig, output, "computational_efficiency")


def build_all(study: dict, root: Path = RESULTS_ROOT, output: Path | None = None) -> list[Path]:
    output = output or root / "figures"
    written: list[Path] = []
    for builder in (learning_curves, seed_scatter, sweep_curves, ablation_curves, efficiency):
        written.extend(builder(study, root, output))
    return written
