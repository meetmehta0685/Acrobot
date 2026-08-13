"""Run tuning, experiments, analysis, and demonstrations for the Acrobot study."""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.demos import generate_demos, show_demo
from analysis.figures import build_all
from analysis.tables import write_tables
from experiments.configuration import (
    DEFAULT_STUDY_PATH, DEFAULT_TUNED_PATH, RunSpec, base_run_settings, load_yaml,
)
from experiments.matrix import (
    build_ablation_specs, build_main_specs, build_sweep_specs, run_all,
)
from experiments.results import RESULTS_ROOT
from experiments.tune import tune


def analyse(study: dict) -> None:
    figures = build_all(study)
    tables = write_tables()
    demos = generate_demos()
    for path in [*figures, *tables, *demos]:
        print(f"Wrote {path}")


def command_tune(args, study: dict) -> None:
    tune(study, args.tuned, workers=args.workers, force=args.force)


def command_matrix(args, study: dict) -> None:
    tuned = load_yaml(args.tuned)
    builders = {
        "main": build_main_specs,
        "sweep": build_sweep_specs,
        "ablation": build_ablation_specs,
    }
    specs = []
    for group in args.groups:
        specs.extend(builders[group](study, tuned))
    run_all(specs, workers=args.workers, force=args.force)


def command_single(args, study: dict) -> None:
    tuned = load_yaml(args.tuned)
    base = base_run_settings(study, args.steps)
    spec = RunSpec(
        method=args.method, seed=args.seed, group="debug",
        params=tuned[args.method], **base,
    )
    run_all([spec], workers=1, force=args.force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--tuned", type=Path, default=DEFAULT_TUNED_PATH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("tune")
    matrix = subparsers.add_parser("matrix")
    matrix.add_argument(
        "--groups", nargs="+", choices=("main", "sweep", "ablation"),
        default=["main", "sweep", "ablation"],
    )
    subparsers.add_parser("analyse")
    subparsers.add_parser("demos")
    demo = subparsers.add_parser("demo")
    demo.add_argument("--method", choices=("q_learning", "dqn", "ppo"), default="ppo")
    single = subparsers.add_parser("single")
    single.add_argument("--method", choices=("q_learning", "dqn", "ppo"), required=True)
    single.add_argument("--seed", type=int, default=0)
    single.add_argument("--steps", type=int, default=5_000)
    subparsers.add_parser("all")
    args = parser.parse_args()
    study = load_yaml(args.study)

    if args.command == "tune":
        command_tune(args, study)
    elif args.command == "matrix":
        command_matrix(args, study)
    elif args.command == "analyse":
        analyse(study)
    elif args.command == "demos":
        for path in generate_demos():
            print(f"Wrote {path}")
    elif args.command == "demo":
        show_demo(args.method)
    elif args.command == "single":
        command_single(args, study)
    else:
        command_tune(args, study)
        args.groups = ["main", "sweep", "ablation"]
        command_matrix(args, study)
        analyse(study)


if __name__ == "__main__":
    main()
