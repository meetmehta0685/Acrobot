from dataclasses import replace

from experiments.configuration import RunSpec
from experiments.runner import run_experiment


def test_run_key_is_stable_and_sensitive_to_configuration() -> None:
    spec = RunSpec(method="q_learning", seed=0, group="test")
    assert spec.key == RunSpec(method="q_learning", seed=0, group="test").key
    assert spec.key != replace(spec, seed=1).key
    assert spec.key != replace(spec, total_steps=50).key


def test_periodic_evaluation_and_training_are_reproducible(tmp_path) -> None:
    spec = RunSpec(
        method="q_learning", seed=3, group="test", total_steps=40,
        eval_every=20, eval_episodes=2, max_episode_steps=20,
        params={"bins": [2, 2, 2, 2, 2, 2], "epsilon_decay_steps": 20},
    )
    first = run_experiment(spec, tmp_path / "first")
    second = run_experiment(spec, tmp_path / "second")
    assert first.eval_steps == [0, 20, 40]
    assert first.eval_returns == second.eval_returns
    assert first.train_episode_returns == second.train_episode_returns

