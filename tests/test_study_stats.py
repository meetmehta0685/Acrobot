import numpy as np

from analysis.stats import bootstrap_ci, iqm


def test_iqm_uses_middle_half_and_fractional_boundaries() -> None:
    assert iqm([0.0, 1.0, 2.0, 100.0]) == 1.5
    assert iqm([1.0, 2.0, 3.0]) == 2.0


def test_bootstrap_interval_is_reproducible_and_ordered() -> None:
    first = bootstrap_ci([1, 2, 3, 4, 5], resamples=200, seed=7)
    second = bootstrap_ci([1, 2, 3, 4, 5], resamples=200, seed=7)
    assert first == second
    assert first[0] <= iqm([1, 2, 3, 4, 5]) <= first[1]

