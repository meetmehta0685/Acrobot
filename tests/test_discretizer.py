import numpy as np
import pytest

from utils.discretizer import AcrobotDiscretizer


def test_discretizer_maps_bounds_and_clips() -> None:
    discretizer = AcrobotDiscretizer((4, 4, 4, 4, 5, 5))
    assert discretizer.encode(discretizer.LOW) == (0, 0, 0, 0, 0, 0)
    assert discretizer.encode(discretizer.HIGH) == (3, 3, 3, 3, 4, 4)
    assert discretizer.encode(discretizer.HIGH * 2) == (3, 3, 3, 3, 4, 4)


def test_discretizer_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        AcrobotDiscretizer((4,) * 6).encode(np.zeros(5))

