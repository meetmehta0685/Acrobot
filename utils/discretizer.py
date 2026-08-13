"""Convert Acrobot's six continuous observations into Q-table indices."""

import numpy as np


class AcrobotDiscretizer:
    """Uniformly discretize cos/sin values and the two angular velocities.

    Acrobot observations are ``[cos(t1), sin(t1), cos(t2), sin(t2), dt1, dt2]``.
    The first four values lie in [-1, 1]. Gymnasium clips angular velocities to
    [-4π, 4π] and [-9π, 9π]. Each value is mapped to an integer in its configured
    bin range, forming a six-dimensional Q-table index.
    """

    LOW = np.array([-1.0, -1.0, -1.0, -1.0, -4 * np.pi, -9 * np.pi])
    HIGH = np.array([1.0, 1.0, 1.0, 1.0, 4 * np.pi, 9 * np.pi])

    def __init__(self, bins: tuple[int, ...]):
        if len(bins) != 6 or any(value < 2 for value in bins):
            raise ValueError("Acrobot requires six bin counts, each at least 2")
        self.bins = tuple(int(value) for value in bins)
        self._width = (self.HIGH - self.LOW) / np.asarray(self.bins)

    def encode(self, observation: np.ndarray) -> tuple[int, ...]:
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (6,):
            raise ValueError(f"Expected observation shape (6,), got {observation.shape}")
        indices = np.floor((observation - self.LOW) / self._width).astype(int)
        indices = np.clip(indices, 0, np.asarray(self.bins) - 1)
        return tuple(indices.tolist())

