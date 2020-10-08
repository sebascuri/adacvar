"""Learning rate decay provides different scheduling implementations."""

from abc import ABC, abstractmethod

import numpy as np

__author__ = "Sebastian Curi"
__all__ = ["Constant", "RobbinsMonro", "AdaGrad"]


class LearningRateDecay(ABC):
    """Learning rate decay scheduler abstract implementation.

    Parameters
    ----------
    eta: initial value of the learning rate.
    dim: dimension of the learning rate.

    Methods
    -------
    __call__: return the current value of the learning rate.

    """

    def __init__(self, eta, dim):
        self._eta = eta * np.ones(dim)

    @abstractmethod
    def __call__(self, time, losses, idx):
        """Return the current value of the learning rate.

        Parameters
        ----------
        time: int
            Number of steps.
        losses: ndarray
            Observed losses.
        idx: ndarray
            Observed indexes.

        Returns
        -------
        eta: float
            Learning rate.

        """
        raise NotImplementedError


class Constant(LearningRateDecay):
    """Constant learning rate implementation."""

    def __call__(self, time, losses, idx):
        """See `LearningRateDecay.__call__' for help."""
        return np.clip(self._eta, 0, 1000)


class RobbinsMonro(LearningRateDecay):
    """Robbins Monro Scheduling implementation.

    eta(t) = eta(0) / sqrt(t)
    """

    def __call__(self, time, losses, idx):
        """See `LearningRateDecay.__call__' for help."""
        return np.clip(self._eta / np.sqrt(time + 1), 0, 1)


class AdaGrad(LearningRateDecay):
    """Ada-Grad Scheduling implementation.

    eta_i(t) = eta_i(0) / sqrt( eps + sum(l(t')_i)^2)
    """

    def __init__(self, eta, dim):
        super().__init__(eta, dim)
        self._sum_squared_gradient_norm = 1e-6 * np.ones(dim)

    def __call__(self, time, gradients, idx):
        """See `LearningRateDecay.__call__' for help."""
        eta = np.clip(self._eta / np.sqrt(self._sum_squared_gradient_norm), 0, 1)
        self._sum_squared_gradient_norm[idx] += gradients ** 2
        return eta
