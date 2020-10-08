"""approximatekdpp.py module provides an approximate K-DPP data structure."""

import numpy as np
from scipy.optimize import fsolve

__author__ = "Sebastian Curi"
__all__ = ["ApproximateKDPP"]


class ApproximateKDPP(object):
    """Approximate K-DPP marignal distribution with a Multinomial distribution.

    Parameters
    ----------
    num_ground: int
        number of elements in ground set.
    k_sample:
        number of elements in the sample of the k-dpp.

    Methods
    -------
    get_values: Get values that parametrize the K-DPP.
    set_values: Set values that parametrize the K-DPP.
    marginals: Get the approximate values of the K-DPP.
    normalize: Normalize and recalculate the approximation.

    References
    ----------
    Barthelmé, S., Amblard, P. O., & Tremblay, N. (2018).
    Asymptotic equivalence of fixed-size and varying-size determinantal point
    processes. arXiv preprint arXiv:1803.01576.

    """

    def __init__(self, num_ground, k_sample):
        self.num_ground = num_ground
        self.k_sample = k_sample

        self._list = np.ones(self.num_ground)
        self._enu = self.k_sample / (1 - self.k_sample + self.num_ground)
        self.normalize()

    @property
    def values(self):
        """Get values that parametrize the K-DPP."""
        return self._list

    @values.setter
    def values(self, new_values):
        """Set values that parametrize the K-DPP."""
        assert len(new_values) == self.num_ground
        self._list = new_values

    @property
    def marginals(self):
        """Get the approximate marginals of the K-DPP."""
        return self._enu * self._list / (1 + self._enu * self._list)

    def normalize(self):
        """Normalize and recalculate the approximation."""
        self._enu = fsolve(
            lambda x: (np.sum(x * self._list / (1 + x * self._list)) - self.k_sample),
            self._enu,
        )
