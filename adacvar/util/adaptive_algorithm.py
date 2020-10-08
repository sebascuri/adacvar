"""adaptive_algorithm.py provides adaptive sampling algorithms."""

from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import Sampler

from adacvar.util.approximate_kdpp import ApproximateKDPP
from adacvar.util.learning_rate_decay import Constant

__author__ = "Sebastian Curi"
__all__ = ["Hedge", "Exp3", "Exp3Sampler"]


class AbstractExpertAlgorithm(ABC):
    """AbstractExpertAlgorithm is a template for expert algorithms.

    Parameters
    ----------
    num_actions: int
        Number of actions that the learner can select.
    size: int
        Size of actions that the learner can select.
    eta: callable or float
        Learning rate scheduler.
    gamma: float, optional
        Mixing of uniform distribution.

    Other Parameters
    ----------------
    eps: float, optional
        Estimate division for EXP3.IX.
    beta: float, optional
        Estimate bias for EXP3.P.


    Methods
    -------
    probabilities: Get the marginal probabilities of sampling each point
    update: Update algorithm after observing losses at the given indexes.
    normalize: Normalize the sampling distribution

    References
    ----------
    Cesa-Bianchi, N., & Lugosi, G. (2006). Prediction, learning, and games.
    Cambridge university press.

    """

    def __init__(self, num_actions, size, eta, gamma=0.0, eps=0.0, beta=0.0):
        self.num_actions = num_actions
        self.sample_size = size
        self._losses = np.ones(self.num_actions) / self.num_actions
        self._t = 1

        self._kdpp = ApproximateKDPP(
            num_ground=self.num_actions, k_sample=self.sample_size
        )

        if eta == 0 or eta is None:
            self.is_constant = True
        else:
            self.is_constant = False

        if not callable(eta):
            eta = Constant(eta, num_actions)
        self._eta = eta  # Algorithm Learning rate.
        self._gamma = gamma  # Mixture coefficient with uniform distribution
        self._eps = eps  # Loss estimate division for EXP3IX
        self._beta = beta  # Loss estimate bias for EXP3.P

    @property
    def probabilities(self):
        """Get the marginal probabilities of sampling each point."""
        q = self.sample_size / self.num_actions
        if self.is_constant:
            return q * np.ones(self.num_actions)

        p = self._kdpp.marginals
        return (1 - self._gamma) * p + self._gamma * q

    @abstractmethod
    def update(self, losses, indexes=None, probabilities=None):
        """Update algorithm after observing losses at the given indexes.

        Parameters
        ----------
        losses: np.ndarray
            Seen losses.
        indexes: np.ndarray
            Seen indexes.
        probabilities: np.ndarray
            If probabilities is not None, then sampling probabilities.
        """
        raise NotImplementedError

    def normalize(self):
        """Normalize the sampling distribution."""
        if not self.is_constant:
            self._kdpp.normalize()


class Hedge(AbstractExpertAlgorithm):
    """Implementation of Hedge algorithm.

    References
    ----------
    Littlestone, N., & Warmuth, M. K. (1994). The weighted majority algorithm.
    Information and computation, 108(2), 212-261.

    """

    def update(self, losses, indexes=None, probabilities=None):
        """See `AbstractExpertAlgorithm.update' for help."""
        self._t += 1

        val = self._kdpp.values
        eta = self._eta(self._t, losses, indexes)

        self._losses += losses
        val *= np.exp(-eta * losses)

        self._kdpp.values = val


class Exp3(Hedge):
    """Implementation of Exp3, Exp3.P, and Exp3.IX algorithms.

    References
    ----------
    Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002).
    The nonstochastic multiarmed bandit problem. SIAM journal on computing,
    32(1), 48-77.

    Neu, G. (2015). Explore no more: Improved high-probability regret bounds
    for non-stochastic bandits.
    In Advances in Neural Information Processing Systems (pp. 3168-3176).

    """

    def update(self, losses, indexes=None, probabilities=None):
        """See `AbstractExpertAlgorithm.update' for help."""
        loss_vector = np.zeros(self.num_actions)
        losses = np.clip(losses, 0, 1)

        if probabilities is None:
            probs = self.probabilities
        else:
            probs = probabilities

        np.add.at(loss_vector, indexes, losses / (probs[indexes] + self._eps))

        super().update(loss_vector)


class Exp3Sampler(Sampler):
    """Adaptive data sampler that is used as a BatchSampler in a DataLoader.

    The underlying algorithm is Exp3.

    The algorithm first sample a set of size `size' according to Exp3. In the
    second step it samples from this sample a set of size `batch_size', but
    uniformly at random and with replacement.

    Parameters
    ----------
    batch_size: int
        Batch size that the sampler should provide
    num_actions: int
        Number of actions that the learner can select.
    size: int
        Size of actions that the learner can select.
    eta: callable or float
        Learning rate scheduler.
    gamma: float, optional
        Mixing of uniform distribution.
    iid_batch: bool
        If true, then sample i.i.d. from the marginals with replacement a set
        of size `batch_size'.
        If false, sample without replacement a set of size `size'. Then sample
        from this subset a set of size `batch_size' with replacement.

    Other Parameters
    ----------------
    eps: float, optional
        Estimate division for EXP3.IX.
    beta: float, optional
        Estimate bias for EXP3.P.

    """

    def __init__(
        self, batch_size, num_actions, size, eta, gamma=0, eps=0, beta=0, iid_batch=True
    ):
        self._exp3 = Exp3(num_actions, size, eta, gamma, eps, beta)

        self.num_indexes = num_actions
        self.set_size = size
        self.batch_size = batch_size
        self.iid_batch = iid_batch

    def __len__(self):
        """Get the number of batches per epoch."""
        return self.num_indexes // self.batch_size

    def __iter__(self):
        """Iterate through the dataset."""
        for _ in range(len(self)):
            weights = self._exp3.probabilities
            if self._exp3.is_constant:
                idx = np.random.choice(self.num_indexes, size=self.batch_size)
            elif self.iid_batch:
                idx = np.random.choice(
                    self.num_indexes,
                    size=self.batch_size,
                    p=weights / np.sum(weights),
                    replace=True,
                )
            else:
                kdpp_sample = np.random.choice(
                    self.num_indexes,
                    self.set_size,
                    p=weights / np.sum(weights),
                    replace=False,
                )
                idx = np.random.choice(kdpp_sample, size=self.batch_size, replace=True)

            yield idx

    @property
    def probabilities(self):
        """Get the marginal probabilities of sampling each point."""
        return self._exp3.probabilities

    def update(self, losses, indexes=None, probabilities=None):
        """Update Exp3 distribution."""
        self._exp3.update(losses, indexes, probabilities)

    def normalize(self):
        """Normalize Exp3 distribution."""
        self._exp3.normalize()
