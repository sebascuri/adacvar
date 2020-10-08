"""cvar.py module provides classes to optimize the CVaR."""

import torch
import torch.nn as nn

__author__ = "Sebastian Curi"
__all__ = ["CVaR", "SoftCVaR"]


class CVaR(nn.Module):
    """Evaluate the sample C-Var of a RV.

    If alpha -> 1, CVAR -> Expectation.
    If alpha -> 0, CVAR -> ESS SUP.

    Parameters
    ----------
    alpha: float
        tail of the distribution (between 0 and 1.).
    learning_rate: float
        learning rate of the internal parameter.

    References
    ----------
    Eq. (27) from Rockafellar, R. T., & Uryasev, S. (2000).
    Optimization of conditional value-at-risk. Journal of risk, 2, 21-42.

    """

    def __init__(self, alpha, learning_rate=1e-2):
        super(CVaR, self).__init__()
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in [0, 1].")

        self._alpha = alpha
        if alpha == 1:
            self._var = nn.Parameter(-torch.tensor(1.0), requires_grad=False)
        else:
            self._var = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self._optimizer = torch.optim.Adam([{"params": self._var, "lr": learning_rate}])

    def forward(self, losses):
        """Execute forward operation of CVaR module.

        output = VaR + 1 / alpha * max(losses - VaR, 0)

        Parameters
        ----------
        losses: torch.tensor

        Returns
        -------
        output: torch.tensor

        """
        return self._var + 1 / self._alpha * torch.relu(losses - self._var)

    def zero_grad(self):
        """Reset CVaR internal optimizer."""
        self._optimizer.zero_grad()

    def step(self):
        """Execute a step of the internal CVaR optimizer."""
        self._optimizer.step()

    @property
    def var(self):
        """Get estimated Value-at-Risk."""
        return self._var


class SoftCVaR(CVaR):
    """Evaluate the sample C-Var of a RV via a relaxed non-linearity.

    If alpha -> 1, CVAR -> Expectation.
    If alpha -> 0, CVAR -> ESS SUP.

    Parameters
    ----------
    alpha: float
        tail of the distribution (between 0 and 1.).
    learning_rate: float
        learning rate of the internal parameter.

    References
    ----------
    Nemirovski, A., & Shapiro, A. (2006).
    Convex approximations of chance constrained programs.
    SIAM Journal on Optimization, 17(4), 969-996.

    """

    def __init__(self, alpha, learning_rate=1e-3, temperature=1):
        super(SoftCVaR, self).__init__(alpha, learning_rate=learning_rate)
        if temperature <= 0:
            raise ValueError("Temperature must be larger than 0.")

        self._temperature = temperature

    def forward(self, losses):
        """Execute forward operation of CVaR module.

        output = VaR + T / alpha * log( 1 + exp((losses - VaR) / T)

        Parameters
        ----------
        losses: torch.tensor

        Returns
        -------
        output: torch.tensor

        """
        return self._var + self._temperature / self._alpha * torch.log(
            1 + torch.exp((losses - self._var) / self._temperature)
        )
