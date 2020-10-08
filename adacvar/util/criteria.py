"""criteria.py provides simple loss function implementations."""

import torch
import torch.nn as nn

__author__ = "Sebastian Curi"
__all__ = ["Hinge", "SVM"]


def _reduce_losses(losses, reduction):
    if reduction == "none":
        return losses
    elif reduction == "sum":
        return losses.sum()
    elif reduction == "mean":
        return losses.mean()
    else:
        raise NotImplementedError("reduction {} not implemented.".format(reduction))


class SVM(nn.Module):
    """Implementation of the SVM loss.

    The hinge loss is defined as:
    l(y, p) = max(0, 1 - y.p),
    where y is the true label and p is the predicted label, both in {-1, 1}.

    """

    def __init__(self, margin=1.0, reduction="none"):
        super().__init__()
        self._reduction = reduction
        self._margin = margin

    def forward(self, output, target):
        """Execute forward propagation of the SVM loss.

        Parameters
        ----------
        output: tensor [batch_size x 2]
            The output of a linear classifier.
        target: tensor [batch_size]
            The classes are labeled as {0, 1}

        Returns
        -------
        loss: tensor [batch_size]
        """
        assert output.shape[1] == 2
        assert output.shape[0] == target.shape[0]

        label = 2 * target.float() - 1
        losses = torch.relu(self._margin - output[:, 0] * label)
        return _reduce_losses(losses, self._reduction)


class Hinge(SVM):
    """Implementation of the hinge loss.

    The hinge loss is defined as:
    l(y, p) = max(0, -y.p),
    where y is the true label and p is the predicted label, both in {-1, 1}.

    """

    def __init__(self, reduction="none"):
        super().__init__(margin=0.0, reduction=reduction)
