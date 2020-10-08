import pytest
import torch
import torch.testing

from adacvar.util.criteria import SVM, Hinge


@pytest.fixture(params=["none", "sum", "mean", "other"])
def reduction(request):
    return request.param


@pytest.fixture(params=[SVM, Hinge])
def criterion_(request):
    return request.param


def test_svm(reduction, criterion_):
    criterion = criterion_(reduction=reduction)
    if isinstance(criterion, Hinge):
        margin = 0
    else:
        margin = 1

    pred = 2 * torch.rand((32, 2)) - 1
    true = torch.randint(2, (32,))

    if reduction == "other":
        with pytest.raises(NotImplementedError):
            criterion(pred, true)
            return
    else:
        loss = criterion(pred, true)
        losses = torch.relu(margin - pred[:, 0] * (2 * true.float() - 1))

        if reduction == "sum":
            losses = losses.sum()
        elif reduction == "mean":
            losses = losses.mean()

        torch.testing.assert_allclose(loss, losses)
