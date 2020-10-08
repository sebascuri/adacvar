"""Provides implementation of common classification metrics."""

import numpy as np
import torch
import torch.nn as nn

__author__ = "Sebastian Curi"
__all__ = ["print_metrics", "metrics"]


def _conf_matrix(model, data_loader, train_criterion, device="cpu"):
    """Build the confusion matrix that this classifier predicts.

    Parameters
    ----------
    model: nn.Model
    data_loader: torch.util.data.DataLoader
    train_criterion: nn.Model
    device: str

    Returns
    -------
    confusion matrix: nd_array
        matrix of size [n_classes x n_classes].

    """
    conf_matrix = np.zeros(
        (data_loader.dataset.number_classes, data_loader.dataset.number_classes)
    )
    with torch.no_grad():
        for data, target, _ in data_loader:
            data, target = data.to(device), target.to(device)

            # Predict with model
            if isinstance(train_criterion, nn.CrossEntropyLoss):
                predicted_labels = model(data).argmax(dim=1, keepdim=False)
            else:
                predicted_labels = (torch.sign(model(data)[:, 0]) + 1) / 2
                predicted_labels = predicted_labels.long()

            for t, p in zip(target.view(-1), predicted_labels.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

    return conf_matrix


def _accuracy(confusion_matrix_):
    """Get the accuracy of a classifier.

    Parameters
    ----------
    confusion_matrix_: nd_array
        matrix of size [n_classes x n_classes]

    Returns
    -------
    accuracy: float

    """
    return np.trace(confusion_matrix_) / np.sum(np.sum(confusion_matrix_))


def _recall(confusion_matrix_):
    """Get the recall of a classifier.

    Parameters
    ----------
    confusion_matrix_: nd_array
        matrix of size [n_classes x n_classes].

    Returns
    -------
    recall: float

    """
    true_positives = np.diag(confusion_matrix_)
    predictive_positives = np.sum(confusion_matrix_, axis=1)
    predictive_positives[predictive_positives == 0] = 1  # true positives are still 0
    return (true_positives / predictive_positives).min()


def _precision(confusion_matrix_):
    """Get the precision of a classifier.

    Parameters
    ----------
    confusion_matrix_: nd_array
        matrix of size [n_classes x n_classes].

    Returns
    -------
    precision: float

    """
    true_positives = np.diag(confusion_matrix_)
    actual_positives = np.sum(confusion_matrix_, axis=0)
    actual_positives[actual_positives == 0] = 1  # true positives are still 0
    return (true_positives / actual_positives).min()


def _f1(confusion_matrix_):
    """Get the F1-score of a classifier.

    Parameters
    ----------
    confusion_matrix_: nd_array
        matrix of size [n_classes x n_classes].

    Returns
    -------
    f1: float

    """
    recall = _recall(confusion_matrix_)
    precision = _precision(confusion_matrix_)
    sum_ = recall + precision
    if sum_ == 0:
        return 0
    else:
        return 2 * recall * precision / sum_


def print_metrics(model, data_loader, train_criterion, device="cpu"):
    """Print classification metrics.

    Parameters
    ----------
    model: nn.Model
        Model to evaluate.
    data_loader: DataLoader
        Object to load test data.
    train_criterion: nn.Model
        Training criterion used.
    device: string
        Device where to run the evaluation.

    """
    log = metrics(model, data_loader, train_criterion, device)
    print(log["confusion matrix"])
    print(
        "Accuracy: {}. Precision: {}. Recall: {}. F1: {}".format(
            log["accuracy"], log["precision"], log["recall"], log["f1"]
        )
    )


def metrics(model, data_loader, train_criterion, device="cpu"):
    """Get classification metrics.

    Parameters
    ----------
    model: nn.Model
        Model to evaluate.
    data_loader: DataLoader
        Object to load test data.
    train_criterion: nn.Model
        Training criterion used.
    device: string
        Device where to run the evaluation.

    Returns
    -------
    metrics: dict
        Dictionary with different classification metrics.

    """
    log = dict()
    log["confusion_matrix"] = _conf_matrix(model, data_loader, train_criterion, device)
    log["recall"] = _recall(log["confusion_matrix"])
    log["precision"] = _precision(log["confusion_matrix"])
    log["f1"] = _f1(log["confusion_matrix"])
    log["accuracy"] = _accuracy(log["confusion_matrix"])

    return log
