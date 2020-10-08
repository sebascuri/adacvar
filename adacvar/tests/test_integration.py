import os
from collections import namedtuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from adacvar.util.dataset import ClassificationDataset
from adacvar.util.early_stopping import EarlyStopping
from adacvar.util.load_data import OPEN_ML_TASKS, SYNTHETIC_TASKS, load_data
from adacvar.util.parsers import *
from adacvar.util.train import cvar_evaluate, cvar_train

TEST_COUNT = 0


@pytest.fixture(params=OPEN_ML_TASKS + SYNTHETIC_TASKS)
def dataset(request):
    return request.param


@pytest.fixture(params=["adacvar", "trunc_cvar", "mean", "soft_cvar"])
def algorithm(request):
    return request.param


class Experiment(namedtuple("Experiment", ["log_dir", "name"])):
    """Test Experiment."""

    def __new__(cls, log_dir, name):
        """Create new experiment."""
        try:
            os.makedirs(log_dir)
        except FileExistsError:
            pass
        return super().__new__(cls, log_dir, name)

    def __str__(self):
        """Return experiment name."""
        return self.name


def test_integration(dataset, algorithm):
    global TEST_COUNT
    TEST_COUNT += 1
    if dataset in ["madelon", "adult"]:
        return
    alpha = 0.1
    epochs = 1
    batch_size = 256
    learning_rate = 1e-3
    folds = np.random.choice(5)
    verbose = np.random.choice(5)

    if np.random.choice(2):
        device = "local"
    else:
        device = "cpu"

    if np.random.choice(2):
        sampler = "batch"
    else:
        sampler = "cyclic"

    m = np.random.choice(2)
    if m == 0:
        network = "linear"
    else:
        network = "FC20FC20"

    k = np.random.choice(3)
    if k == 0:
        scheduler = "constant"
    elif k == 1:
        scheduler = "robbins_monro"
    else:
        scheduler = "adagrad"

    loggers = parse_loggers(dataset)

    dataset, test_set = load_data(dataset, folds=folds)
    train_set, valid_set = dataset.get_split(0)
    if valid_set is None:
        valid_set = test_set

    if isinstance(train_set, ClassificationDataset):
        k = np.random.choice(3)
        if k == 0:
            criterion = "cross_entropy"
        elif k == 1:
            criterion = "hinge"
        else:
            criterion = "svm"
    else:
        if np.random.choice(2):
            criterion = "l2"
        else:
            criterion = "l1"

    device = parse_device(device)
    criterion = parse_criterion(criterion)

    model = parse_model(name=network, dataset=train_set)

    eta = parse_learning_rate(scheduler, -1, alpha, epochs, batch_size, len(train_set))

    train_loader, exp3 = parse_sampling(
        algorithm, sampler, alpha, train_set, batch_size, eta, 0.1, 0, 0
    )
    train_eval_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    cvar = parse_cvar(algorithm, alpha, learning_rate, temperature=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2], 0.1)

    early_stopping = EarlyStopping(
        Experiment("tmp/", "test_{}".format(TEST_COUNT)), np.random.choice(1)
    )

    for logger in loggers.values():
        logger.new_run()
    cvar_evaluate(model, criterion, test_loader, loggers["test"], alpha, device)
    cvar_train(
        model,
        optimizer,
        lr_scheduler,
        criterion,
        cvar,
        train_loader,
        exp3,
        train_eval_loader,
        valid_loader,
        epochs,
        alpha,
        early_stopping,
        "cpu",
        loggers,
        verbose,
    )
    cvar_evaluate(model, criterion, test_loader, loggers["test"], alpha, device)
