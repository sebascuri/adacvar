"""Main file of the module."""

import argparse
import os
from sys import exit

import numpy as np
import torch
from torch.utils.data import DataLoader

from adacvar.util.io import save_args, save_model, save_object
from adacvar.util.load_data import load_data
from adacvar.util.parsers import Experiment
from adacvar.util.train import cvar_evaluate, cvar_train, set_seeds


def main(experiment):
    """Run main application of the module."""
    if os.path.isfile(experiment.log_dir + str(experiment) + ".obj"):
        exit(1)

    # Set seed.
    set_seeds(experiment.seed)

    # Initialize loggers:
    loggers = experiment.get_loggers()

    if experiment.verbose >= 1:  # Print Info.
        experiment.print()

    # Initialize criterion.
    criterion = experiment.get_criterion()

    # Initialize Early stopping
    early_stopping = experiment.get_early_stopping()

    # Initialize loggers
    for logger in loggers.values():
        logger.new_run()

    # Load Datasets
    dataset, test_set = load_data(
        experiment.dataset,
        folds=1,
        random_seed=experiment.seed,
        shift=experiment.shift,
        shift_fraction=experiment.shift_fraction,
        upsample=experiment.upsample,
        downsample=experiment.downsample,
    )
    train_set, valid_set = dataset.get_split()
    if valid_set is None:
        valid_set = train_set

    # Initialize model
    model = experiment.get_model(train_set)

    # Initialize Samplers and Data Loaders.
    train_loader, exp3 = experiment.get_train_loader(train_set)
    train_eval_loader = DataLoader(train_set, batch_size=experiment.test_batch_size)
    valid_loader = DataLoader(valid_set, batch_size=experiment.test_batch_size)
    test_loader = DataLoader(test_set, batch_size=experiment.test_batch_size)

    # Initialize cvar and optimizer
    cvar = experiment.get_cvar()
    optimizer, scheduler = experiment.get_optimizer(model)

    # Send modules to device
    for module in [model, criterion, cvar]:
        module.to(experiment.device)

    # Train model
    cvar_train(
        model,
        optimizer,
        scheduler,
        criterion,
        cvar,
        train_loader,
        exp3,
        train_eval_loader,
        valid_loader,
        experiment.epochs,
        experiment.alpha,
        early_stopping,
        experiment.device,
        loggers,
        experiment.verbose,
    )

    # Test model
    cvar_evaluate(
        model,
        criterion,
        test_loader,
        loggers["test"],
        experiment.alpha,
        experiment.device,
    )

    # Print Test log
    if experiment.verbose >= 2:
        print("test", experiment)
        for key in loggers["test"].keys:
            if key == "confusion_matrix":
                continue
            log = loggers["test"][key]
            print("test", key, np.mean(log))
            log = np.array(loggers["train"][key])
            print("train", key, np.mean(log[:, -1]))
        print()

    # Save Logs
    save_object(experiment, loggers)
    save_args(experiment, vars(experiment))
    save_model(experiment, model)

    return loggers["test"]["cvar"][-1][-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the CVaR.")
    # TASK ARGUMENTS
    parser.add_argument("--name", type=str, default=None, help="experiment name.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="boston",
        help="name of dataset (default: boston)",
    )
    parser.add_argument(
        "--shift",
        type=str,
        default=None,
        help="""dataset shift (default: None)
                                test: test set shift.
                                train: train set shift.
                                """,
    )
    parser.add_argument(
        "--shift-fraction",
        type=float,
        default=None,
        help="""dataset shift fraction (default: None)""",
    )
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--downsample", action="store_true")

    parser.add_argument(
        "--alpha", type=float, default=0.1, help="alpha of CVAR (default: 0.1)"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="adacvar",
        help="""algorithm used (default adacvar)
                        mean: optimize the ERM.
                        trunc_cvar: optimize the Truncated CVaR.
                        soft_cvar: optimize the Soft relaxation of the CVaR.
                        adacvar: optimize the mean but with adaptive sampling.
                        """,
    )

    # GAME ARGUMENTS
    parser.add_argument(
        "--criterion",
        type=str,
        default="l2",
        help="""loss function used (default l2)
                        l2: mse loss function.
                        l1: l1 loss function.
                        hinge: hinge loss function.
                        svm: svm loss function.
                        cross_entropy: logistic regression.
                        """,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size for training set (default: 32)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=256,
        help="batch size for test set (default: 256)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs to train (default: 10)"
    )

    # LEARNER ARGUMENTS
    parser.add_argument(
        "--network",
        type=str,
        default="linear",
        help="""classifier network (default: linear)
                        for other look at models sub-folder.
                        """,
    )
    parser.add_argument(
        "--optimizer", type=str, default="SGD", help="""Optimizer name, adam or sgd."""
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.1, help="momentum (default: 0.001)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        help="L-2 Regularization (default: 1e-5)",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=[20, 40],
        help="epochs in which LR is decreased.",
    )
    parser.add_argument(
        "--lr-decay", type=float, default=1.0, help="Decrease factor of learning rate.."
    )

    parser.add_argument(
        "--early-stopping-steps",
        type=int,
        default=None,
        help="""Number of steps to wait before stopping (default: None).
                        If None, then the early stopping will be set to num_epochs + 2
                        so it does not stop.
                        """,
    )

    parser.add_argument(
        "--early-stopping-start",
        type=int,
        default=None,
        help="""Number of steps to wait before stopping (default: None).
                        If None, then the early stopping will start after half the
                        number of epochs.
                        """,
    )

    parser.add_argument(
        "--sampler",
        type=str,
        default="batch",
        help="""sampler used (default: batch)
                        batch: iterate through dataset through random batches.
                        cyclic: iterate cyclically through dataset.
                        """,
    )

    # SAMPLER ARGUMENTS
    parser.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Exp3 learning rate (default: no-regret optimal)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="constant",
        help="""Exp3 eta rate schedule (default: constant)
                        constant: no-regret rate or set rate
                        robbins_monro: decay as O(1/sqrt(T))
                        adagrad: adaptive method.
                        """,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="mixture of adaptive with uniform (default: 0.0",
    )
    parser.add_argument(
        "--eps", type=float, default=0.0, help="Exp3IX bias (default: 0.0)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.0, help="Exp3.P bias (default: 0.0)"
    )
    # SOFT PARAMETERS
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of CVaR relaxation (default: 1.)",
    )

    # EXPERIMENTAL PARAMETERS
    parser.add_argument(
        "--seed", type=int, default=1, help="initial random seed (default: 0)"
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="""Verbose Level (default: 1).
                        0: No print at all.
                        1: Print run info.
                        2: Print test set.
                        3: Print validation.
                        4: Print training.""",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="local",
        help="""Device that runs algorithm (default: local)
                        gpu: run at gpu
                        cpu: run at cpu
                        """,
    )
    parser.add_argument(
        "--num-threads", type=int, default=4, help="Number of threads to run pytorch."
    )

    args_ = parser.parse_args()
    torch.set_num_threads(args_.num_threads)
    experiment = Experiment(args_)
    main(experiment)
