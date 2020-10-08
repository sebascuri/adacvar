"""Parsers for run_experiments options."""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from adacvar.util.adaptive_algorithm import Exp3, Exp3Sampler
from adacvar.util.criteria import SVM, Hinge
from adacvar.util.cvar import CVaR, SoftCVaR
from adacvar.util.early_stopping import EarlyStopping
from adacvar.util.io import ClassificationLogger, RegressionLogger
from adacvar.util.learning_rate_decay import AdaGrad, Constant, RobbinsMonro
from adacvar.util.load_data import CLASSIFICATION_TASKS, REGRESSION_TASKS
from adacvar.util.models import LinearNet, ReLUNet

__author__ = "Sebastian Curi"
__all__ = [
    "Experiment",
    "parse_cvar",
    "parse_device",
    "parse_model",
    "parse_loggers",
    "parse_adaptive_algorithm",
    "parse_train_loader",
    "parse_sampling",
    "parse_learning_rate",
    "parse_criterion",
    "parse_task",
]


class Experiment(object):
    """Experiment data-structure."""

    def __init__(self, args):
        """Create new experiment."""
        for key in vars(args):
            setattr(self, key, getattr(args, key))

        if not self.early_stopping_steps:
            self.early_stopping_steps = self.epochs + 2

        if not self.early_stopping_start:
            self.early_stopping_start = self.epochs // 2

        # Parse device.
        self.device = parse_device(self.device)
        base_dir = os.environ.get("SCRATCH", os.environ.get("RESULTS", os.getcwd()))

        if self.shift_fraction is None:
            shift_str = "/{}".format(self.shift) if self.shift is not None else ""
            self.shift_fraction = self.alpha
        else:
            shift_str = "/{}".format(self.shift_fraction)

        if self.name is None:
            self.name = self.dataset
        self.log_dir = base_dir + "/experiments/runs/{}/{}/{}/{}/".format(
            self.name + shift_str, self.network, self.alpha, self.algorithm
        )

        try:
            os.makedirs(self.log_dir)
        except FileExistsError:
            pass

    def __str__(self):
        """Return experiment name."""
        if self.algorithm == "adacvar":
            str_ = "{}_" * 11 + "{}"
            exp_name = str_.format(
                self.network,
                self.shift,
                self.sampler,
                self.batch_size,
                self.epochs,
                self.learning_rate,
                self.early_stopping_start,
                self.early_stopping_steps,
                self.scheduler,
                self.eta,
                self.gamma,
                self.seed,
            )

        elif self.algorithm == "soft_cvar":
            str_ = "{}_" * 9 + "{}"
            exp_name = str_.format(
                self.network,
                self.shift,
                self.sampler,
                self.batch_size,
                self.epochs,
                self.learning_rate,
                self.early_stopping_start,
                self.early_stopping_steps,
                self.temperature,
                self.seed,
            )
        else:
            str_ = "{}_" * 8 + "{}"
            exp_name = str_.format(
                self.network,
                self.shift,
                self.sampler,
                self.batch_size,
                self.epochs,
                self.learning_rate,
                self.early_stopping_start,
                self.early_stopping_steps,
                self.seed,
            )

        if self.upsample:
            exp_name += "_upsample"
        elif self.downsample:
            exp_name += "_downsample"

        return exp_name

    def get_criterion(self):
        """Get criterion."""
        return parse_criterion(self.criterion)

    def get_early_stopping(self):
        """Get criterion."""
        return EarlyStopping(
            self,
            start_epoch=self.early_stopping_start,
            patience=self.early_stopping_steps,
        )

    def get_model(self, dataset):
        """Get model."""
        return parse_model(name=self.network, dataset=dataset)

    def get_train_loader(self, dataset):
        """Get train loader and adaptive algorithm."""
        eta = parse_learning_rate(
            self.scheduler,
            self.eta,
            self.alpha,
            self.epochs,
            self.batch_size,
            len(dataset),
        )

        return parse_sampling(
            self.algorithm,
            self.sampler,
            self.alpha,
            dataset,
            self.batch_size,
            eta,
            self.gamma,
            self.beta,
            self.eps,
        )

    def get_cvar(self):
        """Get CVaR layer."""
        return parse_cvar(
            self.algorithm, self.alpha, self.learning_rate, temperature=self.temperature
        )

    def get_optimizer(self, model):
        """Get model optimizer."""
        optimizer = parse_optimizer(
            self.optimizer,
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.milestones, gamma=self.lr_decay, last_epoch=-1
        )
        return optimizer, scheduler

    def get_loggers(self):
        """Get loggers for experiment."""
        return parse_loggers(self.dataset)

    def print(self):
        """Print experimental details."""
        str_ = "Dataset: {}. Algorithm: {}. Alpha: {}. Name: {}. Device: {}"
        print(
            str_.format(
                self.dataset, self.algorithm, self.alpha, str(self), self.device
            )
        )


def parse_cvar(algorithm, alpha, learning_rate=1e-3, temperature=0):
    """Parse and initialize the CVaR Object.

    If cvar is false, then CVaR with alpha=1 is returned (which is the mean).
    If temperature > 0, then the CVaR is relaxed via the log(1+exp) relaxation.

    Parameters
    ----------
    algorithm: string
        Learning rule.
    alpha: float
        Mass of tail of the distribution to average.
    learning_rate: float
        Learning rate to update the VaR estimate.
    temperature: float
        Temperature -> 0 then the relaxation -> exact.
        Temperature -> inf, then the relaxation is loose and cvar -> mean,
        for all alpha.

    Returns
    -------
    cvar: CVaR

    """
    if algorithm == "soft_cvar":
        return SoftCVaR(
            alpha=alpha, learning_rate=learning_rate, temperature=temperature
        )
    elif algorithm == "trunc_cvar":
        return CVaR(alpha=alpha, learning_rate=learning_rate)
    elif algorithm == "mean" or algorithm == "adacvar":
        return CVaR(alpha=1, learning_rate=0)  # don't optimize the CVaR here!
    else:
        raise ValueError(
            """
        algorithm {} wrongly parsed. Available options are `soft_cvar', `trunc_cvar',
        `mean', or `adacvar'.""".format(
                algorithm
            )
        )


def parse_device(device):
    """Parse device where to run the algorithm.

    If device local and Cuda is available return GPU with largest free memory.


    Parameters
    ----------
    device: string
        'local', 'cpu', or 'cuda:x'

    Returns
    -------
    device: string

    """
    if device == "local":
        if torch.cuda.is_available():
            device = "cuda:{}".format(_get_free_gpu())
        else:
            device = "cpu"

    if not ("cpu" in device or "cuda" in device):
        raise ValueError(
            """
        device {} wrongly parsed. Available options are `cpu', `cuda:x' or
        `local'.""".format(
                device
            )
        )

    return device


def parse_model(name, dataset):
    """Parse and initialize the model used.

    Parameters
    ----------
    name: string
    dataset: cvar.dataset.Dataset

    Returns
    -------
    model: nn.Model

    """
    in_channels = dataset.number_channels
    in_features = dataset.number_features
    try:
        out_features = dataset.number_classes  # Classification dataset.
    except AttributeError:
        out_features = dataset.target_size  # Regression dataset.

    if name == "linear":
        model = LinearNet(name, in_channels, in_features, out_features)
    elif name in ReLUNet.implementations:
        model = ReLUNet(name, in_channels, in_features, out_features)
    elif name in models.vgg.__all__:
        model = getattr(models.vgg, name)(num_classes=out_features)
    elif name in models.resnet.__all__:
        model = getattr(models.resnet, name)(num_classes=out_features)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        raise ValueError(
            """network name {} wrongly parsed. Available options
        are `linear', in ReLUNet.implementations or vgg implementations
        """.format(
                name
            )
        )

    return model


def parse_criterion(criterion_name):
    """Parse criterion used in experiment."""
    if criterion_name == "l2" or criterion_name == "mse":
        return nn.MSELoss(reduction="none")
    elif criterion_name == "l1":
        return nn.L1Loss(reduction="none")
    elif criterion_name == "cross_entropy":
        return nn.CrossEntropyLoss(reduction="none")
    elif criterion_name == "hinge":
        return Hinge(reduction="none")
    elif criterion_name == "svm":
        return SVM(reduction="none")
    else:
        raise ValueError(
            """
        criterion {} wrongly parsed. Available options are `l2', `mse', `l1',
        `hinge', `svm', or `cross_entropy'.""".format(
                criterion_name
            )
        )


def parse_optimizer(optimizer_name, model_params, lr, momentum=None, weight_decay=None):
    """Parse optimizer used in experiment."""
    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model_params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(
            """optimizer {} wrongly parsed. Available options are `Adam' or
            `SGD'""".format(
                optimizer_name.lower()
            )
        )
    return optimizer


def parse_learning_rate(scheduler, eta, alpha, epochs, batch_size, num_points):
    """Parse learning rate decay scheduler.

    Parameters
    ----------
    scheduler: string
        name of scheduling strategy.
    eta: float, optional (default: optimal)
        Algorithm initial learning rate.
    alpha: float
        Fraction of distribution tail.
    num_points: int
        Number of points.
    epochs: int
        Number of epochs the sampler has to run
    batch_size: int
        Batch size for the sampler.

    Returns
    -------
    scheduler: LearningRateDecay
        Scheduler of EXP3 learning rate.

    """
    horizon = epochs * (num_points // batch_size)
    if eta is None or eta < 0:
        eta = np.sqrt(1 / alpha * np.log(1 / alpha))

    if scheduler == "constant":
        eta = Constant(eta / np.sqrt(horizon), num_points)
    elif scheduler == "robbins_monro":
        eta = RobbinsMonro(eta, num_points)
    elif scheduler == "adagrad":
        eta = AdaGrad(eta, num_points)
    else:
        raise ValueError(
            """
        scheduler {} wrongly parsed. Available options are `constant',
        `robbins_monro', or `ada_grad'.""".format(
                scheduler
            )
        )
    return eta


def parse_sampling(
    algorithm, sampler, alpha, train_set, batch_size, eta, gamma=0, beta=0, eps=0
):
    """Parse sampling strategy and training algorithm as these are coupled.

    Parameters
    ----------
    algorithm: string
        algorithm name.
    sampler: string
        Sampler name.
    alpha: : float
        Fraction of distribution tail to optimize.
    train_set: Dataset
        Dataset to optimize.
    batch_size: int
        Batch size for the sampler.
    eta: LearningRateScheduler
        Scheduler of EXP3 learning rate.
    gamma: float, optional (default: 0)
        Algorithm mixing with uniform distribution.
    beta: float, optional (default: 0)
        Bias for high-probability bounds (Exp3.P).
    eps: float, optional (default: 0)
        Bias for high-probability bounds (Exp3.IX).

    Returns
    -------
    train_loader: DataLoader
        Loader of training set.
    adaptive_algorithm: AdaptiveAlgorithm
        Adaptive algorithm to update.

    """
    exp3 = parse_adaptive_algorithm(
        algorithm, sampler, alpha, len(train_set), batch_size, eta, gamma, beta, eps
    )
    loader = parse_train_loader(sampler, train_set, batch_size, adaptive_sampler=exp3)

    if sampler == "cyclic":
        return loader, exp3
    else:
        return loader, loader.batch_sampler


def parse_adaptive_algorithm(
    algorithm, sampler, alpha, num_points, batch_size, eta, gamma=0, beta=0, eps=0
):
    """Parse and initialize the Sampler.

    Parameters
    ----------
    algorithm: string
        algorithm name.
    sampler: string
        Sampler name.
    alpha: float
        Fraction of distribution tail.
    num_points: int
        Number of points.
    batch_size: int
        Batch size for the sampler.
    eta: LearningRateScheduler
        Scheduler of EXP3 learning rate.
    gamma: float, optional (default: 0)
        Algorithm mixing with uniform distribution.
    beta: float, optional (default: 0)
        Bias for high-probability bounds (Exp3.P).
    eps: float, optional (default: 0)
        Bias for high-probability bounds (Exp3.IX).

    Returns
    -------
    sampler: Sampler
        sampler used as batch_sampler.
    """
    k = int(np.ceil(alpha * num_points))
    if algorithm != "adacvar":
        eta = 0
        gamma = 0
        beta = 0
        eps = 1e-16
        iid_batch = True
    else:
        iid_batch = False

    if sampler == "batch":
        return Exp3Sampler(
            batch_size=batch_size,
            num_actions=num_points,
            size=k,
            eta=eta,
            gamma=gamma,
            beta=beta,
            eps=eps,
            iid_batch=iid_batch,
        )
    elif sampler == "cyclic":
        return Exp3(
            num_actions=num_points, size=k, eta=eta, gamma=gamma, beta=beta, eps=eps
        )
    else:
        raise ValueError(
            """
        sampler {} wrongly parsed. Available options are `batch' or `cyclic'.
        """.format(
                sampler
            )
        )


def parse_train_loader(sampler, train_set, batch_size, adaptive_sampler=None):
    """Parse train loader according to batch or cyclic sampling strategy.

    Parameters
    ----------
    sampler: string
        Either batch or cyclic sampling strategy.
    train_set: torch.util.data.Dataset
        Training set.
    batch_size: int
        Batch size to sample.
    adaptive_sampler: Exp3Sampler, optional
        Batch Sampling strategy.

    Returns
    -------
    data_loader: DataLoader

    """
    if sampler == "batch":
        return DataLoader(train_set, batch_sampler=adaptive_sampler)
    elif sampler == "cyclic":
        return DataLoader(train_set, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError(
            """
        sampler {} wrongly parsed. Available options are `batch' or `cyclic'.
        """.format(
                sampler
            )
        )


def parse_task(dataset):
    """Parse a task for the dataset."""
    if dataset in CLASSIFICATION_TASKS:
        return "classification"
    elif dataset in REGRESSION_TASKS:
        return "regression"
    else:
        raise ValueError("dataset {} wrongly parsed".format(dataset))


def parse_loggers(dataset):
    """Parse loggers for a dataset."""
    task = parse_task(dataset)
    if task == "regression":
        logger_ = RegressionLogger
    elif task == "classification":
        logger_ = ClassificationLogger
    else:
        raise NotImplementedError("Task {} not implemented".format(task))

    return {"train": logger_(), "validation": logger_(), "test": logger_()}


def _get_free_gpu():
    """Get the GPU with largest free memory.

    Returns
    -------
    gpu: int

    """
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_gpu")
    memory_available = [int(x.split()[2]) for x in open("tmp_gpu", "r").readlines()]
    return np.argmax(memory_available)
