"""train.py provides helpers for training using the ECVaR learning rule."""

import numpy as np
import torch

from adacvar.util.adaptive_algorithm import Exp3Sampler
from adacvar.util.classification_metrics import metrics
from adacvar.util.io import ClassificationLogger, ProgressPrinter

__author__ = "Sebastian Curi"
__all__ = ["cvar_train", "cvar_evaluate", "set_seeds"]


def cvar_train(
    model,
    optimizer,
    scheduler,
    criterion,
    cvar,
    train_loader,
    adaptive_algorithm,
    train_eval_loader,
    valid_loader,
    epochs,
    alpha,
    early_stopping,
    device,
    loggers,
    verbose,
):
    """Train a model that minimizes the Expected CVaR.

    Parameters
    ----------
    model: nn.Model
        Model to optimize.
    optimizer: nn.optimizer
        Optimizer used fit the `model' parameters.
    scheduler: nn.optim.scheduler.
    criterion: nn.Model
        Criterion used to evaluate the `model'.
    cvar: nn.Model
        CVaR Module to on top of the criterion.
    train_loader: torch.util.data.DataLoader
        Loader to iterate the training set.
    adaptive_algorithm: AdaptiveSamplingAlgorithm
        Algorithm with which to do adaptive sampling.
    train_eval_loader: torch.util.data.DataLoader
        Loader to iterate the training set.
    valid_loader: torch.util.data.DataLoader
        Loader to iterate the validation set.
    epochs: int
        Number of epochs to iterate the training set.
    alpha: float
        Fraction of tail of distribution to fit.
    early_stopping: EarlyStopping, optional.
        Early stopping object.
    device: string
        Device where to run training and evaluation.
    loggers: dict
        Dictionary with loggers.
    verbose: int
        Verbose Level.
    """
    printer = ProgressPrinter(epochs)

    cvar_evaluate(model, criterion, train_eval_loader, loggers["train"], alpha, device)
    cvar_evaluate(model, criterion, valid_loader, loggers["validation"], alpha, device)

    if verbose >= 3:
        printer.print_epoch(0, loggers["validation"])

    for epoch_idx in range(epochs):
        model.train()
        for batch_idx, (data, target, idx) in enumerate(train_loader):
            # Sample data
            data, target = data.to(device), target.to(device)

            # Reset optimizers
            optimizer.zero_grad()
            cvar.zero_grad()

            # Predict with model and calculate loss.
            out = model(data).squeeze()
            loss = criterion(out, target)

            # Find weights and probabilities according to algorithm.
            if isinstance(train_loader.batch_sampler, Exp3Sampler):
                # Enter here if the sampler is batch.
                weights = 1.0
                probabilities = adaptive_algorithm.probabilities
            else:
                # Enter here if the sampler is cyclic.
                weights = adaptive_algorithm.probabilities[idx]
                num_actions = adaptive_algorithm.num_actions
                probabilities = np.ones(num_actions) / num_actions

            # Feedback loss to sampler.
            adaptive_algorithm.update(
                1 - np.clip(loss.cpu().detach().numpy(), 0, 1), idx, probabilities
            )

            # Calculate CVaR and Reduce to mean.
            cvar_loss = (torch.tensor(weights).float().to(device) * cvar(loss)).mean()
            loss = loss.mean()

            # Compute Criterion and back-propagate.
            cvar_loss.backward()

            # Optimize model and cvar.
            optimizer.step()
            cvar.step()

            # Print training step
            if verbose >= 4:
                printer.print(
                    epoch_idx + 1, batch_idx, cvar=cvar_loss, loss=loss, var=cvar.var
                )

            # Renormalize sampler
            adaptive_algorithm.normalize()

            # Check numerical error!
            if np.isnan(cvar_loss.detach().item()):
                return

        # Evaluate after training epoch.
        cvar_evaluate(
            model, criterion, train_eval_loader, loggers["train"], alpha, device
        )
        cvar_evaluate(
            model, criterion, valid_loader, loggers["validation"], alpha, device
        )

        scheduler.step()

        if verbose >= 3:
            printer.print_epoch(epoch_idx + 1, loggers["validation"])

        if epoch_idx >= early_stopping.start_epoch:
            early_stopping.update(model, loggers["validation"]["cvar"][-1][-1])

        # if early_stopping.stop:
        #     early_stopping.restore_model(model)
        #     cvar_evaluate(model, criterion, valid_loader, loggers['validation'],
        #                   alpha, device)
        #     break

    # if early_stopping.started:  # and not early_stopping.stop: # Restore best model.
    # early_stopping.restore_model(model)
    # cvar_evaluate(model, criterion, valid_loader, loggers['validation'],
    #               alpha, device)

    if verbose >= 3:
        printer.print_epoch(epochs, loggers["validation"])


def cvar_evaluate(model, criterion, data_loader, logger, alpha, device):
    """Evaluate a model by computing its CVaR (Top-K average).

    Parameters
    ----------
    model: nn.Model
        Model to optimize.
    criterion: nn.Model
        Criterion used to evaluate the `model'.
    data_loader: torch.util.data.DataLoader
        Loader to iterate the dataset.
    logger: Logger
        Object where to log the data.
    alpha: float
        Fraction of tail of distribution to fit.
    device: string
        Device where to run training and evaluation.

    """
    model.eval()
    log = {key: 0 for key in logger.keys}
    k = int(np.ceil(alpha * len(data_loader.dataset)))
    top_k = None
    count = 0
    with torch.no_grad():
        for data, target, _ in data_loader:
            count += data.shape[0]
            data, target = data.to(device), target.to(device)

            # Predict with model
            out = model(data).squeeze()

            losses = criterion(out, target).sort(descending=True)[0]
            if top_k is None:
                top_k = losses[:k]
            else:
                top_k = (torch.cat((top_k, losses))).sort(descending=True)[0]
                top_k = top_k[:k]

            log["loss"] += losses.sum().item()

    if type(logger) is ClassificationLogger:
        metric_log = metrics(model, data_loader, criterion, device)
        for key, value in metric_log.items():
            logger.append(key, value)

    logger.append("loss", log["loss"] / count)
    logger.append("var", top_k[-1].detach().item())
    logger.append("cvar", top_k.mean().detach().item())


def set_seeds(random_seed=None):
    """Set random seed in all libraries.

    Parameters
    ----------
    random_seed: int, optional.

    """
    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
