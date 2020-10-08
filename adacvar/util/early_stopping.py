"""Implementation of early stopping criterion."""
import os

from adacvar.util.io import load_model, save_model

__author__ = "Sebastian Curi"
__all__ = ["EarlyStopping"]


class EarlyStopping(object):
    """Implementation of early stopping algorithm.

    The algorithm will track the losses given by the model and store the LOWEST loss.
    If the early stopping flag is True, call the restore_model method.

    Parameters
    ----------
    experiment: Experiment
        Experiment data structure.
    patience: int
        Number of steps to wait before early stopping is implemented.
    start_epoch: int
        Epoch to start tracking the loss

    Methods
    -------
    update: update the early stopping value and model.
    restore_model: restore model to the best saved value.
    stop: flag that indicates if the algorithm is ready to stop.
    __del__: erase files created during early stopping tracking.

    """

    def __init__(self, experiment, patience, start_epoch=0):

        self._experiment = experiment
        self._dir = experiment.log_dir
        self._name_best = str(experiment) + "_best"
        self._patience = patience
        self._constant_steps = 0

        self.start_epoch = start_epoch
        self.started = False
        self.best_value = float("+Inf")

    def update(self, model, new_value):
        """Update the early stopping value and model.

        Parameters
        ----------
        model: nn.Module
            model to save.
        new_value: float
            value to track.

        """
        self.started = True
        if new_value < self.best_value:
            self.best_value = new_value
            save_model(self._experiment, model)
            self._constant_steps = 0
        else:
            self._constant_steps += 1

    def restore_model(self, model):
        """Restore model to the best saved value.

        Parameters
        ----------
        model: nn.Module
            model to save.

        Notes
        -----
        The model will mutate its parameters.

        """
        load_model(self._experiment, model)

    @property
    def stop(self):
        """Flag that indicates if the algorithm is ready to stop.

        Returns
        -------
        stop: bool

        """
        return self._constant_steps >= self._patience and self.started

    def __del__(self):
        """Erase files created during early stopping tracking."""
        os.system("rm -f {}".format(self._dir + self._name_best + ".pth"))
