"""io provides helper classes for input/output interactions of the project."""

import os
import pickle
import time

import torch
import yaml

__author__ = "Sebastian Curi"
__all__ = [
    "Logger",
    "RegressionLogger",
    "ClassificationLogger",
    "HTMLWriter",
    "ProgressPrinter",
    "save_model",
    "load_model",
    "save_object",
    "load_object",
    "save_args",
    "load_args",
]


def save_model(experiment, model):
    """Save torch model at a given directory with a given name.

    Parameters
    ----------
    experiment: Experiment
    model: nn.Model

    """
    file_name = experiment.log_dir + str(experiment) + ".pth"
    torch.save(model.state_dict(), file_name)


def load_model(experiment, model):
    """Load model state dictionary from a given directory and name.

    Parameters
    ----------
    experiment: Experiment
    model: nn.Model

    """
    file_name = experiment.log_dir + str(experiment) + ".pth"
    model.load_state_dict(torch.load(file_name))


def save_object(experiment, logs):
    """Save log at a given directory with a given name.

    Parameters
    ----------
    experiment: Experiment
    logs: dict

    """
    file_name = experiment.log_dir + str(experiment) + ".obj"
    with open(file_name, "wb") as file:
        pickle.dump(logs, file)


def load_object(experiment):
    """Get a log at a given directory with a given name.

    Parameters
    ----------
    experiment: Experiment

    Returns
    -------
    logs: dict

    """
    file_name = experiment.log_dir + str(experiment) + ".obj"
    with open(file_name, "rb") as file:
        return pickle.load(file)


def save_args(experiment, args):
    """Save arguments used to run an experiment.

    Parameters
    ----------
    experiment: Experiment

    """
    file_name = experiment.log_dir + str(experiment) + ".yml"
    with open(file_name, "w") as file:
        yaml.dump(args, file)


def load_args(experiment):
    """Load arguments used to run an experiment.

    Parameters
    ----------
    experiment: Experiment

    Returns
    -------
    args: dict

    """
    file_name = experiment.log_dir + str(experiment) + ".yml"
    with open(file_name, "r") as file:
        return yaml.load(file, Loader=yaml.SafeLoader)


class Logger(object):
    """Logger object to hold data with different keys and different runs.

    This object is specially useful for logging using k-fold cross-validation.

    Parameters
    ----------
    keys: iterable of str
        Keys to log.

    Methods
    -------
    new_run: start a new run and initialize logs.
    append: append a new value with a given key.
    """

    def __init__(self, keys):
        self._data = {}
        self.keys = keys
        for key in keys:
            self._data[key] = []

    def __getitem__(self, key):
        """Get log with a given key."""
        return self._data[key]

    def new_run(self):
        """Start a new run and initialize logs."""
        for key in self._data.keys():
            self._data[key].append([])

    def append(self, key, value):
        """Append a new value with a given key in current run.

        Parameters
        ----------
        key: string
            Key used for logging.
        value: float
            value to log.

        """
        self._data[key][-1].append(value)


class RegressionLogger(Logger):
    """Logger used for regression tasks."""

    def __init__(self):
        super().__init__(["loss", "cvar", "var"])


class ClassificationLogger(Logger):
    """Logger used for classification tasks."""

    def __init__(self):
        super().__init__(
            [
                "loss",
                "cvar",
                "var",
                "accuracy",
                "recall",
                "precision",
                "f1",
                "confusion_matrix",
            ]
        )


class HTMLWriter(object):
    """Writer to save an experiment in .html format.

    Parameters
    ----------
    file_name: string
        Name used to save the HTML file.
    headers: dict
        headers used to modify the HTML file.

    Methods
    -------
    add_title: add a title to the HTML file
    add_table: add a table to the HTML file
    add_image: add an image to the HTML file.
    """

    HEADER = ".html"

    def __init__(self, file_name, headers=None):
        if file_name[-5:] != self.HEADER:
            file_name += self.HEADER

        directory = "/".join(file_name.split("/")[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)

        self._file = open(file_name, "w+")
        message = """<html>\n<head>\n"""
        self._file.write(message)
        self._file.write('<meta charset="utf-8">')
        self._file.write('<meta author="Sebastian Curi">')
        if headers is not None:
            if "title" in headers.keys():
                message = "\t<title> {} </title>\n".format(headers["title"])
                self._file.write(message)
            self._add_style(headers)
        self._file.write("</head>\n<body>\n")

    def __del__(self):
        """Write a closing comment when the object is deleted."""
        self._file.write("""</body>\n</html>""")
        self._file.close()

    def _add_style(self, headers):
        self._file.write("<style>\n")
        self._file.write("img {width: 100%; height: auto;} \n")
        self._file.write("</style>\n")

    def add_title(self, title):
        """Add a title to the HTML file."""
        message = "<h1> {} </h1>\n".format(title)
        self._file.write(message)

    def add_table(self, headers, rows, caption=None, style=""):
        """Add a table to the HTML file."""
        message = '<table style = "{}"> \n'.format(style)
        self._file.write(message)
        message = "\t <tr> \n"
        self._file.write(message)
        for header in headers:
            message = "\t \t <td><b> {} </b></td> \n".format(header)
            self._file.write(message)
        message = "\t </tr> \n"
        self._file.write(message)

        for row in rows:
            message = "\t <tr> \n"
            self._file.write(message)
            for entry in row:
                message = "\t \t <td>{}</td> \n".format(entry)

                self._file.write(message)

            message = "\t </tr> \n"
            self._file.write(message)

        if caption is not None:
            message = """\t<caption align=\"top\" style=\"text-align:left\">
                      <b> {} </b> </caption>\n""".format(
                caption
            )
            self._file.write(message)

        message = "</table>\n <hr>\n"
        self._file.write(message)

    def add_image(self, file, style=""):
        """Add an image to the HTML file."""
        message = "\n<img src={} style={}>\n".format(file, style)
        self._file.write(message)


class ProgressPrinter(object):
    """Pretty Printer used for training.

    Parameters
    ----------
    epochs: int
        Number of epochs the training is expected to run.

    Methods
    -------
    print_epoch: print the running statistics before or after an epoch.
    print_train: print the running statistics during a training epoch.
    """

    def __init__(self, epochs):
        self._start_time = time.time()
        self._epochs = epochs

    def print_epoch(self, epoch_idx, log):
        """Print the running statistics before or after an epoch.

        Parameters
        ----------
        epoch_idx: int
            Epoch number.
        log: Logger

        """
        if epoch_idx == 0:
            msg = "Before Training"
        else:
            msg = "After {} Epochs".format(epoch_idx)

        msg += "\tLoss: {:.4f}, CVaR: {:.4f}, VaR: {:.4f}".format(
            log["loss"][-1][-1], log["cvar"][-1][-1], log["var"][-1][-1]
        )

        if "accuracy" in log.keys:
            msg += " Accuracy: {:.4f}".format(log["accuracy"][-1][-1])
        msg += "\tElapsed time: {}s".format(int(time.time() - self._start_time))
        print(msg)

    def print_train(self, epoch_idx, batch_idx, log):
        """Print the running statistics during a training epoch.

        Parameters
        ----------
        epoch_idx: int
            Epoch number.
        batch_idx:
            Batch number
        log: Logger

        """
        print(
            """Epoch[{}/{}] Batch[{}]. \t
        Loss: {:.4f}, CVaR: {:.4f}, VaR: {:.4f}\t Elapsed Time: {}s""".format(
                epoch_idx,
                self._epochs,
                batch_idx,
                log["loss"][-1][-1],
                log["cvar"][-1][-1],
                log["var"][-1][-1],
                int(time.time() - self._start_time),
            )
        )

    def print(self, epoch_idx, batch_idx, **kwargs):
        """Print the kwargs during a training epoch.

        Parameters
        ----------
        epoch_idx: int
            Epoch number.
        batch_idx:
            Batch number

        """
        str_ = "Epoch[{}/{}] Batch[{}] \t".format(epoch_idx, self._epochs, batch_idx)
        for key, val in kwargs.items():
            str_ = str_ + "{}: {:.4f} ".format(key.capitalize(), val)
        str_ = str_ + "\t Elapsed Time: {}s".format(int(time.time() - self._start_time))
        print(str_)
