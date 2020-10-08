"""cvar.py module provides classes to handle datasets in the project."""


import numpy as np
import torch
import torch.utils.data

__author__ = "Sebastian Curi"
__all__ = [
    "Dataset",
    "ClassificationDataset",
    "RegressionDataset",
    "VisionDatasetWrapper",
    "KFoldDataset",
    "TrainValidationDataset",
]


class Dataset(torch.utils.data.Dataset):
    """Basic Dataset class.

    Parameters
    ----------
    data: array_like [num_points x channels x features]
    target: array_like [num_points x num_targets]

    """

    def __init__(self, data, target):
        self.data = data
        self.target = target
        assert (
            self.data.shape[0] == self.target.shape[0]
        ), "Data and target should have the same number of points."

    def __len__(self):
        """Get the number of points in the dataset."""
        return self.data.shape[0]

    def __getitem__(self, idx):
        """Get the data, target and indexes."""
        return self.data[idx], self.target[idx], idx

    @property
    def number_points(self):
        """Get number of points in dataset."""
        return len(self)

    @property
    def number_channels(self):
        """Get number of channels in dataset covariates."""
        return self.data.shape[1]

    @property
    def number_features(self):
        """Get number of features in dataset covariates."""
        return self.data.shape[2:]


class ClassificationDataset(Dataset):
    """Classification Dataset class.

    Parameters
    ----------
    data: array_like [num_points x channels x features]
    target: array_like [num_points x 1]
        The target is a value that indicates the class.

    """

    def __init__(self, data, target):
        super().__init__(data, target)

    @property
    def number_classes(self):
        """Get the number of classes in the dataset."""
        return len(np.unique(self.target))


class RegressionDataset(Dataset):
    """Regression Dataset class.

    Parameters
    ----------
    data: array_like [num_points x channels x features]
    target: array_like [num_points x num_targets]

    """

    def __init__(self, data, target):
        super().__init__(data, target)

    @property
    def target_size(self):
        """Get the size of the regression target."""
        if len(self.target.shape) == 1:
            return 1
        else:
            return self.target.shape[1]


class VisionDatasetWrapper(torch.utils.data.Dataset):
    """VisionDatasetWrapper gets a vision dataset and wraps it to a Dataset.

    Parameters
    ----------
    dataset: torch.utils.data.Dataset

    """

    def __init__(self, dataset, classification=True):
        self.dataset = dataset
        self.number_points = len(dataset)
        data, _ = dataset[0]
        self.number_channels = data.shape[0]
        self.number_features = (data.shape[1], data.shape[2])

        if classification:
            self.number_classes = 10

        self.target_size = 1

    def __len__(self):
        """Return dataset length."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """See `ClassificationDataset.__getitem__' for help ."""
        data, target = self.dataset[idx]
        if hasattr(self, "number_classes"):
            return data, target, idx
        else:
            return data, torch.tensor(target / 10.0), idx


class KFoldDataset(object):
    """A K-fold Dataset class.

    Given a dataset, it splits it into `folds'-folds.

    Methods
    -------
    get_split: return the train and validation sets of a given split.

    Parameters
    ----------
    dataset: Dataset
    folds: int, optional

    """

    def __init__(self, dataset, folds=None):
        super().__init__()
        self.dataset = dataset
        self.folds = folds if folds else 1
        self._fold_idx = 0
        self._train_indexes = {}
        self._valid_indexes = {}

        indexes = np.arange(len(dataset))
        if self.folds > 1:
            validation_indexes = np.split(indexes[(len(dataset) % folds) :], folds)
            for idx in range(folds):
                valid_idx = validation_indexes[idx]
                train_idx = np.delete(indexes, valid_idx)
                self._train_indexes[idx] = train_idx
                self._valid_indexes[idx] = valid_idx

        else:
            self._train_indexes[0] = indexes
            self._valid_indexes[0] = indexes

        assert len(self._train_indexes) == self.folds
        assert len(self._valid_indexes) == self.folds

    def get_split(self, fold_idx=0):
        """Get the train and validation set for a split.

        Parameters
        ----------
        fold_idx: int
            split number

        Returns
        -------
        train_set: Dataset
        test_set: Dataset

        """
        assert 0 <= fold_idx < self.folds
        if self.folds == 1:
            return self.dataset, None
        else:
            return self._train_set(fold_idx), self._validation_set(fold_idx)

    def _train_set(self, fold_idx):
        indexes = self._train_indexes[fold_idx]
        class_ = type(self.dataset)
        data, target, _ = self.dataset[indexes]
        return class_(data, target)

    def _validation_set(self, fold_idx):
        indexes = self._valid_indexes[fold_idx]
        class_ = type(self.dataset)
        data, target, _ = self.dataset[indexes]
        return class_(data, target)


class TrainValidationDataset(object):
    """A Train-Validation split Dataset class.

    Given a dataset, it splits it into `folds'-folds.

    Methods
    -------
    get_split: return the train and validation sets of a given split.

    Parameters
    ----------
    train_set: Dataset
    validation_set: Dataset

    """

    def __init__(self, train_set, validation_set):
        self.train_set = train_set
        self.validation_set = validation_set

    def get_split(self, _=None):
        """Get the train and validation set for a split.

        Parameters
        ----------
        _: int
            Dummy parameter for compatibility.

        Returns
        -------
        train_set: Dataset
        test_set: Dataset

        """
        return self.train_set, self.validation_set
