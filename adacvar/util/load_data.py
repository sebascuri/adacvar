"""Load data with corresponding pre-processing techniques."""

import os
from collections import defaultdict

import numpy as np
import scipy
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, label_binarize
from torch.utils.data import Subset
from torchvision import datasets, transforms

from adacvar.util.dataset import (
    ClassificationDataset,
    KFoldDataset,
    RegressionDataset,
    TrainValidationDataset,
    VisionDatasetWrapper,
)

__author__ = "Sebastian Curi"
__all__ = ["load_data", "CLASSIFICATION_TASKS", "REGRESSION_TASKS"]

OPEN_ML_CLASSIFICATION_TASKS = {
    "Australian": 4,
    "monks-problems-1": 1,
    "madelon": 1,
    "splice": 2,
    "spambase": 1,
    "Titanic": 2,
    "phoneme": 1,
    "german.numer": 1,
    "adult": 2,
}

OPEN_ML_REGRESSION_TASKS = {
    "boston": 1,
    "cpu_small": 1,
    "abalone": 1,
    "energy-efficiency": 1,
}

OPEN_ML_TASKS = list(OPEN_ML_CLASSIFICATION_TASKS.keys())
OPEN_ML_TASKS += list(OPEN_ML_REGRESSION_TASKS.keys())

SYNTHETIC_REGRESSION_TASKS = ["sinc", "pareto", "normal"]
SYNTHETIC_CLASSIFICATION_TASKS = []
SYNTHETIC_TASKS = SYNTHETIC_REGRESSION_TASKS + SYNTHETIC_CLASSIFICATION_TASKS

VISION_CLASSIFICATION_TASKS = [
    "mnist",
    "cifar-10",
    "fashion-mnist",
    "mnist-augmented",
    "cifar-10-augmented",
    "fashion-mnist-augmented",
]
VISION_REGRESSION_TASKS = [
    "mnist-regression",
    "cifar-10-regression",
    "fashion-mnist-regression",
]
VISION_TASKS = VISION_CLASSIFICATION_TASKS + VISION_REGRESSION_TASKS

CLASSIFICATION_TASKS = (
    list(OPEN_ML_CLASSIFICATION_TASKS)
    + SYNTHETIC_CLASSIFICATION_TASKS
    + VISION_CLASSIFICATION_TASKS
)
REGRESSION_TASKS = (
    list(OPEN_ML_REGRESSION_TASKS)
    + SYNTHETIC_REGRESSION_TASKS
    + VISION_REGRESSION_TASKS
)


DATA_DIR = os.environ.get("DATA", os.path.join("~", "data"))


def load_data(
    name,
    folds=5,
    test_fraction=0.2,
    random_seed=None,
    shift=None,
    shift_fraction=0.1,
    upsample=False,
    downsample=False,
):
    """Load dataset.

    Parameters
    ----------
    name: string
        dataset name.
    folds: int, optional
        number of splits of dataset (default=5).
    test_fraction: float, optional
        fraction of dataset to split into test set (default=0.2).
    random_seed: int, optional
        random seed for pre-processing and splitting data (default=None).
    shift: str, optional.
        String that indicates how to shift data.
    shift_fraction: float, optional.
        how much to shift the data distribution.
    upsample: bool, optional.
        Upsample shifted data.
    downsample: bool, optional.
        Downsample shifted data.

    Returns
    -------
    dataset: Dataset
    test_set: Dataset

    """
    # assert not (test_shift and train_shift), "Only allow train or test shift."
    assert not (upsample and downsample), "Only allow to upsample or downsample."

    if name in OPEN_ML_TASKS:
        return _load_open_ml(
            name,
            folds,
            test_fraction,
            random_seed,
            shift=shift,
            shift_fraction=shift_fraction,
            upsample=upsample,
            downsample=downsample,
        )
    elif name in SYNTHETIC_TASKS:
        return _load_synthetic_data(name, folds, test_fraction, random_seed)
    elif name in VISION_TASKS:
        return _load_vision(
            name, shift == "train", shift_fraction, upsample, downsample
        )
    else:
        raise ValueError("Name {} badly parsed".format(name))


def _load_open_ml(
    name,
    folds=5,
    test_fraction=0.2,
    random_seed=None,
    shift=None,
    shift_fraction=0.1,
    upsample=False,
    downsample=False,
):
    """Load Open ML dataset.

    Parameters
    ----------
    name: string
        dataset name.
    folds: int, optional
        number of splits of dataset (default=5).
    test_fraction: float, optional
        fraction of dataset to split into test set (default=0.2).
    random_seed: int, optional
        random seed for pre-processing and splitting data (default=None).

    Returns
    -------
    dataset: Dataset
    test_set: Dataset

    """
    if name in OPEN_ML_REGRESSION_TASKS:
        binarize = False
        version = OPEN_ML_REGRESSION_TASKS[name]
        class_ = RegressionDataset
    elif name in OPEN_ML_CLASSIFICATION_TASKS:
        binarize = True
        version = OPEN_ML_CLASSIFICATION_TASKS[name]
        class_ = ClassificationDataset
    else:
        raise ValueError("name {} not parsed".format(name))

    dataset = fetch_openml(name, return_X_y=False, version=version)

    data = dataset.data
    if type(data) is not np.ndarray:
        data = data.toarray()
    data = pre_process_data(data, dataset.feature_names, dataset.categories)

    data = torch.from_numpy(data).float()

    target = dataset.target
    if binarize:
        target = ohe_target(target)
        target = torch.from_numpy(target)
    else:
        target = target.astype("float")
        target = (target - np.mean(target)) / np.std(target)
        target = torch.from_numpy(target).float()

    data = np.expand_dims(data, 1)

    rng = np.random.RandomState(random_seed)
    indexes = np.arange(data.shape[0])
    rng.shuffle(indexes)
    train_indexes, test_indexes = np.split(
        indexes, [int(data.shape[0] * (1 - test_fraction))]
    )

    if name in OPEN_ML_CLASSIFICATION_TASKS:
        # Make the majority class a minority
        maj = int(target.float().mean())
        if shift is None:
            shift = ""
        if shift.lower() == "test":
            test_indexes = _reshift_indexes(test_indexes, target, maj, shift_fraction)
        elif shift.lower() == "train":
            train_indexes = _reshift_indexes(
                train_indexes, target, maj, shift_fraction, upsample, downsample
            )
        elif shift.lower() == "both":
            train_indexes = _reshift_indexes(
                train_indexes, target, maj, shift_fraction, upsample, downsample
            )
            test_indexes = _reshift_indexes(
                test_indexes, target, 1 - maj, shift_fraction
            )
        elif shift.lower() == "double":
            train_indexes = _reshift_indexes(
                train_indexes, target, maj, shift_fraction, upsample, downsample
            )
            test_indexes = _reshift_indexes(
                test_indexes, target, maj, shift_fraction ** 2
            )

    train_set = class_(data=data[train_indexes], target=target[train_indexes])
    train_set = KFoldDataset(train_set, folds=folds)
    test_set = class_(data[test_indexes], target[test_indexes])

    return train_set, test_set


def _load_synthetic_data(name, folds=5, test_fraction=0.2, random_seed=None):
    """Load synthetic dataset.

    Parameters
    ----------
    name: string
        dataset name.
    folds: int, optional
        number of splits of dataset (default=5).
    test_fraction: float, optional
        fraction of dataset to split into test set (default=0.2).
    random_seed: int, optional
        random seed for pre-processing and splitting data (default=None).

    Returns
    -------
    dataset: Dataset
    test_set: Dataset

    """
    if name == "sinc":
        data, target = _sinc(random_seed=random_seed)

        data = np.expand_dims(data, 1)
        class_ = RegressionDataset
    elif name == "pareto":
        data, target = _pareto(random_seed=random_seed)
        data = np.expand_dims(data, 1)
        class_ = RegressionDataset
    elif name == "normal":
        data, target = _normal(random_seed=random_seed)
        data = np.expand_dims(data, 1)
        class_ = RegressionDataset

    else:
        raise ValueError

    data = torch.tensor(data).float()
    target = torch.tensor(target).float()

    rng = np.random.RandomState(random_seed)
    indexes = np.arange(data.shape[0])
    rng.shuffle(indexes)
    train_indexes, test_indexes = np.split(
        indexes, [int(data.shape[0] * (1 - test_fraction))]
    )

    train_set = class_(data=data[train_indexes], target=target[train_indexes])
    train_set = KFoldDataset(train_set, folds=folds)
    test_set = class_(data[test_indexes], target[test_indexes])

    return train_set, test_set


def _load_vision(
    name, train_shift=False, shift_fraction=0.1, upsample=False, downsample=False
):
    """Load torchvision dataset.

    Parameters
    ----------
    name: str
        name of dataset.

    Returns
    -------
    dataset: Dataset
    test_set: Dataset

    """
    if "mnist" in name:
        t = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_set = datasets.MNIST(
            root=DATA_DIR, download=True, train=True, transform=t
        )
        valid_set = datasets.MNIST(
            root=DATA_DIR, download=True, train=True, transform=t
        )
        test_set = datasets.MNIST(
            root=DATA_DIR, download=True, train=False, transform=t
        )

    elif "fashion-mnist" in name:
        if "augmented" in name:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )

        transform_valid = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        train_set = datasets.FashionMNIST(
            root=DATA_DIR, download=True, train=True, transform=transform_train
        )
        valid_set = datasets.FashionMNIST(
            root=DATA_DIR, download=True, train=True, transform=transform_valid
        )
        test_set = datasets.FashionMNIST(
            root=DATA_DIR, download=True, train=False, transform=transform_test
        )

    elif "cifar-10" in name:
        if "augmented" in name:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

        transform_valid = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_set = datasets.CIFAR10(
            root=DATA_DIR, download=True, train=True, transform=transform_train
        )

        valid_set = datasets.CIFAR10(
            root=DATA_DIR, download=True, train=True, transform=transform_valid
        )

        test_set = datasets.CIFAR10(
            root=DATA_DIR, download=True, train=False, transform=transform_test
        )

    else:
        raise ValueError

    if train_shift:
        train_set = _multi_class_reshift(train_set, shift_fraction, upsample)
        valid_set = _multi_class_reshift(valid_set, shift_fraction, upsample)

    classification = "regression" not in name
    return (
        TrainValidationDataset(
            VisionDatasetWrapper(train_set, classification),
            VisionDatasetWrapper(valid_set, classification),
        ),
        VisionDatasetWrapper(test_set, classification),
    )


def _reshift_indexes(
    indexes, target, majority, shift_fraction, upsample=False, downsample=False
):
    """Reshift indexes so that the majority class becomes the minority class."""
    minority_indexes = indexes[np.where(target[indexes] == 1 - majority)[0]]
    majority_indexes = indexes[np.where(target[indexes] == majority)[0]]
    majority_indexes = majority_indexes[
        : int(np.ceil(shift_fraction * len(minority_indexes)))
    ]
    # len (majority_indexes) = shift_fraction * len(minority_indexes)

    if upsample:
        # Upsample the majority index.
        majority_indexes = np.random.choice(
            majority_indexes, size=len(minority_indexes), replace=True
        )
    elif downsample:
        # Downsample the minority index.
        minority_indexes = np.random.choice(
            minority_indexes, size=len(majority_indexes), replace=False
        )

    indexes = np.concatenate((minority_indexes, majority_indexes))
    np.random.shuffle(indexes)
    return indexes


def _multi_class_reshift(dataset, shift_fraction, upsample=False, downsample=False):
    hist = defaultdict(list)
    for idx, (X, target) in enumerate(dataset):
        hist[target].append(idx)  # List[class] = [indexes]

    final = []
    num_classes = len(hist)
    k = np.log(shift_fraction) / np.log(num_classes)
    for class_idx, indexes in hist.items():
        p = (class_idx + 1) ** k
        class_samples = list(
            np.random.choice(indexes, size=int(len(indexes) * p), replace=False)
        )
        if upsample:
            class_samples = list(
                np.random.choice(class_samples, size=len(indexes), replace=True)
            )
        elif downsample:
            p = 1 ** k
            class_samples = list(
                np.random.choice(
                    class_samples, size=int(len(indexes) * p), replace=False
                )
            )
        final += class_samples
    np.random.shuffle(final)

    return Subset(dataset, final)


#  Synthetic Data Generators
def _sinc(n_points=1000, n_centers=10, noise_level=0.2, random_seed=None):
    """Generate data from the sinc function (defined as (sin(x)/x)).

    x ~ U[-10, 10]
    y = sinc(x)

    c ~ U[-10, 10]

    X = RBF(x, c_i)
    Y = y + N(0, noise_level)


    Parameters
    ----------
    n_points: int, optional
        Number of points in dataset.
    n_centers: int, optional
        Number of feature centers.
    noise_level: float, optional
        Noise to corrupt target.
    random_seed: int, optional
        Random seed.

    Returns
    -------
    data: nd-array
    target: nd-array.

    """
    rng = np.random.RandomState(random_seed)

    x = 20 * (rng.random_sample(n_points) - 0.5)
    noise = noise_level * rng.randn(n_points)
    y = np.sinc(x)

    data = []
    feature_names = []
    for i in range(n_centers):
        c = 20 * (rng.random_sample() - 0.5)
        data.append(np.exp(-((c - x) ** 2)))
        feature_names.append("c{}".format(i))

    data = np.array(data).T
    target = y + noise

    target = (target - target.mean()) / target.std()
    data = pre_process_data(data)

    return data, target


def _pareto(n_points=1000, beta=3.0, rho=0.9, noise_level=0.1, random_seed=None):
    r"""Generate a Linear Regression Model with a pareto distribution noise.

    X = N(0, (1-rho)I + rho) #equi-correlation matrix with correlation rho)
    y = X.\theta
    Y = y + noise_level * Pareto(beta)

    Parameters
    ----------
    n_points: int, optional
        Number of points.
    beta: float, optional
        Pareto shape distribution.
    rho: float, optional
        Co-variate co-relation coefficient.
    noise_level: float
        Noise scale.
    random_seed: int, optional
        Experiment
    Returns
    -------
    data: nd-array
    target: nd-array.

    References
    ----------
    Brownlees, C., Joly, E., & Lugosi, G. (2015). Empirical risk minimization
    for heavy-tailed losses. The Annals of Statistics, 43(6), 2507-2536.

    """
    rng = scipy.random.RandomState(random_seed)
    theta = np.array([0.25, -0.25, 0.50, 0.70, -0.75])
    n_dim = len(theta)
    mu = np.zeros(n_dim)
    sigma = (1 - rho) * np.eye(n_dim) + rho * np.ones((n_dim, n_dim))
    data = rng.multivariate_normal(mu, sigma, n_points)

    noise = rng.pareto(beta, n_points)
    noise = noise - 1 / (beta - 1)  # subtract the mean.

    target = np.dot(data, theta) + noise_level * noise

    return data, target


def _normal(n_points=1000, rho=0.9, noise_level=0.1, random_seed=None):
    r"""Generate a Linear Regression Model with a pareto distribution noise.

    X = N(0, (1-rho)I + rho) #equi-correlation matrix with correlation rho)
    y = X.\theta
    Y = y + noise_level * N(0, 1)

    Parameters
    ----------
    n_points: int, optional
        Number of points.
    noise_level: float
        Noise scale.
    random_seed: int, optional
        Experiment
    Returns
    -------
    data: nd-array
    target: nd-array.

    """
    rng = scipy.random.RandomState(random_seed)
    theta = np.array([0.25, -0.25, 0.50, 0.70, -0.75])
    n_dim = len(theta)
    mu = np.zeros(n_dim)
    sigma = (1 - rho) * np.eye(n_dim) + rho * np.ones((n_dim, n_dim))
    data = rng.multivariate_normal(mu, sigma, n_points)

    noise = rng.randn(n_points)

    target = np.dot(data, theta) + noise_level * noise

    return data, target


#  Utilities
def pre_process_data(data, feature_names=None, categories=None):
    """Pre process the data co-variates.

    Categorical features will be one-hot-encoded.
    Numerical features will be standarized.

    Parameters
    ----------
    data: ndarray
    feature_names: list
    categories: dictionary

    Returns
    -------
    transformed data: ndarray
    """
    n_points = data.shape[0]
    processed_data = np.array([], dtype=np.float).reshape(n_points, 0)

    feature_names = (
        feature_names if feature_names else ["" for _ in range(data.shape[1])]
    )
    categories = categories if categories else {}

    for idx, feature_name in enumerate(feature_names):
        if feature_name in categories:  # OneHotEncode
            enc = OneHotEncoder(categories="auto", sparse=False)
            data[np.argwhere(np.isnan(data[:, idx])), idx] = -1
            features = enc.fit_transform(data[:, idx].reshape(n_points, 1))
        else:  # Normalize
            features = data[:, idx] - np.mean(data[:, idx])
            features = features / np.std(data[:, idx])

        processed_data = np.concatenate(
            (processed_data, features.reshape(n_points, -1)), axis=1
        )
    return processed_data


def ohe_target(target):
    """Return a One-Hot Encoding of the target.

    Parameters
    ----------
    target: ndarray

    Returns
    -------
    transformed target: ndarray
    """
    return label_binarize(target, classes=np.unique(target)).ravel()
