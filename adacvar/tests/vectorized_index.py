"""Testing script for vectorization in update method."""

import numpy as np

if __name__ == "__main__":
    loss = np.zeros(5)
    array = np.array([1, 1.5, 1, 0.5, 0.2])

    idx = np.array([0, 2, 0, 1, 4])
    np.add.at(loss, idx, array)
    assert np.all(loss == np.array([2, 0.5, 1.5, 0, 0.2]))
