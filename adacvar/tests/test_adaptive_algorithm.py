import numpy as np
import pytest

from adacvar.util.adaptive_algorithm import Exp3, Exp3Sampler


class TestExp3(object):
    def test_implementation(self):
        num_actions = 100
        size = 10
        eta = 0.1
        gamma = 0.01
        algorithm = Exp3(num_actions, size, eta, gamma)
        losses = np.linspace(0, 1, num_actions)

        for t in range(100):
            probs = algorithm.probabilities
            idx = np.random.choice(
                num_actions, size=size, p=probs / np.sum(probs), replace=False
            )
            idx = np.random.choice(idx, size=3, replace=True)

            algorithm.update(losses[idx], idx)
            algorithm.normalize()

    def test_update(self):
        num_actions = 100
        size = 10
        eta = 0.1
        algorithm = Exp3(num_actions, size, eta)
        algorithm.update(np.array([0.25, 0.25, 0.5]), np.array([0, 0, 2]))
        algorithm.normalize()
        np.testing.assert_almost_equal(
            algorithm.probabilities[0], algorithm.probabilities[2]
        )

        losses = np.ones(num_actions) * 0.5
        losses[0] = 0
        losses[2] = 0
        algorithm.update(
            losses, np.arange(0, num_actions), size * np.ones(num_actions) / num_actions
        )
        algorithm.normalize()
        np.testing.assert_almost_equal(
            algorithm.probabilities, size * np.ones(num_actions) / num_actions
        )

        algorithm.update(np.array([0.25, 0.25, 0.5]), np.array([0, 0, 2]))
        algorithm.normalize()
        np.testing.assert_almost_equal(
            algorithm.probabilities[0], algorithm.probabilities[2]
        )

        losses = np.ones(num_actions) * 0.5
        losses[0] = 0
        losses[2] = 0
        algorithm.update(losses, np.arange(0, num_actions))
        algorithm.normalize()
        assert not (
            algorithm.probabilities == size * np.ones(num_actions) / num_actions
        ).all()


class TestExp3Sampler(object):
    @pytest.fixture(scope="class", params=[True, False])
    def iid_batch(self, request):
        return request.param

    def test_implementation(self, iid_batch):
        num_actions = 100
        size = 10
        batch_size = 3
        eta = 0.1
        algorithm = Exp3Sampler(batch_size, num_actions, size, eta, iid_batch=iid_batch)
        losses = np.linspace(0, 1, num_actions)

        for idx in algorithm:
            assert len(idx) == batch_size
            algorithm.update(losses[idx], idx)
            algorithm.normalize()
