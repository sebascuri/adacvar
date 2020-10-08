import numpy as np
import pytest

from adacvar.util.approximate_kdpp import ApproximateKDPP


@pytest.fixture(params=[100, 1000])
def num_ground(request):
    return request.param


@pytest.fixture(params=[1, 10, 100])
def k_sample(request):
    return request.param


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_init(num_ground, k_sample):
    kdpp = ApproximateKDPP(num_ground, k_sample)

    assert kdpp.num_ground == num_ground
    assert kdpp.k_sample == k_sample

    np.testing.assert_equal(kdpp.values, np.ones(num_ground))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_value_setter(num_ground, k_sample):
    kdpp = ApproximateKDPP(num_ground, k_sample)

    np.testing.assert_equal(kdpp.values, np.ones(num_ground))

    values = np.random.rand(num_ground)
    kdpp.values = values
    np.testing.assert_equal(kdpp.values, values)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_marginals(num_ground, k_sample):
    kdpp = ApproximateKDPP(num_ground, k_sample)

    np.testing.assert_almost_equal(
        kdpp.marginals, k_sample / num_ground * np.ones(num_ground)
    )

    values = np.zeros(num_ground)
    values[:k_sample] = 1
    kdpp.values = values
    kdpp.normalize()
    np.testing.assert_almost_equal(kdpp.marginals, values)
