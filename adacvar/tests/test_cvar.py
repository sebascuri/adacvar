import pytest
import torch
import torch.testing

from adacvar.util.cvar import CVaR, SoftCVaR


@pytest.fixture(params=[-1, 0, 0.1, 1, 2])
def alpha(request):
    return request.param


class TestCVaR(object):
    def test_alphas(self, alpha):
        if 0 < alpha <= 1:
            module = CVaR(alpha=alpha)
            if alpha == 1:
                assert not module.var.requires_grad
            else:
                assert module.var.requires_grad
        else:
            with pytest.raises(ValueError):
                CVaR(alpha=alpha)

    def test_forward(self, alpha):
        if 0 < alpha <= 1:
            module = CVaR(alpha=alpha)
            losses = torch.rand(32)
            cvar = module(losses)
            if alpha == 1:
                torch.testing.assert_allclose(cvar, losses)
            else:
                var = module.var.item()
                torch.testing.assert_allclose(
                    cvar, var + torch.relu(losses - var) / alpha
                )

    def test_optimization(self):
        module = CVaR(alpha=0.1)
        module.zero_grad()
        module.step()


class TestSoftCVaR(object):
    @pytest.fixture(scope="class", params=[-1, 0, 0.1, 1.0, 1.0])
    def temperature(self, request):
        return request.param

    def test_alphas(self, alpha, temperature):
        if 0 < alpha <= 1 and temperature > 0:
            module = SoftCVaR(alpha=alpha, temperature=temperature)
            if alpha == 1:
                assert not module.var.requires_grad
            else:
                assert module.var.requires_grad
        else:
            with pytest.raises(ValueError):
                SoftCVaR(alpha=alpha, temperature=temperature)

    def test_forward(self, alpha, temperature):
        if 0 < alpha <= 1 and temperature > 0:
            module = SoftCVaR(alpha=alpha, temperature=temperature)
            losses = torch.rand(32)
            cvar = module(losses)
            if alpha == 1:
                var = module.var.item()
                soft = torch.log(1 + torch.exp((losses - var) / temperature))
                torch.testing.assert_allclose(cvar, var + temperature / alpha * soft)
