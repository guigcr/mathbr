import math

import pytest

import mathbr


def test_beta_binomial_posterior_and_interval():
    posterior = mathbr.bayesian.beta_binomial_update(1, 1, 7, 10)
    assert (posterior.alpha, posterior.beta) == (8, 4)
    assert posterior.mean() == pytest.approx(2 / 3)
    low, high = posterior.credible_interval(0.9)
    assert mathbr.distributions.beta_cdf(low, 8, 4) == pytest.approx(0.05)
    assert mathbr.distributions.beta_cdf(high, 8, 4) == pytest.approx(0.95)
    assert low < posterior.mean() < high
    with pytest.raises(ValueError):
        posterior.credible_interval(1)


def test_normal_normal_posterior_and_interval():
    posterior = mathbr.bayesian.normal_normal_update(0, 1, 1, [1, 3])
    assert posterior.mean == pytest.approx(4 / 3)
    assert posterior.standard_deviation == pytest.approx(math.sqrt(1 / 3))
    low, high = posterior.credible_interval(0.95)
    assert (low + high) / 2 == pytest.approx(posterior.mean)
    assert high > low


@pytest.mark.parametrize("call", [
    lambda b: b.beta_binomial_update(0, 1, 1, 2),
    lambda b: b.beta_binomial_update(1, 1, 3, 2),
    lambda b: b.normal_normal_update(0, 1, 1, []),
    lambda b: b.normal_normal_update(0, 0, 1, [1]),
    lambda b: b.normal_normal_update(0, 1, 1, [math.nan]),
])
def test_invalid_updates(call):
    with pytest.raises(ValueError):
        call(mathbr.bayesian)
