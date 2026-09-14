import math

import pytest

import mathbr


d = mathbr.distributions


def test_poisson_reference_values_and_quantile_contract():
    assert d.poisson_pmf(-1, 4) == 0
    assert d.poisson_cdf(-1, 4) == 0
    assert d.poisson_pmf(2, 4) == pytest.approx(8 * math.exp(-4))
    assert d.poisson_cdf(2, 4) == pytest.approx(13 * math.exp(-4))
    assert d.poisson_pmf(0, 0) == 1
    assert d.poisson_cdf(0, 0) == 1
    assert d.poisson_ppf(1, 0) == 0
    assert d.poisson_ppf(0, 4) == 0
    assert d.poisson_ppf(1, 4) == math.inf
    for rate in (0.1, 1, 4, 30, 1000):
        for q in (0.01, 0.1, 0.5, 0.9, 0.99):
            k = int(d.poisson_ppf(q, rate))
            assert d.poisson_cdf(k, rate) >= q
            if k:
                assert d.poisson_cdf(k - 1, rate) < q


def test_poisson_cdf_matches_direct_sum_for_moderate_rates():
    for rate in (0.2, 2, 15, 40):
        cumulative = 0.0
        for k in range(100):
            cumulative += d.poisson_pmf(k, rate)
            assert d.poisson_cdf(k, rate) == pytest.approx(cumulative, abs=2e-13)


@pytest.mark.parametrize("call", [
    lambda: d.poisson_pmf(1, -1),
    lambda: d.poisson_cdf(1, math.nan),
    lambda: d.poisson_ppf(math.inf, 2),
    lambda: d.poisson_ppf(0.5, math.inf),
])
def test_poisson_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
