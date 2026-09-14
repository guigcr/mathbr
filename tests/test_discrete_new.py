import math

import pytest

import mathbr


d = mathbr.distributions


def test_geometric_mass_cdf_and_quantile():
    assert d.geometric_pmf(0, 0.25) == 0
    assert d.geometric_pmf(1, 0.25) == pytest.approx(0.25)
    assert d.geometric_pmf(3, 0.25) == pytest.approx(0.25 * 0.75**2)
    assert d.geometric_cdf(0, 0.25) == 0
    assert d.geometric_cdf(3, 0.25) == pytest.approx(1 - 0.75**3)
    assert d.geometric_ppf(0, 0.25) == 1
    assert d.geometric_ppf(0.25, 0.25) == 1
    assert d.geometric_ppf(0.2500001, 0.25) == 2
    assert d.geometric_ppf(1, 0.25) == math.inf
    assert d.geometric_ppf(1, 1) == 1
    for p in (0.01, 0.2, 0.8, 1):
        for k in (1, 2, 5, 20):
            q = d.geometric_cdf(k, p)
            assert d.geometric_ppf(q, p) <= k
            if q < 1:
                assert d.geometric_cdf(int(d.geometric_ppf(q, p)), p) >= q


def test_discrete_uniform_extreme_bounds_and_quantiles():
    assert d.discrete_uniform_pmf(2, 1, 3) == pytest.approx(1 / 3)
    assert d.discrete_uniform_pmf(0, 1, 3) == 0
    assert d.discrete_uniform_cdf(2, 1, 3) == pytest.approx(2 / 3)
    assert d.discrete_uniform_ppf(0, 1, 3) == 1
    assert d.discrete_uniform_ppf(1 / 3, 1, 3) == 1
    assert d.discrete_uniform_ppf(1, 1, 3) == 3
    assert d.discrete_uniform_pmf(0, -(2**31), 2**31 - 1) == pytest.approx(1 / 2**32)
    assert d.discrete_uniform_ppf(0.5, -(2**31), 2**31 - 1) == -1
    assert d.discrete_uniform_cdf(2**31 - 1, -(2**31), 2**31 - 1) == 1


@pytest.mark.parametrize("call", [
    lambda: d.geometric_pmf(1, 0),
    lambda: d.geometric_cdf(1, 1.1),
    lambda: d.geometric_ppf(math.nan, 0.5),
    lambda: d.discrete_uniform_pmf(0, 2, 1),
    lambda: d.discrete_uniform_cdf(0, 2, 1),
    lambda: d.discrete_uniform_ppf(-0.1, 0, 1),
])
def test_discrete_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
