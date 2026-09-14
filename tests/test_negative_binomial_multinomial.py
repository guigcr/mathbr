import math

import pytest

import mathbr


d = mathbr.distributions


def test_negative_binomial_known_values_and_quantile_contract():
    # Number of failures before the second success with p=1/2.
    assert d.negative_binomial_pmf(-1, 2, 0.5) == 0
    assert d.negative_binomial_pmf(0, 2, 0.5) == pytest.approx(0.25)
    assert d.negative_binomial_pmf(2, 2, 0.5) == pytest.approx(0.1875)
    assert d.negative_binomial_cdf(2, 2, 0.5) == pytest.approx(0.6875)
    assert d.negative_binomial_ppf(0.6875, 2, 0.5) == 2
    assert d.negative_binomial_ppf(0, 2, 0.5) == 0
    assert d.negative_binomial_ppf(1, 2, 0.5) == math.inf
    assert d.negative_binomial_ppf(1, 2, 1) == 0
    for successes, p in ((1, 0.2), (2, 0.5), (10, 0.8), (30, 0.3)):
        cumulative = 0.0
        for k in range(80):
            cumulative += d.negative_binomial_pmf(k, successes, p)
            assert d.negative_binomial_cdf(k, successes, p) == pytest.approx(
                cumulative, abs=2e-12)
        for q in (0.01, 0.2, 0.5, 0.9, 0.99):
            k = int(d.negative_binomial_ppf(q, successes, p))
            assert d.negative_binomial_cdf(k, successes, p) >= q
            if k:
                assert d.negative_binomial_cdf(k - 1, successes, p) < q


def test_multinomial_known_values_and_zero_probability():
    assert d.multinomial_pmf([1, 1, 1], [0.2, 0.3, 0.5]) == pytest.approx(0.18)
    assert d.multinomial_pmf([0, 2], [0, 1]) == 1
    assert d.multinomial_pmf([1, 1], [0, 1]) == 0
    assert d.multinomial_pmf([0, 0, 0], [0.2, 0.3, 0.5]) == 1
    assert sum(d.multinomial_pmf([k, 3 - k], [0.4, 0.6]) for k in range(4)) == pytest.approx(1)


@pytest.mark.parametrize("call", [
    lambda: d.negative_binomial_pmf(1, 0, 0.5),
    lambda: d.negative_binomial_cdf(1, 2, 0),
    lambda: d.negative_binomial_ppf(math.nan, 2, 0.5),
    lambda: d.multinomial_pmf([], []),
    lambda: d.multinomial_pmf([1], [0.5]),
    lambda: d.multinomial_pmf([1, 1], [1]),
    lambda: d.multinomial_pmf([-1, 1], [0.5, 0.5]),
    lambda: d.multinomial_pmf([1, 1], [math.nan, 1]),
])
def test_new_discrete_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
