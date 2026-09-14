import math

import pytest

import mathbr


d = mathbr.distributions


def test_closed_form_distribution_fits():
    mean, sigma = d.normal_fit([1, 2, 3, 4])
    assert mean == pytest.approx(2.5)
    assert sigma == pytest.approx(math.sqrt(1.25))
    assert d.normal_fit([1e100, 1e100 + 1e85])[0] == pytest.approx(1e100)
    assert d.exponential_fit([0, 1, 2, 3]) == pytest.approx(2 / 3)
    assert d.poisson_fit([0, 1, 2, 3]) == pytest.approx(1.5)
    assert d.poisson_fit([0, 0, 0]) == 0


@pytest.mark.parametrize("call", [
    lambda: d.normal_fit([]),
    lambda: d.normal_fit([1, 1, 1]),
    lambda: d.normal_fit([1, math.nan]),
    lambda: d.exponential_fit([]),
    lambda: d.exponential_fit([0, 0]),
    lambda: d.exponential_fit([1, -1]),
    lambda: d.poisson_fit([]),
    lambda: d.poisson_fit([0, -1]),
])
def test_closed_form_fits_reject_invalid_data(call):
    with pytest.raises(ValueError):
        call()
