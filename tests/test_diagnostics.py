import math

import pytest

import mathbr


def test_information_criteria_and_residual_scale():
    d = mathbr.diagnostics
    assert d.aic(-10, 2) == 24
    assert d.bic(-10, 2, 100) == pytest.approx(20 + 2 * math.log(100))
    assert d.hqic(-10, 2, 100) == pytest.approx(20 + 4 * math.log(math.log(100)))
    assert d.residual_standard_error([1, -1, 2, -2], 2) == pytest.approx(math.sqrt(5))


def test_durbin_watson_and_jarque_bera():
    d = mathbr.diagnostics
    assert d.durbin_watson([1, -1, 1, -1]) == 3
    residuals = [-2, -1, 0, 1, 2]
    statistic = d.jarque_bera_statistic(residuals)
    assert statistic >= 0
    assert d.jarque_bera_p_value(residuals) == pytest.approx(math.exp(-statistic / 2))


@pytest.mark.parametrize("call", [
    lambda d: d.aic(math.nan, 2),
    lambda d: d.bic(-10, 2, 0),
    lambda d: d.hqic(-10, 2, 2),
    lambda d: d.residual_standard_error([1, 2], 2),
    lambda d: d.durbin_watson([0, 0]),
    lambda d: d.jarque_bera_statistic([1, 1, 1]),
])
def test_invalid_diagnostics(call):
    with pytest.raises(ValueError):
        call(mathbr.diagnostics)
