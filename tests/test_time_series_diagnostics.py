import math

import pytest

import mathbr


def test_acf_pacf_and_ljung_box_reference():
    d = mathbr.time_series_diagnostics
    assert d.acf([1, 2, 3], 2) == pytest.approx([1, 0, -0.5])
    assert d.pacf([1, 2, 3], 2) == pytest.approx([1, 0, -0.5])
    lb = d.ljung_box([1, 2, 3], 2)
    assert lb.statistic == pytest.approx(3.75)
    assert lb.p_value == pytest.approx(math.exp(-3.75 / 2))
    assert lb.lags == 2


def test_lag_zero_and_invalid_series():
    d = mathbr.time_series_diagnostics
    assert d.acf([1, 2], 0) == [1]
    assert d.pacf([1, 2], 0) == [1]
    with pytest.raises(ValueError):
        d.acf([1, 1, 1], 1)
    with pytest.raises(ValueError):
        d.pacf([1, math.nan, 2], 1)
    with pytest.raises(ValueError):
        d.acf([1, 2], 2)
    with pytest.raises(ValueError):
        d.ljung_box([1, 2], 0)
