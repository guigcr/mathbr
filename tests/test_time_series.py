import math

import pytest

import mathbr


def test_ar_known_recurrence_and_forecast():
    model = mathbr.AR(lags=1)
    model.fit([1, 2, 4, 8, 16, 32])
    assert model.coefficients() == pytest.approx([0, 2])
    assert model.forecast(2) == pytest.approx([64, 128])


def test_var_two_series_known_system():
    values = [[1.0, 2.0]]
    for _ in range(20):
        x, y = values[-1]
        values.append([1 + 0.5 * x + 0.2 * y, -1 + 0.3 * x + 0.4 * y])
    model = mathbr.VAR(n_series=2, lags=1)
    model.fit(values)
    assert model.coefficients()[0] == pytest.approx([1, 0.5, 0.2], abs=1e-7)
    assert model.coefficients()[1] == pytest.approx([-1, 0.3, 0.4], abs=1e-7)
    x, y = values[-1]
    assert model.forecast(1)[0] == pytest.approx(
        [1 + 0.5 * x + 0.2 * y, -1 + 0.3 * x + 0.4 * y]
    )
    assert len(model.residual_covariance()) == 2


def test_time_series_input_checks():
    with pytest.raises(ValueError):
        mathbr.VAR(2, 0)
    with pytest.raises(ValueError):
        mathbr.AR(1).fit([1, 2, 3])
    with pytest.raises(ValueError):
        mathbr.VAR(2, 1).fit([[1, 2]] * 10)  # Singular lag design.
    with pytest.raises(RuntimeError):
        mathbr.AR(1).forecast(1)


@pytest.mark.parametrize("model_class", [mathbr.ARCH, mathbr.GARCH])
def test_volatility_models_positive_recursion(model_class):
    observations = [-2.0, -1.0, 0.5, 1.0, 2.0] * 20
    model = model_class(max_iter=300, tol=1e-5)
    model.fit(observations)
    assert model.trained()
    assert model.omega() > 0
    assert model.alpha() >= 0
    assert model.beta() >= 0
    assert model.alpha() + model.beta() < 1
    if model_class is mathbr.ARCH:
        assert model.beta() == 0
    path = model.conditional_variance()
    assert len(path) == len(observations)
    assert all(math.isfinite(h) and h > 0 for h in path)
    future = model.forecast_variance(2)
    expected = model.omega() + model.alpha() * (observations[-1] - model.mean()) ** 2
    expected += model.beta() * path[-1]
    assert future[0] == pytest.approx(expected)
    assert future[1] == pytest.approx(model.omega() + (model.alpha() + model.beta()) * future[0])
    assert math.isfinite(model.log_likelihood())


def test_garch_validation():
    with pytest.raises(ValueError):
        mathbr.GARCH(max_iter=0)
    with pytest.raises(ValueError):
        mathbr.GARCH().fit([1.0] * 30)
    with pytest.raises(ValueError):
        mathbr.GARCH().fit([1.0, 2.0])
