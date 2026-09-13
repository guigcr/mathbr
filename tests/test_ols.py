import pytest

import mathbr


def test_ols_estimates_and_diagnostics():
    model = mathbr.OLS(1)
    model.fit([[0], [1], [2], [3]], [1, 3, 5, 8])
    assert model.trained()
    assert model.coefficients() == pytest.approx([0.8, 2.3])
    assert model.standard_errors() == pytest.approx([0.3240370349, 0.1732050808])
    assert model.degrees_of_freedom() == 2
    assert model.residual_variance() == pytest.approx(0.15)
    assert model.r_squared() == pytest.approx(1 - 0.3 / 26.75)
    assert model.predict([4]) == pytest.approx(10)


def test_ols_requires_fit_and_full_rank():
    model = mathbr.OLS(1)
    with pytest.raises(RuntimeError):
        model.coefficients()
    with pytest.raises(ValueError):
        model.fit([[0], [1]], [1, 2])  # No residual degrees of freedom.
    with pytest.raises(ValueError):
        model.fit([[1], [1], [1]], [1, 2, 3])
    with pytest.raises(ValueError):
        model.fit([[0], [1], [2]], [1, float("nan"), 3])


def test_ols_multiple_features():
    X = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]]
    y = [1, 3, 4, 6, 8]
    model = mathbr.OLS(2)
    model.fit(X, y)
    assert model.coefficients() == pytest.approx([1, 2, 3])
    assert model.predict_batch([[2, 2], [3, 1]]) == pytest.approx([11, 10])
