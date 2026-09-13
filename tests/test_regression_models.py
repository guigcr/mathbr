import pytest

import mathbr


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (lambda: mathbr.Ridge(1, alpha=1), 0.8),
        (lambda: mathbr.Lasso(1, alpha=0.5), 1.25),
        (lambda: mathbr.ElasticNet(1, alpha=1, l1_ratio=0.5), 5 / 7),
    ],
)
def test_penalized_regression_known_solution(model, expected):
    fitted = model()
    fitted.fit([[-1], [0], [1]], [-2, 0, 2])
    assert fitted.coefficients() == pytest.approx([expected])
    assert fitted.intercept() == pytest.approx(0)
    assert fitted.predict([2]) == pytest.approx(2 * expected)
    assert fitted.converged()
    assert fitted.iterations() > 0


def test_lasso_sets_small_coefficient_to_zero():
    model = mathbr.Lasso(1, alpha=2)
    model.fit([[-1], [0], [1]], [-2, 0, 2])
    assert model.coefficients() == [0.0]


def test_regularized_validation():
    with pytest.raises(ValueError):
        mathbr.Ridge(1, alpha=-1)
    with pytest.raises(ValueError):
        mathbr.ElasticNet(1, l1_ratio=2)
    with pytest.raises(RuntimeError):
        mathbr.Lasso(1).coefficients()
    with pytest.raises(ValueError):
        mathbr.Ridge(1).fit([[1, 2]], [1])


def test_wls_known_solution_and_validation():
    model = mathbr.WLS(1)
    model.fit([[0], [1], [2]], [0, 1, 4], [1, 1, 0.1])
    assert model.coefficients() == pytest.approx([-2 / 15, 1.4])
    assert model.predict([3]) == pytest.approx(61 / 15)
    assert model.degrees_of_freedom() == 1
    with pytest.raises(ValueError):
        model.fit([[0], [1], [2]], [0, 1, 4], [1, 0, 1])
