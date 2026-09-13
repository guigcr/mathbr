import pytest
import random

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


def test_householder_qr_dense_multifeature_fit():
    rng = random.Random(17)
    features = 8
    slopes = [(-1) ** j * (j + 1) / 3 for j in range(features)]
    X = [[rng.uniform(-2, 2) for _ in range(features)] for _ in range(80)]
    y = [1.25 + sum(a * b for a, b in zip(row, slopes))
         + 0.001 * rng.uniform(-1, 1) for row in X]
    model = mathbr.OLS(features)
    model.fit(X, y)
    assert model.coefficients() == pytest.approx([1.25, *slopes], abs=3e-4)
    assert all(error > 0 for error in model.standard_errors())
    assert model.r_squared() > 0.999999


def test_weighted_qr_against_weighted_line_formula():
    X = [[0], [1], [2], [3]]
    y = [1, 3, 5, 8]
    weights = [1, 2, 3, 0.25]
    total = sum(weights)
    mean_x = sum(w * row[0] for w, row in zip(weights, X)) / total
    mean_y = sum(w * value for w, value in zip(weights, y)) / total
    slope = sum(w * (row[0] - mean_x) * (value - mean_y)
                for w, row, value in zip(weights, X, y)) / sum(
                    w * (row[0] - mean_x) ** 2 for w, row in zip(weights, X))
    model = mathbr.WLS(1)
    model.fit(X, y, weights)
    assert model.coefficients() == pytest.approx([mean_y - slope * mean_x, slope])


@pytest.mark.parametrize("model_type", [mathbr.OLS, mathbr.WLS])
def test_qr_rejects_rank_deficient_design(model_type):
    X = [[float(i), float(2 * i)] for i in range(5)]
    y = [1.0 + 3.0 * i for i in range(5)]
    model = model_type(2)
    with pytest.raises(ValueError, match="linearly dependent"):
        if model_type is mathbr.WLS:
            model.fit(X, y, [1.0, 2.0, 1.0, 3.0, 1.0])
        else:
            model.fit(X, y)
    assert not model.trained()


@pytest.mark.parametrize("model_type", [mathbr.OLS, mathbr.WLS])
def test_qr_handles_small_scale_feature(model_type):
    X = [[i * 1e-8] for i in range(6)]
    y = [2.0 + 3.0 * row[0] for row in X]
    model = model_type(1)
    if model_type is mathbr.WLS:
        model.fit(X, y, [1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    else:
        model.fit(X, y)
    assert model.coefficients() == pytest.approx([2.0, 3.0], abs=1e-6)


@pytest.mark.parametrize("model_type", [mathbr.OLS, mathbr.WLS])
def test_classical_inference_and_adjusted_fit(model_type):
    X = [[0.0], [1.0], [2.0], [3.0]]
    y = [1.0, 3.0, 5.0, 8.0]
    model = model_type(1)
    if model_type is mathbr.WLS:
        model.fit(X, y, [1.0] * len(y))
    else:
        model.fit(X, y)
    coefficients = model.coefficients()
    errors = model.standard_errors()
    assert model.t_statistics() == pytest.approx(
        [coef / error for coef, error in zip(coefficients, errors)])
    assert model.p_values() == pytest.approx(
        [1.0 - abs(t) / (t * t + 2.0) ** 0.5
         for t in model.t_statistics()])  # Closed-form two-sided t(2) tail.
    assert model.adjusted_r_squared() == pytest.approx(1.0 - 0.3 / 26.75 * 3 / 2)
    assert model.f_statistic() == pytest.approx(176.3333333333)
    intervals = model.confidence_intervals(0.95)
    assert len(intervals) == len(coefficients)
    assert all(low < coef < high for (low, high), coef in zip(intervals, coefficients))
    assert intervals[1][1] - coefficients[1] == pytest.approx(
        4.302652729911275 * errors[1])
    assert model.confidence_intervals(0.99)[1][1] > intervals[1][1]


def test_inference_rejects_undefined_cases():
    model = mathbr.OLS(1)
    with pytest.raises(RuntimeError):
        model.p_values()
    model.fit([[0], [1], [2], [3]], [1, 3, 5, 8])
    with pytest.raises(ValueError):
        model.confidence_intervals(1.0)
    perfect = mathbr.OLS(1)
    perfect.fit([[0], [1], [2], [3]], [1, 3, 5, 7])
    with pytest.raises(ValueError):
        perfect.t_statistics()
    with pytest.raises(ValueError):
        perfect.f_statistic()
