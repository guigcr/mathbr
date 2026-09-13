import math

import pytest

import mathbr


def test_activations_and_losses():
    assert mathbr.activations.sigmoid(-1000) == 0.0
    assert mathbr.activations.softmax([1000, 1001]) == pytest.approx(
        [0.26894142137, 0.73105857863]
    )
    assert mathbr.losses.mse([1, 2], [2, 4]) == pytest.approx(2.5)
    assert mathbr.losses.mse_derivative([1, 2], [2, 4]) == pytest.approx([1, 2])
    assert mathbr.losses.logloss([0, 1], [0.1, 0.9]) == pytest.approx(-math.log(0.9))


@pytest.mark.parametrize(
    "call",
    [
        lambda: mathbr.activations.softmax([]),
        lambda: mathbr.losses.mse([], []),
        lambda: mathbr.losses.mae([1], [1, 2]),
        lambda: mathbr.losses.logloss([2], [0.5]),
        lambda: mathbr.losses.logloss([1], [float("nan")]),
        lambda: mathbr.LinearRegression(0),
        lambda: mathbr.LogisticRegression(-1),
        lambda: mathbr.LinearRegression(2).fit([[1]], [1]),
        lambda: mathbr.LinearRegression(1).fit([], []),
        lambda: mathbr.LinearRegression(1).fit([[1]], [1], lr=0),
        lambda: mathbr.LogisticRegression(1).fit([[1]], [2]),
        lambda: mathbr.LogisticRegression(1).predict([1], threshold=2),
    ],
)
def test_invalid_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()


def test_linear_regression_gradient_and_convergence():
    model = mathbr.LinearRegression(1)
    model.fit([[0], [1]], [0, 1], lr=0.1, epochs=1)
    assert model.get_weights() == pytest.approx([0.1])
    assert model.get_bias() == pytest.approx(0.1)

    model.fit([[0], [1], [2]], [1, 3, 5], lr=0.1, epochs=1000)
    assert model.predict_batch([[3], [4]]) == pytest.approx([7, 9], abs=0.01)
    assert model.trained()


def test_logistic_regression_convergence():
    model = mathbr.LogisticRegression(1)
    model.fit([[-2], [-1], [1], [2]], [0, 0, 1, 1], lr=0.2, epochs=500)
    assert model.predict([-1]) == 0
    assert model.predict([1]) == 1
    assert model.predict_proba([2]) > model.predict_proba([1])
