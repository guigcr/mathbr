import pytest

import mathbr


def test_iv2sls_exactly_identified_known_solution():
    model = mathbr.IV2SLS(n_exog=0, n_endog=1, n_instruments=1)
    z = [[0], [1], [2], [3], [4]]
    endogenous = [[0], [2], [1], [4], [3]]
    y = [1, 5, 3, 9, 7]
    model.fit([[], [], [], [], []], endogenous, z, y)
    assert model.coefficients() == pytest.approx([1, 2])
    assert model.predict([], [5]) == pytest.approx(11)
    assert 0 < model.first_stage_r_squared()[0] < 1


def test_iv2sls_rejects_underidentification_and_bad_rows():
    with pytest.raises(ValueError):
        mathbr.IV2SLS(0, 2, 1)
    model = mathbr.IV2SLS(0, 1, 1)
    with pytest.raises(RuntimeError):
        model.coefficients()
    with pytest.raises(ValueError):
        model.fit([[], [], [], [], []], [[0], [1], [2], [3], [4]],
                  [[1], [1], [1], [1], [1]], [0, 1, 2, 3, 4])


def test_entity_fixed_effects_known_solution():
    model = mathbr.FixedEffects(1)
    X = [[0], [1], [2], [0], [1], [2]]
    y = [1, 3, 5, 10, 12, 14]
    ids = [10, 10, 10, 20, 20, 20]
    model.fit(X, y, ids)
    assert model.coefficients() == pytest.approx([2])
    assert model.entity_intercept(10) == pytest.approx(1)
    assert model.entity_intercept(20) == pytest.approx(10)
    assert model.predict([3], 20) == pytest.approx(16)
    assert model.within_r_squared() == pytest.approx(1)
    with pytest.raises(ValueError):
        model.predict([1], 99)


def test_fixed_effects_rejects_no_within_variation():
    model = mathbr.FixedEffects(1)
    with pytest.raises(ValueError):
        model.fit([[1], [1], [2], [2]], [1, 2, 3, 4], [1, 1, 2, 2])
