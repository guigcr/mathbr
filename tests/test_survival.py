import math

import pytest

import mathbr


def test_kaplan_meier_tied_event_and_censor():
    result = mathbr.survival.kaplan_meier([3, 1, 2, 2], [1, 1, 0, 1])
    assert result.times == [1, 2, 3]
    assert result.at_risk == [4, 3, 1]
    assert result.events == [1, 1, 1]
    assert result.survival == pytest.approx([0.75, 0.5, 0])
    censored = mathbr.survival.kaplan_meier([1, 2], [0, 0])
    assert censored.survival == [1, 1]


def test_log_rank_reference():
    result = mathbr.survival.log_rank_test([1, 1, 2, 2], [1, 1, 1, 1], [0, 0, 1, 1])
    assert result.statistic == pytest.approx(3)
    assert result.p_value == pytest.approx(math.erfc(math.sqrt(1.5)))


@pytest.mark.parametrize("call", [
    lambda s: s.kaplan_meier([], []),
    lambda s: s.kaplan_meier([1, 2], [1]),
    lambda s: s.kaplan_meier([-1], [1]),
    lambda s: s.kaplan_meier([1], [2]),
    lambda s: s.log_rank_test([1, 2], [1, 1], [0, 0]),
    lambda s: s.log_rank_test([1, 1], [0, 0], [0, 1]),
])
def test_invalid_survival_inputs(call):
    with pytest.raises(ValueError):
        call(mathbr.survival)
