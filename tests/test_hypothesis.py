import math

import pytest

import mathbr


h = mathbr.hypothesis


def test_one_sample_t_test_known_result():
    result = h.one_sample_t_test([1.0, 2.0, 3.0])
    statistic = math.sqrt(12.0)
    assert result.statistic == pytest.approx(statistic)
    assert result.degrees_of_freedom == 2.0
    assert result.p_value == pytest.approx(1.0 - statistic / math.sqrt(statistic ** 2 + 2))


def test_paired_t_test_uses_within_pair_differences():
    result = h.paired_t_test([3.0, 4.0, 5.0], [1.0, 2.0, 4.0])
    assert result.statistic == pytest.approx(5.0)
    assert result.degrees_of_freedom == 2.0
    assert result.p_value == pytest.approx(1.0 - 5.0 / math.sqrt(27.0))


def test_welch_t_test_known_df():
    result = h.welch_t_test([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert result.statistic == pytest.approx(-2.0 / math.sqrt(5.0 / 3.0))
    assert result.degrees_of_freedom == pytest.approx(50.0 / 17.0)
    assert 0.0 < result.p_value < 1.0


@pytest.mark.parametrize("call", [
    lambda: h.one_sample_t_test([1.0]),
    lambda: h.one_sample_t_test([1.0, 1.0]),
    lambda: h.one_sample_t_test([1.0, 2.0], math.nan),
    lambda: h.paired_t_test([1.0, 2.0], [1.0]),
    lambda: h.paired_t_test([1.0, 2.0], [0.0, 1.0]),
    lambda: h.welch_t_test([1.0], [1.0, 2.0]),
    lambda: h.welch_t_test([1.0, 1.0], [2.0, 2.0]),
])
def test_t_tests_reject_invalid_data(call):
    with pytest.raises(ValueError):
        call()


def test_multiple_comparison_adjustments_preserve_input_order():
    p = [0.01, 0.04, 0.03, 0.2]
    assert h.bonferroni_correction(p) == pytest.approx([0.04, 0.16, 0.12, 0.8])
    assert h.holm_correction(p) == pytest.approx([0.04, 0.09, 0.09, 0.2])
    assert h.benjamini_hochberg_correction(p) == pytest.approx(
        [0.04, 0.05333333333333334, 0.05333333333333334, 0.2])


@pytest.mark.parametrize("p", [[], [math.nan], [-0.1], [1.1]])
def test_multiple_comparison_adjustments_reject_invalid_p(p):
    for correction in (h.bonferroni_correction, h.holm_correction,
                       h.benjamini_hochberg_correction):
        with pytest.raises(ValueError):
            correction(p)


def test_proportion_z_test_known_result():
    result = h.proportion_z_test(60, 100, 0.5)
    assert result.statistic == pytest.approx(2.0)
    assert result.p_value == pytest.approx(0.04550026389635842)


def test_chi_square_goodness_of_fit_known_result():
    result = h.chi_square_goodness_of_fit([10, 20, 30], [20, 20, 20])
    assert result.statistic == pytest.approx(10.0)
    assert result.degrees_of_freedom == 2.0
    assert result.p_value == pytest.approx(math.exp(-5.0))


@pytest.mark.parametrize("call", [
    lambda: h.proportion_z_test(11, 10, 0.5),
    lambda: h.proportion_z_test(1, 10, 0.01),
    lambda: h.proportion_z_test(5, 10, 1.0),
    lambda: h.chi_square_goodness_of_fit([1], [1]),
    lambda: h.chi_square_goodness_of_fit([1, 2], [1, 1]),
    lambda: h.chi_square_goodness_of_fit([1, 2], [1, 0]),
])
def test_new_tests_reject_invalid_input(call):
    with pytest.raises(ValueError):
        call()
