import math

import pytest

import mathbr


h = mathbr.hypothesis


def test_one_way_anova_known_f_and_p_value():
    result = h.one_way_anova([[1, 2, 3], [4, 5, 6]])
    assert result.statistic == pytest.approx(13.5)
    assert result.df_between == 1
    assert result.df_within == 4
    assert result.p_value == pytest.approx(1 - mathbr.distributions.f_cdf(13.5, 1, 4))
    equal_means = h.one_way_anova([[1, 2, 3], [1, 2, 3]])
    assert equal_means.statistic == pytest.approx(0)
    assert equal_means.p_value == pytest.approx(1)


def test_kruskal_wallis_known_ranks_and_tie_correction():
    result = h.kruskal_wallis([[1, 2, 3], [4, 5, 6]])
    assert result.statistic == pytest.approx(27 / 7)
    assert result.degrees_of_freedom == 1
    assert result.p_value == pytest.approx(
        1 - mathbr.distributions.chi_square_cdf(27 / 7, 1))
    tied = h.kruskal_wallis([[1, 2, 2], [2, 3, 4]])
    expected = (7 / 3) / (1 - 24 / 210)
    assert tied.statistic == pytest.approx(expected)
    assert 0 < tied.p_value < 1


@pytest.mark.parametrize("groups", [
    [], [[1]], [[1], []], [[1, 2], [math.nan]],
])
def test_group_tests_reject_invalid_groups(groups):
    for test in (h.one_way_anova, h.kruskal_wallis):
        with pytest.raises(ValueError):
            test(groups)


def test_group_tests_reject_degenerate_variation():
    with pytest.raises(ValueError):
        h.one_way_anova([[1, 1], [2, 2]])
    with pytest.raises(ValueError):
        h.kruskal_wallis([[1, 1], [1, 1]])


def test_chi_square_independence_known_table():
    result = h.chi_square_independence([[10, 20], [20, 10]])
    assert result.statistic == pytest.approx(20 / 3)
    assert result.degrees_of_freedom == 1
    assert result.p_value == pytest.approx(
        1 - mathbr.distributions.chi_square_cdf(20 / 3, 1))
    independent = h.chi_square_independence([[10, 20], [20, 40]])
    assert independent.statistic == pytest.approx(0)
    assert independent.p_value == pytest.approx(1)


@pytest.mark.parametrize("table", [
    [], [[1, 2]], [[1], [2]], [[1, 2], [3]],
    [[0, 0], [1, 2]], [[0, 1], [0, 2]],
    [[1, -1], [1, 1]], [[1, math.nan], [1, 1]],
])
def test_chi_square_independence_rejects_invalid_tables(table):
    with pytest.raises(ValueError):
        h.chi_square_independence(table)


def test_mann_whitney_u_with_and_without_ties():
    result = h.mann_whitney_u([1, 2, 3], [4, 5, 6])
    expected_z = -4 / math.sqrt(5.25)
    assert result.statistic == pytest.approx(0)
    assert result.z_score == pytest.approx(expected_z)
    assert result.p_value == pytest.approx(2 * mathbr.distributions.normal_cdf(expected_z))
    reversed_result = h.mann_whitney_u([4, 5, 6], [1, 2, 3])
    assert reversed_result.statistic == pytest.approx(9)
    assert reversed_result.z_score == pytest.approx(-expected_z)
    assert reversed_result.p_value == pytest.approx(result.p_value)
    tied = h.mann_whitney_u([1, 2, 2], [2, 3, 4])
    assert tied.statistic == pytest.approx(1.0)
    assert 0 < tied.p_value < 1


@pytest.mark.parametrize("x,y", [
    ([], [1]), ([1], []), ([1, math.nan], [2]), ([1], [math.inf]),
    ([1, 1], [1, 1]),
])
def test_mann_whitney_rejects_invalid_input(x, y):
    with pytest.raises(ValueError):
        h.mann_whitney_u(x, y)
