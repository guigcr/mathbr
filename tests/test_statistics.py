import math
import random

import pytest

import mathbr


def test_weighted_quantile_and_stable_log_sum_exp():
    s = mathbr.statistics
    assert s.weighted_quantile([10, 0, 20], [2, 1, 1], 0.5) == 10
    assert s.weighted_quantile([10, 0, 20], [2, 1, 1], 0) == 0
    assert s.weighted_quantile([10, 0, 20], [2, 1, 1], 1) == 20
    assert s.log_sum_exp([1000, 1000]) == pytest.approx(1000 + math.log(2))
    with pytest.raises(ValueError):
        s.weighted_quantile([1], [0], 0.5)
    with pytest.raises(ValueError):
        s.weighted_quantile([1], [1], math.nan)
    with pytest.raises(ValueError):
        s.log_sum_exp([])


def test_empirical_mgf_and_stable_log_mgf():
    s = mathbr.statistics
    x = [-1, 0, 1]
    assert s.empirical_mgf(x, 0) == 1
    assert s.empirical_mgf(x, 0.5) == pytest.approx(sum(math.exp(0.5 * v) for v in x) / 3)
    assert s.log_empirical_mgf(x, 0.5) == pytest.approx(math.log(s.empirical_mgf(x, 0.5)))
    assert s.log_empirical_mgf([1000, 1000], 1) == pytest.approx(1000)
    with pytest.raises(ValueError):
        s.empirical_mgf([1000], 1)
    with pytest.raises(ValueError):
        s.log_empirical_mgf([1], math.inf)
    with pytest.raises(ValueError):
        s.log_empirical_mgf([], 0)


s = mathbr.statistics


def test_descriptive_statistics_known_values():
    x = [1.0, 2.0, 2.0, 3.0, 4.0]
    assert s.mean(x) == pytest.approx(2.4)
    assert s.median(x) == 2.0
    assert s.mode(x) == 2.0
    assert s.variance(x) == pytest.approx(1.3)
    assert s.standard_deviation(x) == pytest.approx(math.sqrt(1.3))
    assert s.quantile(x, 0.25) == 2.0
    assert s.percentile(x, 75.0) == 3.0
    assert s.interquartile_range(x) == 1.0
    assert s.five_number_summary(x) == [1.0, 2.0, 2.0, 3.0, 4.0]
    assert s.mode([2.0, 1.0]) == 1.0  # Smallest tied mode.


def test_moments_and_covariance_known_values():
    assert s.skewness([-1.0, 0.0, 1.0]) == pytest.approx(0.0)
    assert s.excess_kurtosis([-1.0, 0.0, 1.0]) == pytest.approx(-1.5)
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]
    assert s.covariance(x, y) == pytest.approx(2.0)
    assert s.pearson_correlation(x, y) == pytest.approx(1.0)
    assert s.pearson_correlation(x, [-v for v in y]) == pytest.approx(-1.0)
    assert s.spearman_correlation([1.0, 2.0, 2.0, 4.0],
                                  [10.0, 20.0, 20.0, 40.0]) == pytest.approx(1.0)


def test_weighted_statistics_known_values():
    x = [0.0, 2.0]
    y = [1.0, 5.0]
    weights = [1.0, 3.0]
    assert s.weighted_mean(x, weights) == pytest.approx(1.5)
    assert s.weighted_variance(x, weights) == pytest.approx(0.75)
    assert s.weighted_covariance(x, y, weights) == pytest.approx(1.5)
    assert s.weighted_correlation(x, y, weights) == pytest.approx(1.0)


def test_kendall_tau_b_with_ties_against_pairwise_reference():
    def reference(x, y):
        concordant = discordant = x_only = y_only = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                dx = (x[i] > x[j]) - (x[i] < x[j])
                dy = (y[i] > y[j]) - (y[i] < y[j])
                if dx and dy:
                    concordant += dx == dy
                    discordant += dx != dy
                elif dx:
                    y_only += 1
                elif dy:
                    x_only += 1
        return (concordant - discordant) / math.sqrt(
            (concordant + discordant + x_only) *
            (concordant + discordant + y_only))

    rng = random.Random(42)
    for n in (2, 5, 20, 75):
        for _ in range(20):
            x = [rng.randrange(5) for _ in range(n)]
            y = [rng.randrange(5) for _ in range(n)]
            if len(set(x)) > 1 and len(set(y)) > 1:
                assert s.kendall_tau(x, y) == pytest.approx(reference(x, y))
    assert s.kendall_tau([1, 2, 3], [3, 2, 1]) == pytest.approx(-1)


def test_covariance_and_correlation_matrices_match_scalar_apis():
    data = [[1, 4, 3], [2, 2, 1], [3, 4, 2], [4, 1, 4], [5, 3, 5]]
    columns = list(zip(*data))
    covariance = s.covariance_matrix(data)
    population = s.covariance_matrix(data, ddof=0)
    pearson = s.correlation_matrix(data)
    spearman = s.spearman_correlation_matrix(data)
    kendall = s.kendall_correlation_matrix(data)
    for i in range(3):
        for j in range(3):
            assert covariance[i][j] == pytest.approx(s.covariance(columns[i], columns[j]))
            assert population[i][j] == pytest.approx(s.covariance(columns[i], columns[j], 0))
            assert pearson[i][j] == pytest.approx(s.pearson_correlation(columns[i], columns[j]))
            assert spearman[i][j] == pytest.approx(s.spearman_correlation(columns[i], columns[j]))
            assert kendall[i][j] == pytest.approx(s.kendall_tau(columns[i], columns[j]))
    assert s.covariance_matrix([[2, 3]], 0) == [[0, 0], [0, 0]]


def test_weighted_matrices_match_scalar_statistics():
    data = [[1, 4, 3], [2, 2, 1], [3, 4, 2], [4, 1, 4], [5, 3, 5]]
    weights = [1, 3, 2, 5, 4]
    columns = list(zip(*data))
    covariance = s.weighted_covariance_matrix(data, weights)
    correlation = s.weighted_correlation_matrix(data, weights)
    for i in range(3):
        for j in range(3):
            assert covariance[i][j] == pytest.approx(
                s.weighted_covariance(columns[i], columns[j], weights))
            assert correlation[i][j] == pytest.approx(
                s.weighted_correlation(columns[i], columns[j], weights))
    assert s.weighted_covariance_matrix([[3, 8]], [2]) == [[0, 0], [0, 0]]


@pytest.mark.parametrize("weights", [[], [1], [1, 0], [1, math.inf], [1, math.nan]])
def test_weighted_matrix_invalid_weights(weights):
    with pytest.raises(ValueError):
        s.weighted_covariance_matrix([[1, 2], [2, 3]], weights)
    with pytest.raises(ValueError):
        s.weighted_correlation_matrix([[1, 2], [2, 3]], weights)


@pytest.mark.parametrize("data", [[], [[]], [[1, 2], [3]], [[1, math.nan]]])
def test_matrix_invalid_data(data):
    with pytest.raises(ValueError):
        s.covariance_matrix(data)


def test_matrix_invalid_parameters_and_constant_columns():
    with pytest.raises(ValueError):
        s.covariance_matrix([[1], [2]], ddof=2)
    with pytest.raises(ValueError):
        s.correlation_matrix([[1, 2], [2, 2]])
    with pytest.raises(ValueError):
        s.spearman_correlation_matrix([[1, 2], [2, 2]])
    with pytest.raises(ValueError):
        s.kendall_correlation_matrix([[1, 2], [2, 2]])
    with pytest.raises(ValueError):
        s.weighted_correlation_matrix([[1, 2], [2, 2]], [1, 3])


@pytest.mark.parametrize("call", [
    lambda: s.mean([]),
    lambda: s.mean([math.nan]),
    lambda: s.variance([1.0]),
    lambda: s.variance([1.0, 2.0], ddof=2),
    lambda: s.quantile([1.0], 1.1),
    lambda: s.percentile([1.0], -1.0),
    lambda: s.skewness([1.0, 1.0]),
    lambda: s.excess_kurtosis([1.0, 1.0]),
    lambda: s.covariance([1.0], [1.0, 2.0]),
    lambda: s.pearson_correlation([1.0, 1.0], [1.0, 2.0]),
    lambda: s.kendall_tau([1.0, 1.0], [1.0, 2.0]),
    lambda: s.kendall_tau([1.0], [1.0, 2.0]),
    lambda: s.weighted_mean([1.0, 2.0], [1.0, 0.0]),
    lambda: s.weighted_mean([1.0], [math.inf]),
])
def test_statistics_invalid_input(call):
    with pytest.raises(ValueError):
        call()
