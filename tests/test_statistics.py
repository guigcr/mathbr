import math

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
    lambda: s.weighted_mean([1.0, 2.0], [1.0, 0.0]),
    lambda: s.weighted_mean([1.0], [math.inf]),
])
def test_statistics_invalid_input(call):
    with pytest.raises(ValueError):
        call()
