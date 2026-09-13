import math

import pytest

import mathbr


def test_empirical_cdf_with_ties_and_unsorted_sample():
    np = mathbr.nonparametric
    assert np.empirical_cdf([2, 1, 2, 3], [0, 1, 1.5, 2, 3, 4]) == [
        0, 0.25, 0.25, 0.75, 1, 1]
    assert np.empirical_cdf([1], []) == []


def test_gaussian_kde_reference_and_symmetry():
    np = mathbr.nonparametric
    assert np.gaussian_kde([0], [0], 1) == pytest.approx([1 / math.sqrt(2 * math.pi)])
    left, center, right = np.gaussian_kde([-1, 1], [-1, 0, 1], 1)
    assert left == pytest.approx(right)
    assert center == pytest.approx(math.exp(-0.5) / math.sqrt(2 * math.pi))


def test_nadaraya_watson_reference_and_distant_query():
    np = mathbr.nonparametric
    assert np.nadaraya_watson([-1, 1], [2, 4], [0], 1) == pytest.approx([3])
    result = np.nadaraya_watson([0, 1], [2, 4], [100], 1)
    assert result == pytest.approx([4])


@pytest.mark.parametrize("call", [
    lambda np: np.empirical_cdf([], [0]),
    lambda np: np.empirical_cdf([1], [math.nan]),
    lambda np: np.gaussian_kde([1], [0], 0),
    lambda np: np.nadaraya_watson([0], [1, 2], [0], 1),
    lambda np: np.nadaraya_watson([0], [math.inf], [0], 1),
])
def test_invalid_nonparametric_inputs(call):
    with pytest.raises(ValueError):
        call(mathbr.nonparametric)
