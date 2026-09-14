import math

import pytest

import mathbr


def test_multivariate_normal_density_distance_and_batch():
    model = mathbr.distributions.MultivariateNormal(
        [0, 0], [[1, 0.5], [0.5, 2]])
    expected_logpdf = -math.log(2 * math.pi) - 0.5 * math.log(1.75) - 2 / 1.75
    assert model.logpdf([1, 2]) == pytest.approx(expected_logpdf)
    assert model.pdf([1, 2]) == pytest.approx(math.exp(expected_logpdf))
    assert model.mahalanobis_distance([1, 2]) == pytest.approx(math.sqrt(4 / 1.75))
    assert mathbr.statistics.mahalanobis_distance(
        [1, 2], [0, 0], [[1, 0.5], [0.5, 2]]) == pytest.approx(math.sqrt(4 / 1.75))
    data = [[0, 0], [1, 2], [-1, 1]]
    assert model.logpdf_batch(data) == pytest.approx([model.logpdf(row) for row in data])
    assert model.pdf_batch(data) == pytest.approx([model.pdf(row) for row in data])
    assert model.logpdf_batch([]) == []
    one_dimensional = mathbr.distributions.MultivariateNormal([0], [[1]])
    assert one_dimensional.pdf([0]) == pytest.approx(mathbr.distributions.normal_pdf(0))


def test_multivariate_normal_sample_reproducibility_and_moments():
    model = mathbr.distributions.MultivariateNormal([1, -2], [[1, 0.5], [0.5, 2]])
    sample = model.sample(20_000, seed=42)
    assert sample == model.sample(20_000, seed=42)
    assert sample != model.sample(20_000, seed=43)
    assert model.sample(0, seed=42) == []
    first = [row[0] for row in sample]
    second = [row[1] for row in sample]
    assert sum(first) / len(first) == pytest.approx(1, abs=0.04)
    assert sum(second) / len(second) == pytest.approx(-2, abs=0.05)
    assert mathbr.statistics.covariance(first, second, 0) == pytest.approx(0.5, abs=0.06)


@pytest.mark.parametrize("mean,covariance", [
    ([], []),
    ([0, 0], [[1]]),
    ([0, 0], [[1, 0], [0]]),
    ([0, math.nan], [[1, 0], [0, 1]]),
    ([0, 0], [[1, 0.2], [0.3, 1]]),
    ([0, 0], [[1, 1], [1, 1]]),
    ([0, 0], [[1, 2], [2, 1]]),
])
def test_multivariate_normal_rejects_invalid_parameters(mean, covariance):
    with pytest.raises(ValueError):
        mathbr.distributions.MultivariateNormal(mean, covariance)


def test_multivariate_normal_rejects_invalid_observations():
    model = mathbr.distributions.MultivariateNormal([0, 0], [[1, 0], [0, 1]])
    with pytest.raises(ValueError):
        model.logpdf([0])
    with pytest.raises(ValueError):
        model.pdf([math.nan, 0])
    with pytest.raises(ValueError):
        model.logpdf_batch([[0, 0], [0]])
