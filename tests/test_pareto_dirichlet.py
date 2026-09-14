import math

import pytest

import mathbr


d = mathbr.distributions


def test_pareto_reference_and_inverse_cdf():
    assert d.pareto_pdf(0.5, 2, 1) == 0
    assert d.pareto_cdf(0.5, 2, 1) == 0
    assert d.pareto_pdf(1, 2, 1) == pytest.approx(2)
    assert d.pareto_pdf(2, 2, 1) == pytest.approx(0.25)
    assert d.pareto_cdf(2, 2, 1) == pytest.approx(0.75)
    assert d.pareto_ppf(0.75, 2, 1) == pytest.approx(2)
    assert d.pareto_ppf(0, 2, 1) == 1
    assert d.pareto_ppf(1, 2, 1) == math.inf
    for p in (1e-8, 0.01, 0.5, 0.9, 0.999):
        assert d.pareto_cdf(d.pareto_ppf(p, 3, 2), 3, 2) == pytest.approx(p, abs=1e-12)


def test_pareto_seeded_sampling():
    first = d.pareto_sample(5000, 4, 2, 42)
    assert first == d.pareto_sample(5000, 4, 2, 42)
    assert first != d.pareto_sample(5000, 4, 2, 43)
    assert min(first) >= 2
    assert sum(first) / len(first) == pytest.approx(8 / 3, abs=0.07)
    assert d.pareto_sample(0, 4, 2, 42) == []


def test_dirichlet_density_and_seeded_sampling():
    assert d.dirichlet_pdf([0.2, 0.3, 0.5], [1, 1, 1]) == pytest.approx(2)
    assert d.dirichlet_logpdf([0.2, 0.3, 0.5], [1, 1, 1]) == pytest.approx(math.log(2))
    assert d.dirichlet_pdf([0.25, 0.75], [2, 3]) == pytest.approx(
        12 * 0.25 * 0.75**2)
    sample = d.dirichlet_sample(10_000, [2, 3, 5], 42)
    assert sample == d.dirichlet_sample(10_000, [2, 3, 5], 42)
    assert sample != d.dirichlet_sample(10_000, [2, 3, 5], 43)
    assert all(all(value >= 0 for value in row) for row in sample)
    assert all(sum(row) == pytest.approx(1, abs=1e-14) for row in sample)
    means = [sum(row[j] for row in sample) / len(sample) for j in range(3)]
    assert means == pytest.approx([0.2, 0.3, 0.5], abs=0.015)
    assert d.dirichlet_sample(0, [1, 1], 42) == []


@pytest.mark.parametrize("call", [
    lambda: d.pareto_pdf(1, 0, 1),
    lambda: d.pareto_cdf(math.nan, 2, 1),
    lambda: d.pareto_ppf(1.1, 2, 1),
    lambda: d.pareto_sample(1, 2, -1, 42),
    lambda: d.dirichlet_pdf([1], [1]),
    lambda: d.dirichlet_pdf([0.2, 0.2], [1, 1]),
    lambda: d.dirichlet_pdf([0, 1], [1, 1]),
    lambda: d.dirichlet_pdf([0.5, 0.5], [0, 1]),
    lambda: d.dirichlet_sample(1, [1], 42),
    lambda: d.dirichlet_sample(1, [1, math.nan], 42),
])
def test_pareto_dirichlet_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()
