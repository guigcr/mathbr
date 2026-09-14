import math

import pytest

import mathbr


d = mathbr.distributions


@pytest.mark.parametrize("name,args,lower,upper", [
    ("normal_sample", (), None, None),
    ("student_t_sample", (5,), None, None),
    ("chi_square_sample", (4,), 0, None),
    ("f_sample", (5, 10), 0, None),
    ("uniform_sample", (-2, 3), -2, 3),
    ("exponential_sample", (2,), 0, None),
    ("gamma_sample", (3, 2), 0, None),
    ("weibull_sample", (2, 3), 0, None),
    ("cauchy_sample", (1, 2), None, None),
    ("lognormal_sample", (0, 0.5), 0, None),
    ("laplace_sample", (1, 2), None, None),
    ("beta_sample", (2, 3), 0, 1),
])
def test_continuous_samples_reproduce_and_respect_support(name, args, lower, upper):
    sample = getattr(d, name)
    first = sample(1000, *args, seed=42)
    assert first == sample(1000, *args, seed=42)
    assert first != sample(1000, *args, seed=43)
    assert len(first) == 1000
    assert all(math.isfinite(value) for value in first)
    if lower is not None:
        assert all(value >= lower for value in first)
    if upper is not None:
        assert all(value <= upper for value in first)
    assert sample(0, *args, seed=42) == []


def test_continuous_sample_means_are_plausible():
    count = 20_000
    cases = (
        (d.normal_sample(count, 1), 0, 0.04),
        (d.chi_square_sample(count, 4, 1), 4, 0.08),
        (d.uniform_sample(count, -2, 3, 1), 0.5, 0.05),
        (d.exponential_sample(count, 2, 1), 0.5, 0.02),
        (d.gamma_sample(count, 3, 2, 1), 6, 0.12),
        (d.weibull_sample(count, 1, 3, 1), 3, 0.08),
        (d.laplace_sample(count, 1, 2, 1), 1, 0.07),
        (d.beta_sample(count, 2, 3, 1), 0.4, 0.01),
    )
    for sample, expected, tolerance in cases:
        assert sum(sample) / count == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize("call", [
    lambda: d.student_t_sample(1, 0, 1),
    lambda: d.chi_square_sample(1, math.nan, 1),
    lambda: d.f_sample(1, 2, -1, 1),
    lambda: d.uniform_sample(1, 2, 2, 1),
    lambda: d.exponential_sample(1, 0, 1),
    lambda: d.gamma_sample(1, 2, 0, 1),
    lambda: d.weibull_sample(1, math.inf, 1, 1),
    lambda: d.cauchy_sample(1, math.nan, 1, 1),
    lambda: d.lognormal_sample(1, 0, 0, 1),
    lambda: d.laplace_sample(1, 0, 0, 1),
    lambda: d.beta_sample(1, 0, 2, 1),
])
def test_continuous_sample_invalid_parameters(call):
    with pytest.raises(ValueError):
        call()
