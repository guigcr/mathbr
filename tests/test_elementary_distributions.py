"""Contract and inverse-CDF checks for closed-form distribution families."""

import math

import pytest

import mathbr


@pytest.mark.parametrize("family,parameters,median,density", [
    ("exponential", (2.0,), math.log(2) / 2, 1.0),
    ("weibull", (2.0, 3.0), 3 * math.sqrt(math.log(2)), None),
    ("laplace", (1.0, 2.0), 1.0, 0.25),
    ("cauchy", (1.0, 2.0), 1.0, 1 / (2 * math.pi)),
    ("lognormal", (0.0, 1.0), 1.0, 1 / math.sqrt(2 * math.pi)),
])
def test_closed_form_families(family, parameters, median, density):
    d = mathbr.distributions
    pdf = getattr(d, family + "_pdf")
    cdf = getattr(d, family + "_cdf")
    ppf = getattr(d, family + "_ppf")
    assert ppf(0.5, *parameters) == pytest.approx(median)
    assert cdf(median, *parameters) == pytest.approx(0.5)
    if density is not None:
        assert pdf(median, *parameters) == pytest.approx(density)
    for p in (1e-4, 0.1, 0.25, 0.75, 0.9, 0.9999):
        assert cdf(ppf(p, *parameters), *parameters) == pytest.approx(p, abs=1e-11)
    with pytest.raises(ValueError):
        ppf(-0.1, *parameters)


def test_support_boundaries_and_validation():
    d = mathbr.distributions
    assert d.exponential_cdf(-1, 2) == 0
    assert d.weibull_pdf(-1, 2, 3) == 0
    assert d.lognormal_pdf(0, 0, 1) == 0
    assert d.weibull_pdf(0, 0.5, 1) == math.inf
    assert d.weibull_pdf(0, 1, 2) == 0.5
    assert d.laplace_ppf(0, 0, 1) == -math.inf
    assert d.cauchy_ppf(1, 0, 1) == math.inf
    assert d.exponential_ppf(1, 1) == math.inf
    assert d.lognormal_ppf(0, 0, 1) == 0
    assert d.weibull_pdf(1e300, 3, 1) == 0
    assert d.lognormal_pdf(1e-300, 0, 1e-300) == 0
    with pytest.raises(ValueError):
        d.exponential_pdf(1, 0)
    with pytest.raises(ValueError):
        d.weibull_cdf(1, 2, -1)
    with pytest.raises(ValueError):
        d.laplace_cdf(math.nan, 0, 1)
    with pytest.raises(ValueError):
        d.cauchy_pdf(1, 0, math.inf)


def test_gamma_and_beta_reference_values():
    d = mathbr.distributions
    assert d.gamma_pdf(1, 1, 2) == pytest.approx(math.exp(-0.5) / 2)
    assert d.gamma_cdf(1, 1, 2) == pytest.approx(1 - math.exp(-0.5))
    assert d.gamma_ppf(0.5, 1, 2) == pytest.approx(2 * math.log(2))
    assert d.gamma_pdf(1e300, 2, 1e-300) == 0
    assert d.gamma_cdf(1e300, 2, 1e-300) == 1
    assert d.beta_pdf(0.5, 2, 2) == pytest.approx(1.5)
    assert d.beta_cdf(0.5, 2, 2) == pytest.approx(0.5)
    assert d.beta_ppf(0.5, 2, 2) == pytest.approx(0.5)
    assert d.beta_pdf(0, 0.5, 2) == math.inf
    assert d.beta_pdf(1, 2, 1) == pytest.approx(2)
    for p in (0.01, 0.1, 0.9, 0.99):
        assert d.beta_cdf(d.beta_ppf(p, 2, 5), 2, 5) == pytest.approx(p)
        assert d.gamma_cdf(d.gamma_ppf(p, 2, 3), 2, 3) == pytest.approx(p)
    with pytest.raises(ValueError):
        d.beta_pdf(0.5, 0, 1)
    with pytest.raises(ValueError):
        d.gamma_ppf(0.5, 1, -1)
