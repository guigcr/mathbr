import math

import pytest

import mathbr


normal = mathbr.distributions


def test_standard_normal_reference_values():
    assert normal.normal_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi))
    assert normal.normal_logpdf(0.0) == pytest.approx(-0.5 * math.log(2.0 * math.pi))
    assert normal.normal_cdf(0.0) == pytest.approx(0.5)
    assert normal.normal_cdf(1.0) == pytest.approx(0.8413447460685429)
    assert normal.normal_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-12)
    assert normal.normal_ppf(0.0) == -math.inf
    assert normal.normal_ppf(1.0) == math.inf


def test_standard_normal_distribution_properties():
    grid = [-8.0 + i * 0.02 for i in range(801)]
    pdf = [normal.normal_pdf(x) for x in grid]
    cdf = [normal.normal_cdf(x) for x in grid]
    assert all(value >= 0.0 for value in pdf)
    assert all(0.0 <= value <= 1.0 for value in cdf)
    assert cdf == sorted(cdf)
    area = sum((pdf[i] + pdf[i + 1]) * 0.01 for i in range(800))
    assert area == pytest.approx(1.0, abs=1e-10)
    for x in (-3.0, -1.0, 0.0, 1.0, 3.0):
        h = 1e-5
        derivative = (normal.normal_cdf(x + h) - normal.normal_cdf(x - h)) / (2 * h)
        assert derivative == pytest.approx(normal.normal_pdf(x), rel=1e-9, abs=1e-11)
    for p in (1e-12, 0.001, 0.1, 0.5, 0.9, 0.999, 1 - 1e-12):
        assert normal.normal_cdf(normal.normal_ppf(p)) == pytest.approx(p, abs=1e-14)


def test_standard_normal_log_tail():
    assert normal.normal_logcdf(-40.0) == pytest.approx(-804.6084420137538, abs=1e-6)
    assert normal.normal_logcdf(0.0) == pytest.approx(math.log(0.5))
    assert normal.normal_logcdf(10.0) == pytest.approx(0.0, abs=1e-20)


@pytest.mark.parametrize("x", [math.nan, math.inf, -math.inf])
def test_standard_normal_rejects_nonfinite_inputs(x):
    for function in (normal.normal_pdf, normal.normal_logpdf,
                     normal.normal_cdf, normal.normal_logcdf):
        with pytest.raises(ValueError):
            function(x)


@pytest.mark.parametrize("p", [-0.1, 1.1, math.nan, math.inf])
def test_standard_normal_rejects_invalid_probabilities(p):
    with pytest.raises(ValueError):
        normal.normal_ppf(p)


def test_uniform_reference_values():
    assert normal.uniform_pdf(0.0, -2.0, 3.0) == pytest.approx(0.2)
    assert normal.uniform_logpdf(0.0, -2.0, 3.0) == pytest.approx(-math.log(5.0))
    assert normal.uniform_pdf(4.0, -2.0, 3.0) == 0.0
    assert normal.uniform_logpdf(4.0, -2.0, 3.0) == -math.inf
    assert normal.uniform_cdf(0.0, -2.0, 3.0) == pytest.approx(0.4)
    assert normal.uniform_logcdf(0.0, -2.0, 3.0) == pytest.approx(math.log(0.4))
    assert normal.uniform_ppf(0.4, -2.0, 3.0) == pytest.approx(0.0)
    assert normal.uniform_ppf(0.0, -2.0, 3.0) == -2.0
    assert normal.uniform_ppf(1.0, -2.0, 3.0) == 3.0


@pytest.mark.parametrize("a,b", [(1.0, 1.0), (2.0, 1.0), (math.nan, 1.0),
                                  (0.0, math.inf), (-1e308, 1e308)])
def test_uniform_rejects_invalid_bounds(a, b):
    for function in (normal.uniform_pdf, normal.uniform_logpdf,
                     normal.uniform_cdf, normal.uniform_logcdf):
        with pytest.raises(ValueError):
            function(0.0, a, b)
    with pytest.raises(ValueError):
        normal.uniform_ppf(0.5, a, b)


@pytest.mark.parametrize("p", [-0.01, 1.01, math.nan, math.inf])
def test_uniform_rejects_invalid_probability(p):
    with pytest.raises(ValueError):
        normal.uniform_ppf(p, 0.0, 1.0)


def test_bernoulli_and_binomial_reference_values():
    assert normal.bernoulli_pmf(0, 0.3) == pytest.approx(0.7)
    assert normal.bernoulli_pmf(1, 0.3) == pytest.approx(0.3)
    assert normal.bernoulli_cdf(0, 0.3) == pytest.approx(0.7)
    assert normal.binomial_pmf(2, 4, 0.5) == pytest.approx(0.375)
    assert normal.binomial_cdf(2, 4, 0.5) == pytest.approx(0.6875)
    assert normal.binomial_ppf(0.6875, 4, 0.5) == 2
    assert normal.binomial_ppf(0.6875001, 4, 0.5) == 3
    assert normal.binomial_pmf(-1, 4, 0.5) == 0.0
    assert normal.binomial_cdf(-1, 4, 0.5) == 0.0
    assert normal.binomial_cdf(4, 4, 0.5) == 1.0
    assert normal.binomial_pmf(0, 4, 0.0) == 1.0
    assert normal.binomial_pmf(4, 4, 1.0) == 1.0


def test_binomial_normalization_and_cdf_consistency():
    for n, p in ((0, 0.3), (10, 0.2), (50, 0.7)):
        masses = [normal.binomial_pmf(k, n, p) for k in range(n + 1)]
        assert sum(masses) == pytest.approx(1.0, abs=1e-12)
        cumulative = 0.0
        for k, mass in enumerate(masses):
            cumulative += mass
            assert normal.binomial_cdf(k, n, p) == pytest.approx(cumulative, abs=1e-12)


def test_binomial_quantile_contract():
    for n, p in ((0, 0.3), (10, 0.2), (50, 0.7), (1000, 0.5)):
        for q in (0, 0.01, 0.2, 0.5, 0.9, 1):
            k = int(normal.binomial_ppf(q, n, p))
            assert normal.binomial_cdf(k, n, p) >= q
            if k > 0 and 0 < q < 1:
                assert normal.binomial_cdf(k - 1, n, p) < q


@pytest.mark.parametrize("p", [-0.1, 1.1, math.nan, math.inf])
def test_discrete_distributions_reject_bad_probabilities(p):
    with pytest.raises(ValueError):
        normal.bernoulli_pmf(0, p)
    with pytest.raises(ValueError):
        normal.bernoulli_cdf(0, p)
    with pytest.raises(ValueError):
        normal.binomial_pmf(0, 2, p)
    with pytest.raises(ValueError):
        normal.binomial_cdf(0, 2, p)


def test_binomial_rejects_negative_trial_count():
    with pytest.raises(ValueError):
        normal.binomial_pmf(0, -1, 0.5)
    with pytest.raises(ValueError):
        normal.binomial_cdf(0, -1, 0.5)


def test_student_t_known_values():
    # One degree of freedom is the standard Cauchy distribution.
    assert normal.student_t_pdf(0.0, 1.0) == pytest.approx(1.0 / math.pi)
    assert normal.student_t_cdf(1.0, 1.0) == pytest.approx(0.75, abs=1e-12)
    assert normal.student_t_cdf(-1.0, 1.0) == pytest.approx(0.25, abs=1e-12)
    assert normal.student_t_ppf(0.75, 1.0) == pytest.approx(1.0, abs=1e-12)
    assert normal.student_t_ppf(0.975, 10.0) == pytest.approx(2.2281388519649385, abs=1e-10)
    assert normal.student_t_logpdf(0.0, 5.0) == pytest.approx(
        math.log(normal.student_t_pdf(0.0, 5.0)))
    assert normal.student_t_logcdf(-1.0, 5.0) == pytest.approx(
        math.log(normal.student_t_cdf(-1.0, 5.0)))
    assert normal.student_t_ppf(0.0, 5.0) == -math.inf
    assert normal.student_t_ppf(1.0, 5.0) == math.inf


@pytest.mark.parametrize("df", [0.0, -1.0, math.nan, math.inf])
def test_student_t_rejects_invalid_df(df):
    for function in (normal.student_t_pdf, normal.student_t_logpdf,
                     normal.student_t_cdf, normal.student_t_logcdf):
        with pytest.raises(ValueError):
            function(0.0, df)
    with pytest.raises(ValueError):
        normal.student_t_ppf(0.5, df)


def test_student_t_rejects_invalid_x_and_p():
    with pytest.raises(ValueError):
        normal.student_t_pdf(math.nan, 5.0)
    with pytest.raises(ValueError):
        normal.student_t_ppf(1.1, 5.0)


def test_chi_square_reference_and_round_trip():
    d = normal
    assert d.chi_square_pdf(0.0, 2.0) == pytest.approx(0.5)
    assert d.chi_square_cdf(2.0, 2.0) == pytest.approx(1.0 - math.exp(-1.0))
    assert d.chi_square_cdf(2.0, 4.0) == pytest.approx(1.0 - 2.0 / math.e)
    assert d.chi_square_ppf(0.95, 2.0) == pytest.approx(-2.0 * math.log(0.05))
    assert d.chi_square_pdf(-1.0, 2.0) == 0.0
    assert d.chi_square_logpdf(-1.0, 2.0) == -math.inf
    assert d.chi_square_logcdf(2.0, 2.0) == pytest.approx(
        math.log(d.chi_square_cdf(2.0, 2.0)))
    for df in (1.0, 2.0, 5.0, 20.0):
        for p in (0.01, 0.5, 0.95):
            assert d.chi_square_cdf(d.chi_square_ppf(p, df), df) == pytest.approx(p)


def test_f_distribution_reference_and_round_trip():
    d = normal
    assert d.f_cdf(1.0, 1.0, 1.0) == pytest.approx(0.5)
    assert d.f_cdf(3.0, 1.0, 1.0) == pytest.approx(
        2.0 / math.pi * math.atan(math.sqrt(3.0)))
    assert d.f_pdf(1.0, 1.0, 1.0) == pytest.approx(1.0 / (2.0 * math.pi))
    assert d.f_ppf(0.75, 1.0, 1.0) == pytest.approx(
        math.tan(0.75 * math.pi / 2.0) ** 2)
    assert d.f_cdf(-1.0, 5.0, 10.0) == 0.0
    assert d.f_logcdf(1.0, 5.0, 10.0) == pytest.approx(
        math.log(d.f_cdf(1.0, 5.0, 10.0)))
    for df1, df2 in ((1.0, 1.0), (5.0, 10.0), (10.0, 20.0)):
        for p in (0.01, 0.5, 0.95):
            assert d.f_cdf(d.f_ppf(p, df1, df2), df1, df2) == pytest.approx(p)


@pytest.mark.parametrize("df", [0.0, -1.0, math.nan, math.inf])
def test_chi_square_and_f_reject_bad_df(df):
    with pytest.raises(ValueError):
        normal.chi_square_cdf(1.0, df)
    with pytest.raises(ValueError):
        normal.f_cdf(1.0, df, 2.0)
    with pytest.raises(ValueError):
        normal.f_ppf(0.5, 2.0, df)


def test_chi_square_and_f_reject_bad_probability():
    with pytest.raises(ValueError):
        normal.chi_square_ppf(-0.1, 2.0)
    with pytest.raises(ValueError):
        normal.f_ppf(1.1, 2.0, 3.0)


@pytest.mark.parametrize("pdf,cdf", [
    (lambda x: normal.chi_square_pdf(x, 4.0),
     lambda x: normal.chi_square_cdf(x, 4.0)),
    (lambda x: normal.f_pdf(x, 5.0, 10.0),
     lambda x: normal.f_cdf(x, 5.0, 10.0)),
])
def test_positive_distribution_density_matches_cdf(pdf, cdf):
    for x in (0.5, 1.0, 2.0, 5.0):
        h = 1e-5
        derivative = (cdf(x + h) - cdf(x - h)) / (2.0 * h)
        assert derivative == pytest.approx(pdf(x), rel=1e-7, abs=1e-10)
