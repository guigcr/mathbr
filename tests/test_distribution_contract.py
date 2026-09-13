"""Shared numerical contract for each continuous distribution in mathbr.

Register every new continuous distribution in CASES so the same properties run.
"""

import math

import pytest

import mathbr


d = mathbr.distributions
CASES = [
    ("normal", d.normal_pdf, d.normal_cdf, d.normal_ppf, -8.0, 8.0,
     (-2.0, -1.0, 0.0, 1.0, 2.0)),
    ("uniform", lambda x: d.uniform_pdf(x, -2.0, 3.0),
     lambda x: d.uniform_cdf(x, -2.0, 3.0),
     lambda p: d.uniform_ppf(p, -2.0, 3.0), -2.0, 3.0,
     (-1.5, -0.5, 0.5, 1.5, 2.5)),
    ("student_t", lambda x: d.student_t_pdf(x, 5.0),
     lambda x: d.student_t_cdf(x, 5.0),
     lambda p: d.student_t_ppf(p, 5.0), -200.0, 200.0,
     (-3.0, -1.0, 0.0, 1.0, 3.0)),
]


@pytest.mark.parametrize("name,pdf,cdf,ppf,low,high,interior", CASES,
                         ids=[case[0] for case in CASES])
def test_continuous_distribution_contract(name, pdf, cdf, ppf, low, high, interior):
    count = 1000
    step = (high - low) / count
    grid = [low + i * step for i in range(count + 1)]
    densities = [pdf(low + (i + 0.5) * step) for i in range(count)]
    probabilities = [cdf(x) for x in grid]

    assert all(math.isfinite(value) and value >= 0.0 for value in densities)
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert probabilities == sorted(probabilities)
    area_tolerance = 1e-4 if name == "student_t" else 1e-8
    assert sum(densities) * step == pytest.approx(1.0, abs=area_tolerance)

    for x in interior:
        h = 1e-5
        derivative = (cdf(x + h) - cdf(x - h)) / (2.0 * h)
        assert derivative == pytest.approx(pdf(x), rel=1e-8, abs=1e-10)

    for p in (0.001, 0.1, 0.5, 0.9, 0.999):
        assert cdf(ppf(p)) == pytest.approx(p, abs=1e-12)
