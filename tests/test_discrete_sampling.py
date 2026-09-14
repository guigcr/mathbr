import pytest

import mathbr


d = mathbr.distributions


@pytest.mark.parametrize("name,args,minimum,maximum", [
    ("binomial_sample", (10, 0.3), 0, 10),
    ("geometric_sample", (0.3,), 1, None),
    ("poisson_sample", (4.0,), 0, None),
    ("negative_binomial_sample", (3, 0.5), 0, None),
])
def test_seeded_samples_reproduce_and_respect_support(name, args, minimum, maximum):
    sample = getattr(d, name)
    first = sample(500, *args, seed=42)
    assert first == sample(500, *args, seed=42)
    assert first != sample(500, *args, seed=43)
    assert len(first) == 500
    assert all(isinstance(value, int) and value >= minimum for value in first)
    if maximum is not None:
        assert all(value <= maximum for value in first)
    assert sample(0, *args, seed=42) == []


def test_seeded_sample_means_are_plausible():
    count = 20_000
    assert sum(d.binomial_sample(count, 10, 0.3, 5)) / count == pytest.approx(3, abs=0.06)
    assert sum(d.geometric_sample(count, 0.5, 5)) / count == pytest.approx(2, abs=0.06)
    assert sum(d.poisson_sample(count, 4, 5)) / count == pytest.approx(4, abs=0.08)
    assert sum(d.negative_binomial_sample(count, 3, 0.5, 5)) / count == pytest.approx(3, abs=0.1)


def test_multinomial_sample_reproducibility_and_counts():
    probabilities = [0.2, 0.3, 0.5]
    counts = d.multinomial_sample(20_000, probabilities, seed=42)
    assert counts == d.multinomial_sample(20_000, probabilities, seed=42)
    assert counts != d.multinomial_sample(20_000, probabilities, seed=43)
    assert sum(counts) == 20_000
    assert [value / 20_000 for value in counts] == pytest.approx(probabilities, abs=0.01)
    assert d.multinomial_sample(10, [0, 1, 0], seed=1) == [0, 10, 0]
    assert d.multinomial_sample(0, probabilities, seed=1) == [0, 0, 0]


def test_bernoulli_and_discrete_uniform_sampling():
    bernoulli = d.bernoulli_sample(1000, 0.3, seed=42)
    assert bernoulli == d.bernoulli_sample(1000, 0.3, seed=42)
    assert set(bernoulli) == {0, 1}
    assert d.bernoulli_sample(10, 0, seed=1) == [0] * 10
    assert d.bernoulli_sample(10, 1, seed=1) == [1] * 10
    uniform = d.discrete_uniform_sample(1000, -2, 2, seed=42)
    assert uniform == d.discrete_uniform_sample(1000, -2, 2, seed=42)
    assert set(uniform) == {-2, -1, 0, 1, 2}
    assert d.discrete_uniform_sample(10, 3, 3, seed=1) == [3] * 10


@pytest.mark.parametrize("call", [
    lambda: d.binomial_sample(1, -1, 0.5, 1),
    lambda: d.geometric_sample(1, 0, 1),
    lambda: d.poisson_sample(1, -1, 1),
    lambda: d.negative_binomial_sample(1, 0, 0.5, 1),
    lambda: d.multinomial_sample(-1, [1], 1),
    lambda: d.multinomial_sample(1, [0.2, 0.3], 1),
    lambda: d.bernoulli_sample(1, -0.1, 1),
    lambda: d.discrete_uniform_sample(1, 2, 1, 1),
])
def test_seeded_sample_invalid_parameters(call):
    with pytest.raises(ValueError):
        call()
