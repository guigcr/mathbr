"""Compare cached multivariate-normal evaluation with repeated refactoring.

Run: python tests/benchmark_multivariate.py
Both paths use prebuilt Python lists and return equivalent distances.
"""

import random
from statistics import median
from time import perf_counter

import mathbr


def elapsed(call, repetitions=9):
    call()
    timings = []
    for _ in range(repetitions):
        start = perf_counter()
        call()
        timings.append((perf_counter() - start) * 1000)
    return median(timings)


if __name__ == "__main__":
    rng = random.Random(42)
    dimensions, rows = 12, 2000
    mean = [0.0] * dimensions
    covariance = [[1.0 if i == j else 0.15 for j in range(dimensions)]
                  for i in range(dimensions)]
    data = [[rng.gauss(0, 1) for _ in range(dimensions)] for _ in range(rows)]
    model = mathbr.distributions.MultivariateNormal(mean, covariance)

    def cached():
        return model.logpdf_batch(data)

    def refactor_each_row():
        return [mathbr.distributions.MultivariateNormal(mean, covariance).logpdf(row)
                for row in data]

    actual, expected = cached(), refactor_each_row()
    assert max(abs(a - b) for a, b in zip(actual, expected)) < 1e-12
    cached_ms, repeated_ms = elapsed(cached), elapsed(refactor_each_row)
    print(f"{rows} rows, {dimensions} dimensions; median of 9 warmed-up calls")
    print(f"Cached batch: {cached_ms:.3f} ms")
    print(f"Refactor every row: {repeated_ms:.3f} ms")
    print(f"Ratio: {repeated_ms / cached_ms:.1f}x")
