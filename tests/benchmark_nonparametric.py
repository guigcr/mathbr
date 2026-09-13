"""Manual Gaussian KDE and kernel-regression benchmark against NumPy.

Run: python tests/benchmark_nonparametric.py
NumPy is benchmark-only; it is not a mathbr runtime dependency.
"""

from statistics import median
from time import perf_counter

import mathbr
import numpy as np


def elapsed(call, repetitions=9):
    call()
    samples = []
    for _ in range(repetitions):
        start = perf_counter()
        call()
        samples.append((perf_counter() - start) * 1000)
    return median(samples)


def numpy_kde(sample, points, bandwidth):
    return np.asarray([np.exp(-0.5 * ((point - sample) / bandwidth) ** 2).mean()
                       / (bandwidth * np.sqrt(2 * np.pi)) for point in points])


def numpy_kernel_regression(x, y, points, bandwidth):
    result = []
    for point in points:
        squared = ((point - x) / bandwidth) ** 2
        weights = np.exp(-0.5 * (squared - squared.min()))
        result.append(np.dot(weights, y) / weights.sum())
    return np.asarray(result)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    print(f"NumPy {np.__version__}, mathbr {mathbr.version}")
    print("Method          N   Queries   mathbr ms   NumPy ms   max abs error")
    for n, m in ((1_000, 25), (10_000, 100)):
        x = rng.normal(size=n)
        y = 1.0 + 2.0 * x + rng.normal(scale=0.1, size=n)
        points = np.linspace(-2, 2, m)
        x_list, y_list, point_list = x.tolist(), y.tolist(), points.tolist()
        for name, mathbr_call, numpy_call in (
            ("KDE", lambda: mathbr.nonparametric.gaussian_kde(x_list, point_list, 0.4),
             lambda: numpy_kde(x, points, 0.4)),
            ("Regression", lambda: mathbr.nonparametric.nadaraya_watson(
                x_list, y_list, point_list, 0.4),
             lambda: numpy_kernel_regression(x, y, points, 0.4)),
        ):
            observed = np.asarray(mathbr_call())
            reference = numpy_call()
            error = np.max(np.abs(observed - reference))
            assert error < 1e-10
            print(f"{name:<10} {n:>6} {m:>9} {elapsed(mathbr_call):>11.3f} "
                  f"{elapsed(numpy_call):>10.3f} {error:>15.3e}")
