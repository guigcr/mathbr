"""Manual OLS benchmark against NumPy's LAPACK-backed least squares.

Run: python tests/benchmark_qr.py
NumPy is a benchmark-only dependency, not a mathbr runtime dependency.
"""

import platform
import statistics
import sys
from time import perf_counter

import mathbr
import numpy as np


def elapsed(call, repetitions=7):
    samples = []
    for _ in range(repetitions):
        start = perf_counter()
        call()
        samples.append((perf_counter() - start) * 1000)
    return statistics.median(samples)


def run_case(n, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p + 1)
    y = beta[0] + X @ beta[1:] + rng.normal(scale=0.01, size=n)
    design = np.column_stack((np.ones(n), X))
    rows = X.tolist()
    targets = y.tolist()

    def mathbr_fit():
        model = mathbr.OLS(p)
        model.fit(rows, targets)
        return np.asarray(model.coefficients())

    def numpy_fit():
        return np.linalg.lstsq(design, y, rcond=None)[0]

    reference = numpy_fit()
    observed = mathbr_fit()
    error = np.max(np.abs(observed - reference))
    print(f"{n:>5} {p:>4} {elapsed(mathbr_fit):>12.3f} "
          f"{elapsed(numpy_fit):>12.3f} {error:>14.3e}")


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}, "
          f"mathbr {mathbr.version}, {platform.platform()}")
    print("Rows  Vars  mathbr ms   NumPy ms    max abs error")
    for case in ((100, 5), (500, 10), (2000, 20)):
        run_case(*case, seed=42)
