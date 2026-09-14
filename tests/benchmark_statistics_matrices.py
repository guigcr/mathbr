"""Compare matrix APIs with equivalent repeated scalar calls.

Run: python tests/benchmark_statistics_matrices.py
Both paths include Python-to-C++ conversion and use prebuilt Python lists.
"""

import random
from statistics import median
from time import perf_counter

import mathbr


def elapsed(call, repetitions=9):
    call()
    samples = []
    for _ in range(repetitions):
        start = perf_counter()
        call()
        samples.append((perf_counter() - start) * 1000)
    return median(samples)


def scalar_matrix(columns, function, weights=None):
    return [[function(x, y, weights) if weights is not None else function(x, y)
             for y in columns] for x in columns]


if __name__ == "__main__":
    rng = random.Random(42)
    rows, count = 2000, 8
    data = [[rng.gauss(0, 1) for _ in range(count)] for _ in range(rows)]
    columns = [[row[j] for row in data] for j in range(count)]
    weights = [0.5 + rng.random() for _ in range(rows)]
    s = mathbr.statistics
    cases = (
        ("Pearson", lambda: s.correlation_matrix(data),
         lambda: scalar_matrix(columns, s.pearson_correlation)),
        ("Weighted Pearson", lambda: s.weighted_correlation_matrix(data, weights),
         lambda: scalar_matrix(columns, s.weighted_correlation, weights)),
    )
    print(f"{rows} rows, {count} columns; median of 9 warmed-up calls, milliseconds")
    print("Method                 Matrix API   Scalar calls   Speedup")
    for name, matrix_call, scalar_call in cases:
        actual, reference = matrix_call(), scalar_call()
        assert max(abs(actual[i][j] - reference[i][j])
                   for i in range(count) for j in range(count)) < 1e-10
        matrix_ms, scalar_ms = elapsed(matrix_call), elapsed(scalar_call)
        print(f"{name:<22} {matrix_ms:>10.3f} {scalar_ms:>14.3f} "
              f"{scalar_ms / matrix_ms:>9.2f}x")
