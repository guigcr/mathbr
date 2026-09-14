"""Measure Python call overhead for batched C++ sampling.

Run: python tests/benchmark_sampling.py
The paths do not produce the same random sequence: one seeds once per batch,
while the repeated-call path seeds once per single draw.
"""

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
    d = mathbr.distributions
    count = 10_000
    print("10,000 draws; median of 9 warmed-up calls; milliseconds")
    print("Distribution          Batch call   Repeated calls   Ratio")
    for name, batch, repeated in (
        ("Standard normal", lambda: d.normal_sample(count, 42),
         lambda: [d.normal_sample(1, i)[0] for i in range(count)]),
        ("Poisson", lambda: d.poisson_sample(count, 4.0, 42),
         lambda: [d.poisson_sample(1, 4.0, i)[0] for i in range(count)]),
    ):
        batch_ms = elapsed(batch)
        repeated_ms = elapsed(repeated)
        print(f"{name:<20} {batch_ms:>10.3f} {repeated_ms:>16.3f} "
              f"{repeated_ms / batch_ms:>7.1f}x")
