# Cached multivariate-normal benchmark

Run `python tests/benchmark_multivariate.py`. It evaluates 2,000 rows of 12 dimensions using prebuilt Python lists. The cached path constructs `MultivariateNormal` once and calls `logpdf_batch`; the comparison constructs a new model and calls `logpdf` for every row. The script checks that values agree to `1e-12` before timing and reports the median of nine warmed-up runs.

| Cached batch | Refactor every row | Ratio |
|---:|---:|---:|
| 0.660 ms | 13.674 ms | 20.7× |

Measured on the current Windows/Python 3.14 build. The ratio includes repeated Python calls, parameter conversion, and Cholesky factorization in the comparison path. It illustrates the benefit of reusing the factor and a batch call for repeated evaluations; it is not a universal speed guarantee.
