# Local benchmark snapshot

Environment: Windows 10, Python 3.14.7, NumPy 2.5.3, mathbr 0.6.0. Values are median wall-clock milliseconds from a local rerun after the AUC optimization. They include Python-to-C++ list conversion for mathbr and NumPy's array operations for the reference. Repeat on the target machine before drawing deployment conclusions.

`python tests/benchmark_qr.py` (mathbr OLS versus NumPy least squares):

| Rows | Features | mathbr ms | NumPy ms | Max coefficient absolute error |
|---:|---:|---:|---:|---:|
| 100 | 5 | 0.024 | 0.014 | 2.220e-15 |
| 500 | 10 | 0.160 | 0.037 | 2.220e-15 |
| 2,000 | 20 | 1.137 | 0.503 | 8.216e-15 |

`python tests/benchmark_evaluation.py` (mathbr ROC AUC versus vectorized NumPy reference):

| Rows | mathbr ms | NumPy ms | Absolute AUC error |
|---:|---:|---:|---:|
| 100 | 0.004 | 0.026 | 5.551e-17 |
| 10,000 | 0.790 | 0.859 | 0 |
| 100,000 | 10.094 | 11.704 | 0 |

Before the AUC change, the same script measured 0.007, 1.042, and 14.327 ms for mathbr. The optimized path accumulates concordant scored pairs without allocating full ROC curve arrays; it produced the same values as the NumPy reference in this run. These are small local comparisons, not an exhaustive performance study. OLS still trails NumPy's optimized LAPACK-backed least squares in these cases.
