# OLS QR benchmark

Run `python tests/benchmark_qr.py` after installing NumPy. This compares the public `mathbr.OLS.fit` path (including Python-list conversion and classical standard-error calculation) with `numpy.linalg.lstsq` on a prebuilt NumPy design matrix. The latter uses optimized numerical-library routines, so these timings are a practical end-to-end comparison, not isolated QR kernel timings. Each figure is the median of seven runs on prebuilt input; no warm-up or thread control was applied. Do not generalize these small samples into a performance guarantee.

Measured after the Householder QR change on Windows 10, Python 3.14.7, NumPy 2.5.3, mathbr 0.6.0:

| Rows | Features | mathbr ms | NumPy ms | Max absolute coefficient difference |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 5 | 0.024 | 0.014 | 2.220e-15 |
| 500 | 10 | 0.160 | 0.037 | 2.220e-15 |
| 2000 | 20 | 1.137 | 0.503 | 8.216e-15 |

Immediately before the change, the same script measured mathbr at 0.033, 0.203, and 1.675 ms on the three respective cases. These are single local benchmark runs, so the apparent gains are indicative, not a performance guarantee. The generated matrices are dense, full-rank normal draws with small Gaussian response noise and a fixed seed. The updated solver keeps coefficient differences below 9e-15 in these cases. Dedicated tests cover rank deficiency, small-scale features, weighted fits, and a wider dense design.
