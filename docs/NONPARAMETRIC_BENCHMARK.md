# Nonparametric benchmark

Run `python tests/benchmark_nonparametric.py` after installing NumPy. This compares public mathbr calls with simple NumPy reference calculations for Gaussian KDE and Nadaraya–Watson kernel regression. Inputs are prebuilt, the bandwidth is 0.4, and each time is the median of nine warmed-up calls. Mathbr timings include Python-list conversion; NumPy inputs are already arrays. The script asserts agreement before timing.

Measured on Windows 10, Python 3.14.7, NumPy 2.5.3, mathbr 0.6.0:

| Method | Sample size | Queries | mathbr ms | NumPy ms | Maximum absolute difference |
|---|---:|---:|---:|---:|---:|
| KDE | 1,000 | 25 | 0.133 | 0.222 | 5.551e-16 |
| Kernel regression | 1,000 | 25 | 0.159 | 0.240 | 5.773e-15 |
| KDE | 10,000 | 100 | 4.980 | 4.763 | 2.054e-15 |
| Kernel regression | 10,000 | 100 | 5.556 | 5.156 | 4.663e-14 |

The earlier two-pass kernel-regression path took 0.190 and 6.753 ms in the respective cases. The new path performs one weight scan for ordinary queries, and uses a stable rescaling fallback for extreme queries. Timings vary across runs; profile target workloads before selecting a bandwidth or making production performance claims.
