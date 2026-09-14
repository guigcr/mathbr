# Statistics matrix benchmark

Run `python tests/benchmark_statistics_matrices.py` to compare matrix APIs with the equivalent nested scalar calls. The benchmark uses 2,000 observation rows and 8 columns, prebuilt Python lists, and the median of nine warmed-up calls. Both paths include Python-to-C++ conversion. Results are empirical, not performance guarantees.

| Method | Matrix API | Repeated scalar calls | Speedup |
|---|---:|---:|---:|
| Pearson correlation | 0.447 ms | 3.697 ms | 8.26× |
| Weighted Pearson correlation | 0.479 ms | 6.492 ms | 13.55× |

Measured on the current Windows/Python 3.14 build. The script checks agreement to an absolute tolerance of `1e-10` before timing. Matrix APIs center columns once and compute only one triangle, while repeated scalar calls convert and validate vectors for every cell. Workload size, data shape, and machine load affect the measured ratio.
