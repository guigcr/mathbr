# Batch sampling benchmark

Run `python tests/benchmark_sampling.py`. This measures 10,000 draws through one C++ call against 10,000 Python calls each drawing one value. The median of nine warmed-up timings is reported. These paths do **not** produce the same random stream: the batch path seeds once and the repeated-call path reseeds for every draw. This isolates the practical cost of repeated Python calls and generator construction; it is not a comparison of random-number quality.

| Distribution | One batch call | Repeated calls | Ratio |
|---|---:|---:|---:|
| Standard normal | 0.164 ms | 17.228 ms | 105.1× |
| Poisson, rate 4 | 0.262 ms | 17.341 ms | 66.2× |

Measured on the current Windows/Python 3.14 build. Timings depend on machine load, compiler, C++ standard library, parameters, and output size. Use the script to measure the target environment.
