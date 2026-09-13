# mathbr: statistics in C++ for Python

`mathbr` is a statistics library written in C++17 and exposed to Python through pybind11. It provides descriptive statistics, probability distributions, statistical tests, regression, time series, and model evaluation. The project is intended for learning and experimentation. It is a work in progress, and the API may change.

---

## Download and usage

Download or clone this repository. Building the Python extension requires Python 3.9 or newer and a C++17 compiler. On Windows, install Microsoft C++ Build Tools first. From the project directory, run:

```bash
python -m pip install .
```

Here is a small ordinary least squares example:

```python
import mathbr

model = mathbr.OLS(1)
model.fit([[0.0], [1.0], [2.0], [3.0]],
          [1.0, 3.0, 5.0, 8.0])

print(model.coefficients())
print(model.predict([4.0]))
print(model.r_squared())
```

The coefficients are the intercept and slope, approximately 0.8 and 2.3. For development, install in editable mode and run the tests:

```bash
python -m pip install -e . pytest
python -m pytest
```

## Documentation

For a guided introduction, read the [user manual](docs/mathbr_manual.pdf). Its [LaTeX source](docs/mathbr_manual.tex) contains runnable examples. The [API reference](docs/API_REFERENCE.md) lists public functions, parameters, results, and errors. [PROJECT.md](PROJECT.md) gives a longer project guide with model assumptions.

To study the code, start with [ARCHITECTURE.md](ARCHITECTURE.md). C++ declarations are in `include/mathbr/`, implementations are in `src/`, and Python bindings are in [main.cpp](main.cpp). Correctness tests and benchmark scripts are in `tests/`. Benchmark reports are in `docs/`. Package build settings are in [pyproject.toml](pyproject.toml) and [setup.py](setup.py).

## Questions and feedback

If this project is hosted on GitHub, use its Issues page to report a bug, ask a question, or suggest a statistical method. Include a small reproducible example and the result you expected.

## Contributing

Contributions are welcome. Keep code, comments, tests, and documentation in English. When changing a public function, update the API reference and user manual, add a meaningful test, and run `python -m pytest` before opening a pull request.
