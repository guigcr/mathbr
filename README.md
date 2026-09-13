# mathbr

`mathbr` is an educational library of statistical and machine learning models written in C++17 and exposed to Python with pybind11. It implements the calculations directly to help readers study C++, pybind11, and the underlying mathematics.

**Version 0.6.0 — work in progress.** The API may change, and the library is not intended for production use yet.

## Features

- Activation and loss functions: sigmoid, ReLU, GELU, softmax, MSE, MAE, RMSE, and log loss.
- Regression: gradient-descent linear and logistic regression, OLS, WLS, Ridge, Lasso, and Elastic Net.
- Econometrics: IV/2SLS and entity fixed effects.
- Time series: AR(p), VAR(p), ARCH(1), and GARCH(1,1).

## Installation

Requires Python 3.9+ and a C++17 compiler. On Windows, install Microsoft C++ Build Tools first.

```bash
python -m pip install .
```

## Quick example

```python
import mathbr

model = mathbr.OLS(n_features=1)
model.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 8.0])

print(model.coefficients())  # intercept and slope: approximately [0.8, 2.3]
print(model.r_squared())
print(model.predict([4.0]))
```

For **all models, examples, assumptions, and limitations**, read [PROJECT.md](PROJECT.md). See [ARCHITECTURE.md](ARCHITECTURE.md) for the code organization and technical decisions.
