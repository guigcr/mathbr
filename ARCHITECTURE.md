# Architecture

## Purpose and scope

`mathbr` is a small educational C++17 extension for Python. It exposes numerical functions, regression models, IV/2SLS, entity fixed effects, autoregressions, and ARCH/GARCH volatility models. The goal is to make the math inspectable while providing a path toward a statistical-modeling API. It is not a replacement for `statsmodels` today.

## Current components

```text
Python caller
    │ lists / scalars
    ▼
main.cpp                 pybind11 module, activations, losses, GD models
    │
    ├── mathbr::OLS/WLS  include/mathbr/{ols,wls}.hpp + src/ols.cpp
    │                    QR fit, predictions, fit diagnostics
    ├── Ridge/Lasso/EN   include/mathbr/regularized.hpp + src/regularized.cpp
    │                    coordinate descent, predictions, convergence state
    ├── IV2SLS/FE       include/mathbr/econometrics.hpp + src/econometrics.cpp
    │                    staged OLS and within transformation
    └── AR/VAR/GARCH    include/mathbr/time_series.hpp + src/time_series.cpp
                         lagged OLS and conditional-variance likelihood
            ▼
       C++ standard library
```

`setup.py` declares the C++ extension and `pyproject.toml` provides build metadata and isolated build dependencies. `tests/` contains pytest checks for numerical results and invalid inputs. `README.md` provides a short overview; `PROJECT.md` documents the public API, examples, and model assumptions.

The Python module is named `mathbr`. Activation and loss functions live under `mathbr.activations` and `mathbr.losses`. Models are top-level classes. pybind11 converts Python sequences to `std::vector`, so calls involving arrays currently copy data. No NumPy, Eigen, BLAS, or external statistical-distribution dependency is required.

## Model behavior

| Model | Estimation | Public results | Constraints |
| --- | --- | --- | --- |
| `LinearRegression` | Full-batch gradient descent minimizing MSE | weights, bias, predictions, trained flag | Positive learning rate and epoch count |
| `LogisticRegression` | Full-batch gradient descent minimizing binary log loss | weights, bias, probabilities, class predictions | Binary labels; class threshold in [0, 1] |
| `OLS` | Modified Gram-Schmidt QR with a second orthogonalization pass | intercept and slopes, classical standard errors, R², residual variance, residual degrees of freedom | Finite full-rank matrix; more observations than parameters |
| `WLS` | QR on rows scaled by square-root weight | Same as OLS, using weighted residuals and weighted R² | OLS constraints plus positive finite weights |
| `Ridge`, `Lasso`, `ElasticNet` | Cyclic coordinate descent on centered features | intercept, slopes, predictions, iterations and convergence flag | Finite data; nonnegative penalty; positive iteration limit and tolerance |
| `IV2SLS` | First-stage OLS projections followed by second-stage OLS | intercept and slopes; first-stage overall R² | Sufficient excluded instruments and full-rank stages |
| `FixedEffects` | Demeaning by entity followed by OLS | within slopes, entity intercepts, within R² | Within-entity feature variation; known entity for prediction |
| `AR(p)`, `VAR(p)` | OLS on lagged observations | intercept and lag coefficients, recursive point forecasts; VAR residual covariance | Finite contiguous time series and full-rank lag matrix |
| `ARCH(1)`, `GARCH(1,1)` | Gaussian quasi-likelihood with constrained coordinate search | mean, variance parameters, conditional variance path and forecasts, convergence flag | At least 20 finite observations with positive variance |

OLS includes an intercept automatically. The coefficient vector is `[intercept, feature_1, ...]`. QR solves the least-squares system without forming `X'X`. The residual variance is `SSE / (n - p)` where `p` includes the intercept. Standard errors are the square roots of the diagonal of `residual_variance * (X'X)^-1`; they rely on independent, homoscedastic errors. A rank-deficient design raises `ValueError`. OLS result methods raise `RuntimeError` before fitting.

WLS shares the OLS solver and multiplies each design row and response by `sqrt(weight)`. Its residual variance uses weighted SSE divided by `n-p`; classical WLS inference assumes weights proportional to inverse error variances. Penalized models center features and response, then cycle through coefficients with soft thresholding. Their intercept is reconstructed after optimization and is never penalized. The penalty convention is documented in the README. These models expose no inferential standard errors.

IV2SLS reuses OLS for the first-stage instrument projections and the second-stage coefficient fit. It deliberately does not expose second-stage OLS standard errors, since those do not estimate the 2SLS covariance correctly. First-stage overall R² is descriptive and cannot by itself establish instrument strength or validity. FixedEffects demeans each feature and outcome by entity, fits the within regression, then reconstructs entity intercepts. Its R² describes within-entity fit. It does not compute clustered uncertainty or time fixed effects.

AR is a one-series wrapper around VAR. VAR builds a lag design in time order, fits each equation by OLS, stores the final observations, and recursively forecasts conditional means. It estimates a residual covariance matrix with a residual-degrees-of-freedom denominator. ARCH and GARCH center observations on their sample mean and optimize conditional Gaussian likelihood under nonnegative variance coefficients and a persistence bound. The optimizer is a simple coordinate search, so convergence is exposed rather than assumed. Variance forecasts use the last squared residual for horizon 1 and expected future squared residuals thereafter.

## Boundaries and limitations

`main.cpp` still combines bindings and older numerical code. Extracting activations, losses, and gradient-descent models into headers and source files is a reasonable next refactor. The current model API accepts lists of rows, not array buffers. OLS/WLS have no coefficient names, p-values, intervals, robust covariance, or formula interface. Penalized models do not standardize features. AR/VAR have no stationarity checks or forecast intervals; GARCH has no robust optimizer, inference, or innovation distributions beyond Gaussian. The QR and coordinate-descent implementations are intended for small to moderate dense problems and are not benchmarked against established numerical libraries.

## Evolution path

1. Add test fixtures for rank-deficient, ill-conditioned, and larger matrices; benchmark QR against a trusted numerical package before promising performance.
2. Move all numerical code out of `main.cpp`, keeping that file for bindings only. Introduce shared shape and finite-value validation.
3. Add a Python-facing result object with named coefficients and `summary()`. Add Student-t distribution support before exposing p-values or confidence intervals.
4. Add robust covariance estimators (HC0–HC3), valid IV covariance and weak-instrument diagnostics, entity-clustered panel errors, and generalized linear models.
5. Add NumPy buffer-based entry points for dense arrays and compare copy/compute costs with benchmarks. Keep list inputs if they remain useful for teaching.
6. Add formula and DataFrame support in Python, where column names and missing-data rules can be handled explicitly.
7. Add lag-order selection, stationarity and cointegration tests, VAR impulse responses, GARCH optimizer improvements, and forecast uncertainty.

Each numerical feature should have known-answer tests, invalid-input tests, and clear assumptions in its documentation.
