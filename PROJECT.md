# mathbr project guide

`mathbr` is a Python extension compiled from C++17. It includes mathematical functions, regression, econometric methods, and time-series models. Its primary purpose is to make the implementations available for study; it does not replace a production statistical library. The current version is 0.6.0.

## Installation and tests

The project requires Python 3.9+ and a C++17 compiler. Installing with `pip` installs the `pybind11` build dependency automatically. On Windows, install Microsoft C++ Build Tools before building the extension.

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e . pytest
python -m pytest
```

GitHub Actions builds and tests the project on Linux and Windows for each push or pull request. `pyproject.toml` declares package metadata and build dependencies; `setup.py` lists the C++ extension sources.

## Numerical API

`mathbr.activations` provides `sigmoid`, `tanh_activation`, `relu`, `leaky_relu`, `gelu`, `swish`, and `softmax`. It also provides scalar derivatives for those activations except softmax. `softmax` takes a nonempty vector and returns probabilities that sum to one.

`mathbr.losses` provides `mse`, `mse_derivative`, `mae`, `rmse`, and `logloss`. Loss inputs must have equal, nonzero lengths. `logloss` takes binary 0/1 labels and probabilities in [0, 1]. Invalid inputs raise `ValueError` in Python.

```python
import mathbr

print(mathbr.activations.softmax([1.0, 2.0, 3.0]))
print(mathbr.losses.mse([1.0, 2.0], [1.1, 1.9]))
```

## Regression and classification

| Class | Estimation | Main results |
| --- | --- | --- |
| `LinearRegression(n_features)` | Full-batch gradient descent for MSE | `get_weights()`, `get_bias()`, `predict()`, `predict_batch()` |
| `LogisticRegression(n_features)` | Full-batch gradient descent for binary log loss | `get_weights()`, `get_bias()`, `predict()`, `predict_proba()`, `predict_proba_batch()` |
| `OLS(n_features)` | QR-based least squares | `coefficients()`, `standard_errors()`, `r_squared()`, `residual_variance()`, `degrees_of_freedom()` |
| `WLS(n_features)` | QR-based weighted least squares | The same results as OLS, using observation weights |
| `Ridge`, `Lasso`, `ElasticNet` | Cyclic coordinate descent | `coefficients()`, `intercept()`, `predict()`, `iterations()`, `converged()` |

`LinearRegression.fit(X, y, lr=0.01, epochs=1000)` and `LogisticRegression.fit(X, y, lr=0.01, epochs=1000)` take rows of features and target values. Logistic targets must be 0 or 1. Its `predict` method uses a 0.5 threshold by default and accepts another threshold in [0, 1].

`OLS.fit(X, y)` includes an intercept automatically. Its coefficient vector starts with the intercept, followed by feature slopes. It requires more observations than parameters, finite values, and a full-rank design matrix. Its classical standard errors assume independent errors with constant variance.

`WLS.fit(X, y, weights)` uses the same coefficient order and requires strictly positive, finite weights. Classical WLS standard errors assume weights proportional to inverse error variances.

```python
ols = mathbr.OLS(n_features=1)
ols.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 8.0])
print(ols.coefficients(), ols.standard_errors(), ols.r_squared())

wls = mathbr.WLS(n_features=1)
wls.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 4.0], [1.0, 1.0, 0.1])
print(wls.coefficients())
```

`Ridge`, `Lasso`, and `ElasticNet` use `fit(X, y)`. Their constructors take `n_features`, `alpha=1.0`, `max_iter=1000`, and `tol=1e-8`. Elastic Net also takes `l1_ratio=0.5`. The objective is `SSE/(2n) + alpha * [l1_ratio * ||coef||_1 + (1-l1_ratio) * ||coef||_2²/2]`. Ridge fixes `l1_ratio=0`; Lasso fixes `l1_ratio=1`. The intercept is not penalized. Features are **not standardized automatically**, so their scale affects the penalty. Check `converged()` after fitting.

```python
ridge = mathbr.Ridge(n_features=1, alpha=1.0)
ridge.fit([[-1.0], [0.0], [1.0]], [-2.0, 0.0, 2.0])
print(ridge.intercept(), ridge.coefficients(), ridge.converged())
```

## Econometrics

`IV2SLS(n_exog, n_endog, n_instruments)` fits two-stage least squares. `fit(exog, endog, instruments, y)` takes separate matrices for exogenous features, endogenous features, and excluded instruments. `coefficients()` returns the intercept, exogenous coefficients, and then endogenous coefficients. `first_stage_r_squared()` returns the overall R² of each first stage; it is **not a weak-instrument test**. Instruments must be relevant and satisfy the exclusion restriction for a causal interpretation, which the class cannot verify. The class does not report standard errors because ordinary second-stage OLS standard errors are invalid for 2SLS.

```python
iv = mathbr.IV2SLS(n_exog=0, n_endog=1, n_instruments=1)
iv.fit(exog=[[], [], [], [], []],
       endog=[[0], [2], [1], [4], [3]],
       instruments=[[0], [1], [2], [3], [4]],
       y=[1, 5, 3, 9, 7])
print(iv.coefficients())
```

`FixedEffects(n_features)` estimates slopes from variation **within** each entity. `fit(X, y, entity_ids)` takes integer entity IDs. `coefficients()` returns the slopes, `entity_intercept(id)` returns an estimated entity intercept, and `within_r_squared()` measures fit after removing entity means. `predict(x, entity_id)` works only for entities present during fitting. Features that are constant within each entity cannot be identified. The implementation does not include time fixed effects or clustered standard errors.

```python
fe = mathbr.FixedEffects(n_features=1)
fe.fit([[0], [1], [2], [0], [1], [2]],
       [1, 3, 5, 10, 12, 14],
       [10, 10, 10, 20, 20, 20])
print(fe.coefficients(), fe.entity_intercept(20), fe.within_r_squared())
```

## Time series and volatility

`AR(lags)` regresses one series on its previous values. `VAR(n_series, lags)` does the same for multiple series, fitting one equation per variable. Both take observations in time order through `fit(...)` and produce recursive forecasts with `forecast(steps)`. For VAR, each row of `coefficients()` contains the intercept, all series at lag 1, all series at lag 2, and so on. `residual_covariance()` returns the residual covariance matrix.

```python
ar = mathbr.AR(lags=1)
ar.fit([1, 2, 4, 8, 16, 32])
print(ar.coefficients(), ar.forecast(2))
```

`ARCH(max_iter=2000, tol=1e-7)` implements ARCH(1). `GARCH(max_iter=2000, tol=1e-7)` implements GARCH(1,1). Both fit a constant mean and conditional variance by Gaussian quasi-maximum likelihood. For GARCH, `h[t] = omega + alpha * error[t-1]² + beta * h[t-1]`; ARCH fixes `beta=0`. The API exposes `mean()`, `omega()`, `alpha()`, `beta()`, `log_likelihood()`, `conditional_variance()`, `forecast_variance(steps)`, and `converged()`. Fitting requires at least 20 observations with positive variance. The optimizer is a simple coordinate search; check `converged()` before interpreting the parameters.

```python
garch = mathbr.GARCH()
garch.fit([-2.0, -1.0, 0.5, 1.0, 2.0] * 20)
print(garch.omega(), garch.alpha(), garch.beta())
print(garch.forecast_variance(3), garch.converged())
```

AR and VAR do not include automatic lag selection, stationarity tests, or forecast intervals. ARCH and GARCH do not include standard errors, asymmetry, or innovation distributions beyond Gaussian.

## Code organization and general limits

`main.cpp` defines the pybind11 module and contains the activation functions, losses, and two gradient-descent regressions. Additional models have declarations in `include/mathbr/` and implementations in `src/`. Tests live in `tests/`. See [ARCHITECTURE.md](ARCHITECTURE.md) for component dependencies and technical decisions.

The interfaces take Python lists converted to `std::vector`, which can copy data. There is no direct NumPy buffer, formula, or DataFrame integration yet. This is an evolving educational project; check each model's assumptions before using its results in an analysis.
