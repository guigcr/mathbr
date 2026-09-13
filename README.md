# mathbr

A small C++ library, exposed to Python via [pybind11](https://github.com/pybind/pybind11), with machine-learning, statistical, econometric, and time-series models.

This project is mostly a way for me to learn C++, pybind11, and the math behind ML fundamentals by implementing them from scratch instead of just calling `numpy`/`sklearn`.

## Status

**v0.6.0 — early / work in progress.**

This is a first working version. The API and internal structure will likely change as I keep learning and adding features. Not meant for production use (yet).

## What's implemented so far

**Activation functions** (`mathbr.activations`)
- Sigmoid
- Tanh
- ReLU
- Leaky ReLU
- GELU 
- Swish / SiLU
- Softmax
- Derivatives for all of the above (for backpropagation)

**Loss functions** (`mathbr.losses`)
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Log Loss / Binary Cross-Entropy
- Derivatives for MSE

**Models**
- `LinearRegression` — trained with gradient descent (MSE)
- `LogisticRegression` — trained with gradient descent (Log Loss)
- `OLS` — ordinary least squares with an intercept, fitted by QR decomposition; exposes coefficients, classical standard errors, R², residual variance, and residual degrees of freedom
- `WLS` — weighted least squares with positive observation weights and the same diagnostics as OLS
- `Ridge`, `Lasso`, `ElasticNet` — penalized linear regression fitted by coordinate descent
- `IV2SLS` — instrumental-variables regression via two-stage least squares
- `FixedEffects` — entity fixed-effects panel regression via within transformation
- `AR(p)` — univariate autoregression with an intercept and recursive forecasts
- `VAR(p)` — vector autoregression with an intercept, recursive forecasts, and residual covariance
- `ARCH(1)` and `GARCH(1,1)` — Gaussian quasi-likelihood volatility models with variance forecasts

The gradient-descent `LinearRegression` is useful for studying optimization. Use `OLS` when you need least-squares estimates and basic fit diagnostics. `WLS` is useful when observations have different known precisions. The OLS/WLS standard errors are classical; WLS assumes weights proportional to inverse error variances. P-values, confidence intervals, robust errors, and inference for penalized models are not implemented yet.

Ridge, Lasso, and Elastic Net minimize `sum((y - prediction)^2) / (2n) + alpha * [l1_ratio * sum(abs(coef)) + (1 - l1_ratio) * sum(coef^2) / 2]`. Ridge fixes `l1_ratio=0`, Lasso fixes it to `1`, and Elastic Net lets you choose. The intercept is unpenalized. Inputs are **not** standardized automatically, so feature scale affects the penalty. Use `converged()` and `iterations()` to inspect optimization. These estimators return coefficients and predictions, not OLS-style standard errors.

`IV2SLS` projects each endogenous regressor onto the exogenous regressors and excluded instruments, then regresses the outcome on the exogenous variables and these projections. Its `first_stage_r_squared()` is overall first-stage R², **not** a weak-instrument test. Instruments must be relevant and satisfy the exclusion restriction for causal interpretation; the class cannot verify that assumption. It does not report standard errors because ordinary second-stage OLS errors are invalid for 2SLS.

`FixedEffects` removes entity means before estimating slopes. It reports a within R² and entity-specific intercepts for entities seen during fitting. Time-invariant regressors cannot be identified. It does not calculate clustered standard errors, time effects, or predictions for unseen entities.

`AR` and `VAR` fit lagged regressions by OLS. Their coefficient order is intercept followed by all series at lag 1, then lag 2, and so on. `VAR.residual_covariance()` divides the residual cross-products by effective observations minus coefficients per equation. Forecasts are recursive conditional means; there are no forecast intervals, lag-order selection, stationarity tests, or cointegration models yet.

`ARCH` and `GARCH` fit a constant sample mean and conditional variance by Gaussian quasi-maximum likelihood. For GARCH(1,1), `h[t] = omega + alpha * error[t-1]^2 + beta * h[t-1]`; ARCH(1) fixes `beta=0`. Parameters satisfy `omega>0`, `alpha,beta>=0`, and `alpha+beta<0.999`. Optimization uses a simple coordinate search, so inspect `converged()` and treat results as educational estimates rather than production-grade inference. No standard errors, asymmetric volatility, heavy-tailed innovations, or model selection are provided.

## Build

Install from the project directory (requires Python 3.9+ and a C++17 compiler):

On Windows, install Microsoft C++ Build Tools (MSVC 14.0+) before running the command below.

```bash
python -m pip install .
```

For development and tests:

```bash
python -m pip install -e . pytest
python -m pytest
```

The build installs `pybind11` automatically. You can still compile directly with `g++` on Unix-like systems:

```bash
g++ -O3 -Wall -shared -std=c++17 -fPIC -Iinclude $(python3 -m pybind11 --includes) main.cpp src/ols.cpp src/regularized.cpp src/econometrics.cpp src/time_series.cpp -o mathbr$(python3-config --extension-suffix)
```

Direct compilation requires `pybind11` installed in the active Python environment.

Inputs to the loss functions must have matching, nonempty lengths. Model training requires a nonempty dataset with the declared number of features in every row, a positive finite learning rate, and at least one epoch. Logistic labels must be 0 or 1. Invalid inputs raise `ValueError` in Python.

## Usage

```python
import mathbr

# activations
print(mathbr.activations.sigmoid(2.0))
print(mathbr.activations.relu_derivative(-1.0))

# losses
print(mathbr.losses.mse([1.0, 2.0], [1.1, 1.9]))

# linear regression
model = mathbr.LinearRegression(n_features=2)
X = [[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]]
y = [5.0, 4.0, 9.0]

model.fit(X, y, lr=0.01, epochs=500)
print(model.predict([1.0, 2.0]))
print(model.get_weights(), model.get_bias())

# logistic regression
clf = mathbr.LogisticRegression(n_features=2)
X_clf = [[1.0, 1.0], [5.0, 5.0], [1.0, 2.0], [6.0, 5.0]]
y_clf = [0, 1, 0, 1]

clf.fit(X_clf, y_clf, lr=0.1, epochs=1000)
print(clf.predict([2.0, 2.0]))
print(clf.predict_proba([2.0, 2.0]))

# statistical regression: intercept comes first
ols = mathbr.OLS(n_features=1)
ols.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 8.0])
print(ols.coefficients())       # about [0.8, 2.3]
print(ols.standard_errors())    # classical standard errors
print(ols.r_squared(), ols.degrees_of_freedom())

# observations with different precision
wls = mathbr.WLS(n_features=1)
wls.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 4.0], [1.0, 1.0, 0.1])
print(wls.coefficients(), wls.r_squared())

# regularized regression
ridge = mathbr.Ridge(n_features=1, alpha=1.0)
ridge.fit([[-1.0], [0.0], [1.0]], [-2.0, 0.0, 2.0])
print(ridge.intercept(), ridge.coefficients(), ridge.converged())
# mathbr.Lasso(...), mathbr.ElasticNet(..., l1_ratio=0.5) use the same fit API

# instrumental variables: one endogenous regressor and one excluded instrument
iv = mathbr.IV2SLS(n_exog=0, n_endog=1, n_instruments=1)
iv.fit(exog=[[], [], [], [], []],
       endog=[[0], [2], [1], [4], [3]],
       instruments=[[0], [1], [2], [3], [4]],
       y=[1, 5, 3, 9, 7])
print(iv.coefficients())  # intercept, then endogenous slope: about [1, 2]

# entity fixed effects on a small panel
fe = mathbr.FixedEffects(n_features=1)
fe.fit(X=[[0], [1], [2], [0], [1], [2]],
       y=[1, 3, 5, 10, 12, 14],
       entity_ids=[10, 10, 10, 20, 20, 20])
print(fe.coefficients(), fe.entity_intercept(20), fe.within_r_squared())

# autoregression and multivariate dynamics
ar = mathbr.AR(lags=1)
ar.fit([1, 2, 4, 8, 16, 32])
print(ar.coefficients(), ar.forecast(2))
var = mathbr.VAR(n_series=2, lags=1)
var.fit([[1, 2], [2, 1], [3, 3], [4, 2], [5, 4], [6, 3],
         [7, 5], [8, 4], [9, 6], [10, 5]])
print(var.forecast(2))

# volatility forecasts for a return series
garch = mathbr.GARCH()
garch.fit([-2.0, -1.0, 0.5, 1.0, 2.0] * 20)
print(garch.omega(), garch.alpha(), garch.beta())
print(garch.forecast_variance(3), garch.converged())
```

`OLS.fit` and `WLS.fit` require more rows than parameters (features plus intercept), finite values, and a full-rank design matrix. WLS additionally requires strictly positive finite weights. Their result methods require a successful `fit` first.

## Ideas for the next releases

To grow toward a library for statistical modeling, the next useful steps are:

1. A `summary()` table with coefficient names, t statistics, confidence intervals and p-values, backed by tested statistical distributions.
2. Heteroscedasticity-robust (HC0–HC3) and cluster-robust covariance estimates.
3. Formula syntax and pandas DataFrame integration in a separate Python layer, preserving a small C++ core.
4. Logistic-model inference, likelihood diagnostics, and generalized linear models.
5. NumPy array bindings and benchmarks before changing the current list-based API.
6. Valid 2SLS covariance estimates, weak-instrument diagnostics, and entity-clustered panel errors.
7. Robust time-series estimators, lag selection, stationarity checks, and forecast intervals.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current design and an incremental roadmap.
