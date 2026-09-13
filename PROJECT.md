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

For exact signatures, parameters, returns, and errors for every public symbol, see [docs/API_REFERENCE.md](docs/API_REFERENCE.md).
For a printable, example-driven introduction, see the [LaTeX user manual](docs/mathbr_manual.tex) or its [PDF](docs/mathbr_manual.pdf).

`mathbr.activations` provides `sigmoid`, `tanh_activation`, `relu`, `leaky_relu`, `gelu`, `swish`, and `softmax`. It also provides scalar derivatives for those activations except softmax. `softmax` takes a nonempty vector and returns probabilities that sum to one.

`mathbr.losses` provides `mse`, `mse_derivative`, `mae`, `rmse`, and `logloss`. Loss inputs must have equal, nonzero lengths. `logloss` takes binary 0/1 labels and probabilities in [0, 1]. Invalid inputs raise `ValueError` in Python.

```python
import mathbr

print(mathbr.activations.softmax([1.0, 2.0, 3.0]))
print(mathbr.losses.mse([1.0, 2.0], [1.1, 1.9]))
```

`mathbr.distributions` provides standard-normal `normal_pdf(x)`, `normal_logpdf(x)`, `normal_cdf(x)`, `normal_logcdf(x)`, and `normal_ppf(p)`. Inputs `x` must be finite; `p` must be finite and in [0, 1]. The quantile returns infinities at the two endpoints. This is the fixed standard normal distribution, not a fitted normal model.
It also provides standardized Student-t `student_t_pdf(x, df)`, `student_t_logpdf(x, df)`, `student_t_cdf(x, df)`, `student_t_logcdf(x, df)`, and `student_t_ppf(p, df)` for finite positive degrees of freedom. OLS/WLS use this distribution for classical coefficient p-values and confidence intervals.
Chi-squared and F distributions provide the same PDF/log-PDF, CDF/log-CDF, and PPF pattern as `chi_square_*(x, df)` and `f_*(x, df1, df2)`. Their degrees of freedom must be positive and finite. Their CDFs use incomplete-gamma and incomplete-beta routines; extreme-tail probabilities remain limited by double precision.
It also provides `uniform_pdf(x, a, b)`, `uniform_logpdf(x, a, b)`, `uniform_cdf(x, a, b)`, `uniform_logcdf(x, a, b)`, and `uniform_ppf(p, a, b)` for finite bounds with `a < b`.
For discrete binary trials, `bernoulli_pmf(k, p)`, `bernoulli_cdf(k, p)`, `binomial_pmf(k, n, p)`, and `binomial_cdf(k, n, p)` accept a success probability in [0, 1]; Binomial also requires nonnegative integer `n`. They assume independent trials with a common success probability.

```python
print(mathbr.distributions.normal_cdf(1.0))
print(mathbr.distributions.normal_ppf(0.975))
print(mathbr.distributions.student_t_ppf(0.975, 10.0))
print(mathbr.distributions.chi_square_ppf(0.95, 4.0))
print(mathbr.distributions.f_cdf(1.0, 5.0, 10.0))
print(mathbr.distributions.uniform_cdf(0.0, -1.0, 1.0))
print(mathbr.distributions.binomial_pmf(2, 4, 0.5))
```

## Regression and classification

`mathbr.statistics` provides descriptive summaries including mean, median, mode, sample variance, standard deviation, skewness, excess kurtosis, quantiles, five-number summaries, covariance, Pearson correlation, and weighted mean/variance/covariance/correlation. It also provides weighted quantiles and stable log-sum-exp. All input values must be finite. Sample variance and covariance default to `ddof=1`; weighted moments use the sum of positive weights without a degrees-of-freedom correction. See the API reference for exact signatures.

Additional submodules expose binary-classification and regression evaluation, regression diagnostics, nonparametric estimators, ACF/PACF and Ljung-Box statistics, right-censored survival analysis, and two conjugate Bayesian updates. The complete signatures, assumptions, and runnable examples are in `docs/API_REFERENCE.md` and `docs/mathbr_manual.tex`.
Spearman correlation is also available and uses average ranks for ties.

```python
print(mathbr.statistics.five_number_summary([1, 2, 2, 3, 4]))
print(mathbr.statistics.weighted_mean([0, 2], [1, 3]))
```

`mathbr.hypothesis` provides `one_sample_t_test(x, null_mean=0.0)`, `paired_t_test(before, after)`, and unequal-variance `welch_t_test(x, y)`. Each returns a `TTestResult` with `statistic`, `degrees_of_freedom`, and a two-sided `p_value`. Exact one-sample and paired reference distributions require Gaussian observations or differences; Welch uses an approximate degrees of freedom and assumes independent samples.
It also provides `bonferroni_correction(p_values)`, `holm_correction(p_values)`, and `benjamini_hochberg_correction(p_values)` for multiple testing. They return adjusted p-values in the original order. Benjamini-Hochberg controls false discovery rate under independence or suitable positive dependence; the other two target family-wise error.
`proportion_z_test(successes, trials, null_p)` is a large-sample two-sided test requiring at least five expected outcomes in each class. `chi_square_goodness_of_fit(observed, expected)` compares bin counts using `k-1` degrees of freedom when no parameters were fitted from those bins. Both return named result objects with a statistic and p-value; small expected counts weaken the chi-squared approximation.

```python
result = mathbr.hypothesis.welch_t_test([1, 2, 3], [2, 4, 6])
print(result.statistic, result.degrees_of_freedom, result.p_value)
print(mathbr.hypothesis.holm_correction([0.01, 0.04, 0.03]))
print(mathbr.hypothesis.proportion_z_test(60, 100, 0.5).p_value)
```

| Class | Estimation | Main results |
| --- | --- | --- |
| `LinearRegression(n_features)` | Full-batch gradient descent for MSE | `get_weights()`, `get_bias()`, `predict()`, `predict_batch()` |
| `LogisticRegression(n_features)` | Full-batch gradient descent for binary log loss | `get_weights()`, `get_bias()`, `predict()`, `predict_proba()`, `predict_proba_batch()` |
| `OLS(n_features)` | QR-based least squares | `coefficients()`, `standard_errors()`, `r_squared()`, `residual_variance()`, `degrees_of_freedom()` |
| `WLS(n_features)` | QR-based weighted least squares | The same results as OLS, using observation weights |
| `Ridge`, `Lasso`, `ElasticNet` | Cyclic coordinate descent | `coefficients()`, `intercept()`, `predict()`, `iterations()`, `converged()` |

`LinearRegression.fit(X, y, lr=0.01, epochs=1000)` and `LogisticRegression.fit(X, y, lr=0.01, epochs=1000)` take rows of features and target values. Logistic targets must be 0 or 1. Its `predict` method uses a 0.5 threshold by default and accepts another threshold in [0, 1].
Both gradient-descent fits accumulate gradients directly over each row and reuse their gradient buffer across epochs, avoiding per-epoch prediction arrays.

`OLS.fit(X, y)` includes an intercept automatically. Its coefficient vector starts with the intercept, followed by feature slopes. It requires more observations than parameters, finite values, and a full-rank design matrix. Its classical standard errors assume independent errors with constant variance.
It now also exposes `t_statistics()`, two-sided `p_values()`, `confidence_intervals(level=0.95)`, `adjusted_r_squared()`, and `f_statistic()`. Exact small-sample t/F inference additionally requires Gaussian errors; zero residual variance makes some statistics undefined.

`WLS.fit(X, y, weights)` uses the same coefficient order and requires strictly positive, finite weights. Classical WLS standard errors assume weights proportional to inverse error variances.
WLS inherits the same inferential methods. Their reference distributions additionally rely on independent Gaussian errors and correctly specified inverse-variance weights.

```python
ols = mathbr.OLS(n_features=1)
ols.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 8.0])
print(ols.coefficients(), ols.standard_errors(), ols.r_squared())
print(ols.t_statistics(), ols.p_values(), ols.confidence_intervals())

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
