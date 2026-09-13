# mathbr API reference

All examples start with `import mathbr`. Sequence inputs are copied into C++ vectors. Invalid numerical input generally raises `ValueError`; requesting an unavailable fitted result raises `RuntimeError`. Python argument type conversion may raise `TypeError` before C++ validation.

## Contents

- [Activations](#activations)
- [Losses](#losses)
- [Distributions](#distributions)
- [Descriptive statistics](#descriptive-statistics)
- [Hypothesis tests](#hypothesis-tests)
- [Regression and classification](#regression-and-classification)
- [OLS and WLS](#ols-and-wls)
- [Regularized regression](#regularized-regression)
- [Econometrics](#econometrics)
- [Time series](#time-series)
- [Volatility](#volatility)

## Activations

Every scalar activation accepts `z: float` and returns a `float`. These functions do not validate finite input; non-finite values may propagate. Derivatives are with respect to `z`.

| Signature | Description and return | Example |
| --- | --- | --- |
| `mathbr.activations.sigmoid(z)` | Stable logistic activation in [0, 1]; `float`. | `mathbr.activations.sigmoid(0.0)` |
| `mathbr.activations.sigmoid_derivative(z)` | Logistic derivative; `float`. | `mathbr.activations.sigmoid_derivative(0.0)` |
| `mathbr.activations.tanh_activation(z)` | Hyperbolic tangent; `float`. | `mathbr.activations.tanh_activation(1.0)` |
| `mathbr.activations.tanh_derivative(z)` | Hyperbolic-tangent derivative; `float`. | `mathbr.activations.tanh_derivative(1.0)` |
| `mathbr.activations.relu(z)` | Rectified linear unit; `float`. | `mathbr.activations.relu(-1.0)` |
| `mathbr.activations.relu_derivative(z)` | ReLU derivative, defined here as zero at zero; `float`. | `mathbr.activations.relu_derivative(0.0)` |
| `mathbr.activations.leaky_relu(z, alpha=0.01)` | Leaky ReLU; `float`. `alpha: float` is the negative-side slope and is not range-checked. | `mathbr.activations.leaky_relu(-1.0)` |
| `mathbr.activations.leaky_relu_derivative(z, alpha=0.01)` | Leaky ReLU derivative; `float`. At zero it returns `alpha`. | `mathbr.activations.leaky_relu_derivative(0.0)` |
| `mathbr.activations.gelu(z)` | Tanh-approximate GELU; `float`. | `mathbr.activations.gelu(1.0)` |
| `mathbr.activations.gelu_derivative(z)` | Derivative of the tanh approximation; `float`. | `mathbr.activations.gelu_derivative(1.0)` |
| `mathbr.activations.swish(z)` | `z * sigmoid(z)`; `float`. | `mathbr.activations.swish(1.0)` |
| `mathbr.activations.swish_derivative(z)` | Swish derivative; `float`. | `mathbr.activations.swish_derivative(1.0)` |

`mathbr.activations.softmax(z)` takes a nonempty sequence of floats and returns a list of probabilities of the same length. It subtracts the maximum input for stability, but does not explicitly reject non-finite values. An empty sequence raises `ValueError`. Example: `mathbr.activations.softmax([1.0, 2.0])`.

## Losses

`y_true` and `y_pred` must be equal-length, nonempty sequences. A length violation raises `ValueError`. Except for `logloss`, finite input is not explicitly checked; non-finite values may propagate.

| Signature | Description and return | Example |
| --- | --- | --- |
| `mathbr.losses.mse(y_true, y_pred)` | Mean squared error; `float`. | `mathbr.losses.mse([1.0], [2.0])` |
| `mathbr.losses.mse_derivative(y_true, y_pred)` | Derivative with respect to each prediction; list of floats. | `mathbr.losses.mse_derivative([1.0], [2.0])` |
| `mathbr.losses.mae(y_true, y_pred)` | Mean absolute error; `float`. | `mathbr.losses.mae([1.0], [2.0])` |
| `mathbr.losses.rmse(y_true, y_pred)` | Root mean squared error; `float`. | `mathbr.losses.rmse([1.0], [2.0])` |
| `mathbr.losses.logloss(y_true, y_pred)` | Mean binary log loss; `float`. Labels must be integers 0 or 1 and predictions finite probabilities in [0, 1]; violations raise `ValueError`. Probabilities are clipped to [1e-15, 1-1e-15] inside the logarithm. | `mathbr.losses.logloss([0, 1], [0.1, 0.9])` |

These are in-sample loss functions, not cross-validation or held-out scoring tools.

## Distributions

The standard normal distribution has mean zero and variance one. All functions below return a `float`. Inputs are scalar; non-finite `x` and probabilities outside [0, 1] raise `ValueError`. The distribution does not accept estimated mean or scale parameters.

| Signature | Description and parameters | Returns | Raises | Assumptions / limitations | Minimal example |
| --- | --- | --- | --- | --- | --- |
| `mathbr.distributions.normal_pdf(x)` | Density at finite `x: float`. | Nonnegative density. | `ValueError` for non-finite `x`. | Values far into the tails may underflow to zero; use `normal_logpdf` there. | `mathbr.distributions.normal_pdf(0.0)` |
| `mathbr.distributions.normal_logpdf(x)` | Log density at finite `x: float`. | Log density. | `ValueError` for non-finite `x`. | Standard normal only. | `mathbr.distributions.normal_logpdf(0.0)` |
| `mathbr.distributions.normal_cdf(x)` | Cumulative probability at finite `x: float`, computed with `erfc`. | Probability in [0, 1]. | `ValueError` for non-finite `x`. | Extreme tails can round to zero or one; use `normal_logcdf` for a log lower tail. | `mathbr.distributions.normal_cdf(1.0)` |
| `mathbr.distributions.normal_logcdf(x)` | Log cumulative probability at finite `x: float`; uses a lower-tail asymptotic expansion below -10. | Log probability. | `ValueError` for non-finite `x`. | The asymptotic tail is approximate. | `mathbr.distributions.normal_logcdf(-40.0)` |
| `mathbr.distributions.normal_ppf(p)` | Quantile for finite `p: float` in [0, 1]. | Real quantile; returns negative/positive infinity at 0/1. | `ValueError` outside [0, 1] or for non-finite `p`. | Bisection is robust but slower than a dedicated inverse-CDF approximation. | `mathbr.distributions.normal_ppf(0.975)` |

The standardized Student-t distribution uses finite positive `df: float` degrees of freedom and finite scalar `x: float`. Its density has heavier tails than the normal distribution. It does not estimate `df`, location, or scale. Invalid `df`, non-finite `x`, or `p` outside [0, 1] raises `ValueError`; an incomplete-beta convergence failure raises `RuntimeError`.

| Signature | Description and parameters | Returns | Raises | Assumptions / limitations | Minimal example |
| --- | --- | --- | --- | --- | --- |
| `mathbr.distributions.student_t_pdf(x, df)` | Student-t density at `x`. | Nonnegative float. | `ValueError` for invalid input. | Extreme tails may underflow; use log-PDF. | `mathbr.distributions.student_t_pdf(0.0, 5.0)` |
| `mathbr.distributions.student_t_logpdf(x, df)` | Log density at `x`. | Float. | `ValueError` for invalid input. | Standardized distribution only. | `mathbr.distributions.student_t_logpdf(0.0, 5.0)` |
| `mathbr.distributions.student_t_cdf(x, df)` | Cumulative probability from the regularized incomplete beta function. | Float in [0, 1]. | `ValueError` for invalid input; `RuntimeError` if the continued fraction fails to converge. | Extreme tails may round; use log-CDF. | `mathbr.distributions.student_t_cdf(1.0, 5.0)` |
| `mathbr.distributions.student_t_logcdf(x, df)` | Log cumulative probability, retaining lower-tail precision. | Float. | `ValueError` for invalid input; `RuntimeError` on nonconvergence. | Double precision limits extreme-parameter accuracy. | `mathbr.distributions.student_t_logcdf(-10.0, 5.0)` |
| `mathbr.distributions.student_t_ppf(p, df)` | Quantile for finite `p` in [0, 1], found by bracketed bisection. | Float; infinities at the endpoints. | `ValueError` for invalid input; `RuntimeError` on nonconvergence. | More costly than a closed-form approximation; very extreme tails are limited by double precision. | `mathbr.distributions.student_t_ppf(0.975, 10.0)` |

Chi-squared uses finite positive degrees of freedom `df`; F uses finite positive `df1` and `df2`. Both have support on nonnegative `x`, return zero density and CDF below zero, and return infinity at `ppf(1, ...)`. Invalid parameters, non-finite `x`, or probabilities outside [0, 1] raise `ValueError`. Their CDFs use incomplete gamma or beta calculations; failure to converge raises `RuntimeError`. Very extreme-tail CDFs may round to zero or one in double precision, and log-CDF may then be infinite.

| Signature | Description and parameters | Returns | Raises / limitations | Minimal example |
| --- | --- | --- | --- | --- |
| `mathbr.distributions.chi_square_pdf(x, df)` | Chi-squared density at finite `x`. | Nonnegative float; may be infinite at zero for `df < 2`. | `ValueError` for invalid input. | `mathbr.distributions.chi_square_pdf(2, 4)` |
| `mathbr.distributions.chi_square_logpdf(x, df)` | Log density. | Float. | Same input rules; infinity at zero is possible. | `mathbr.distributions.chi_square_logpdf(2, 4)` |
| `mathbr.distributions.chi_square_cdf(x, df)` | Regularized incomplete-gamma CDF. | Float in [0, 1]. | `RuntimeError` on numerical nonconvergence. | `mathbr.distributions.chi_square_cdf(2, 4)` |
| `mathbr.distributions.chi_square_logcdf(x, df)` | Log CDF. | Float. | May be negative infinity in extreme lower tails. | `mathbr.distributions.chi_square_logcdf(2, 4)` |
| `mathbr.distributions.chi_square_ppf(p, df)` | Quantile by bracketed bisection. | Nonnegative float. | Finite precision limits extreme tails. | `mathbr.distributions.chi_square_ppf(0.95, 4)` |
| `mathbr.distributions.f_pdf(x, df1, df2)` | F density at finite `x`. | Nonnegative float; may be infinite at zero for `df1 < 2`. | `ValueError` for invalid input. | `mathbr.distributions.f_pdf(1, 5, 10)` |
| `mathbr.distributions.f_logpdf(x, df1, df2)` | Log density. | Float. | Same input rules. | `mathbr.distributions.f_logpdf(1, 5, 10)` |
| `mathbr.distributions.f_cdf(x, df1, df2)` | Regularized incomplete-beta CDF. | Float in [0, 1]. | `RuntimeError` on numerical nonconvergence. | `mathbr.distributions.f_cdf(1, 5, 10)` |
| `mathbr.distributions.f_logcdf(x, df1, df2)` | Log CDF. | Float. | May be negative infinity in extreme lower tails. | `mathbr.distributions.f_logcdf(1, 5, 10)` |
| `mathbr.distributions.f_ppf(p, df1, df2)` | Quantile by bracketed bisection. | Nonnegative float. | Finite precision limits extreme tails. | `mathbr.distributions.f_ppf(0.95, 5, 10)` |

The uniform distribution uses finite bounds `a: float`, `b: float` with `a < b` and finite width. Invalid bounds, non-finite `x`, or `p` outside [0, 1] raise `ValueError`. The density includes both endpoints; this boundary convention has no effect on continuous probabilities.

| Signature | Description and parameters | Returns | Raises | Assumptions / limitations | Minimal example |
| --- | --- | --- | --- | --- | --- |
| `mathbr.distributions.uniform_pdf(x, a, b)` | Uniform density at finite `x`. | `1/(b-a)` inside [a, b], zero outside. | `ValueError` for invalid input. | Fixed finite interval. | `mathbr.distributions.uniform_pdf(0.0, -1.0, 1.0)` |
| `mathbr.distributions.uniform_logpdf(x, a, b)` | Log uniform density at finite `x`. | `-log(b-a)` inside; negative infinity outside. | `ValueError` for invalid input. | Fixed finite interval. | `mathbr.distributions.uniform_logpdf(0.0, -1.0, 1.0)` |
| `mathbr.distributions.uniform_cdf(x, a, b)` | Uniform cumulative probability at finite `x`. | Zero below `a`, one above `b`, linear within. | `ValueError` for invalid input. | Fixed finite interval. | `mathbr.distributions.uniform_cdf(0.0, -1.0, 1.0)` |
| `mathbr.distributions.uniform_logcdf(x, a, b)` | Log cumulative probability at finite `x`. | Log CDF, including negative infinity below `a`. | `ValueError` for invalid input. | Fixed finite interval. | `mathbr.distributions.uniform_logcdf(0.0, -1.0, 1.0)` |
| `mathbr.distributions.uniform_ppf(p, a, b)` | Quantile for finite `p` in [0, 1]. | `a + p(b-a)`, with exact endpoints. | `ValueError` for invalid input. | Fixed finite interval. | `mathbr.distributions.uniform_ppf(0.5, -1.0, 1.0)` |

Bernoulli and Binomial accept integer `k` and finite success probability `p` in [0, 1]. Binomial also accepts nonnegative integer trial count `n`. An out-of-support `k` returns zero mass; CDFs return zero below support and one above it. Invalid `p` or `n` raises `ValueError`. These functions describe independent identically distributed binary trials; they do not fit or test a regression model.

| Signature | Description and parameters | Returns | Raises | Assumptions / limitations | Minimal example |
| --- | --- | --- | --- | --- | --- |
| `mathbr.distributions.bernoulli_pmf(k, p)` | Probability of integer outcome `k` for one trial. | Probability mass as float. | `ValueError` for invalid `p`. | One binary trial. | `mathbr.distributions.bernoulli_pmf(1, 0.3)` |
| `mathbr.distributions.bernoulli_cdf(k, p)` | Cumulative probability through integer `k`. | Probability as float. | `ValueError` for invalid `p`. | One binary trial. | `mathbr.distributions.bernoulli_cdf(0, 0.3)` |
| `mathbr.distributions.binomial_pmf(k, n, p)` | Probability of `k` successes in `n` trials. | Probability mass as float. | `ValueError` for invalid `n` or `p`. | Independent trials with common `p`; extreme masses may underflow. | `mathbr.distributions.binomial_pmf(2, 4, 0.5)` |
| `mathbr.distributions.binomial_cdf(k, n, p)` | Cumulative probability through `k` successes. | Probability as float. | `ValueError` for invalid `n` or `p`. | Direct tail summation costs O(n) in the worst case. | `mathbr.distributions.binomial_cdf(2, 4, 0.5)` |

`exponential_pdf/cdf/ppf` take a positive `rate`. `weibull_pdf/cdf/ppf` take positive `shape` and `scale`. `laplace_pdf/cdf/ppf` and `cauchy_pdf/cdf/ppf` take finite `location` and positive `scale`. `lognormal_pdf/cdf/ppf` take finite `mu` and positive `sigma` for the underlying normal. PDF/CDF calls take finite `x`, PPF calls take `p` in `[0,1]`, and invalid parameters raise `ValueError`. Boundary quantiles return the support endpoints, including infinity when appropriate. Weibull density at zero is infinite for shape below one.

```python
d = mathbr.distributions
print(d.exponential_ppf(0.5, rate=2.0))
print(d.weibull_cdf(3.0, shape=2.0, scale=3.0))
print(d.laplace_cdf(0.0, location=0.0, scale=1.0))
print(d.cauchy_pdf(0.0, location=0.0, scale=1.0))
print(d.lognormal_ppf(0.5, mu=0.0, sigma=1.0))
```

`gamma_pdf/cdf/ppf(x or p, shape, scale)` use positive gamma shape and scale; `beta_pdf/cdf/ppf(x or p, alpha, beta)` use positive beta shape parameters. CDF calculations reuse incomplete-gamma/beta numerical routines, while PPF calculations use bisection. Gamma support is nonnegative and beta support is `[0,1]`; boundary densities can be infinite. Invalid parameters raise `ValueError`.

## Descriptive statistics

The functions in `mathbr.statistics` accept nonempty finite numeric sequences. Invalid shapes, non-finite values, undefined zero-variance moments or correlations, and invalid parameters raise `ValueError`. All return floats except `five_number_summary`, which returns five floats. `ddof` defaults to 1, so variance and covariance use sample denominators; `ddof=0` uses population denominators. Quantiles use linear interpolation at index `p*(n-1)` (type 7). Weighted variance and covariance divide by the sum of positive finite weights and are population-style, with no degrees-of-freedom correction. Mode uses exact equality and chooses the smallest tied value.

| Signature | Description and parameters | Returns | Raises / limitations | Minimal example |
| --- | --- | --- | --- | --- |
| `mathbr.statistics.mean(x)` | Arithmetic mean of sequence `x`. | Float. | `ValueError` for empty or non-finite data. | `mathbr.statistics.mean([1, 2, 3])` |
| `mathbr.statistics.median(x)` | Middle value or midpoint. | Float. | Same data rules. | `mathbr.statistics.median([1, 2, 3])` |
| `mathbr.statistics.mode(x)` | Most frequent exact value; smallest on ties. | Float. | Same data rules; continuous noisy data rarely repeat exactly. | `mathbr.statistics.mode([1, 2, 2])` |
| `mathbr.statistics.variance(x, ddof=1)` | Centered sum of squares divided by `n-ddof`. | Float. | `ddof` must be integer in `[0,n)`. | `mathbr.statistics.variance([1, 2, 3])` |
| `mathbr.statistics.standard_deviation(x, ddof=1)` | Square root of variance. | Float. | Same `ddof` rule. | `mathbr.statistics.standard_deviation([1, 2, 3])` |
| `mathbr.statistics.skewness(x)` | Population standardized third central moment. | Float. | Requires positive variance; no small-sample bias correction. | `mathbr.statistics.skewness([-1, 0, 1])` |
| `mathbr.statistics.excess_kurtosis(x)` | Population standardized fourth central moment minus 3. | Float. | Requires positive variance; no small-sample bias correction. | `mathbr.statistics.excess_kurtosis([-1, 0, 1])` |
| `mathbr.statistics.quantile(x, p)` | Type-7 quantile, `p` finite in `[0,1]`. | Float. | Invalid `p` raises `ValueError`. | `mathbr.statistics.quantile([1, 2, 3], 0.5)` |
| `mathbr.statistics.percentile(x, p)` | Quantile at percent `p` finite in `[0,100]`. | Float. | Invalid `p` raises `ValueError`. | `mathbr.statistics.percentile([1, 2, 3], 50)` |
| `mathbr.statistics.interquartile_range(x)` | 75th minus 25th percentile. | Float. | Same data rules. | `mathbr.statistics.interquartile_range([1, 2, 3])` |
| `mathbr.statistics.five_number_summary(x)` | Min, Q1, median, Q3, max. | Five-float list. | Same data rules. | `mathbr.statistics.five_number_summary([1, 2, 3])` |
| `mathbr.statistics.covariance(x, y, ddof=1)` | Paired centered cross-product divided by `n-ddof`. | Float. | Equal lengths and valid `ddof` required. | `mathbr.statistics.covariance([1, 2], [2, 4])` |
| `mathbr.statistics.pearson_correlation(x, y)` | Pearson linear correlation. | Float. | Requires equal lengths and positive variance in both sequences. | `mathbr.statistics.pearson_correlation([1, 2], [2, 4])` |
| `mathbr.statistics.spearman_correlation(x, y)` | Pearson correlation of average ranks, including ties. | Float. | Requires equal lengths and variation in both rank vectors; O(n log n) sorting. | `mathbr.statistics.spearman_correlation([1, 2], [2, 4])` |
| `mathbr.statistics.weighted_mean(x, weights)` | Mean with strictly positive finite weights. | Float. | Equal lengths and finite weight sum required. | `mathbr.statistics.weighted_mean([0, 2], [1, 3])` |
| `mathbr.statistics.weighted_variance(x, weights)` | Weighted population variance. | Float. | Same weight rules; no unbiased correction. | `mathbr.statistics.weighted_variance([0, 2], [1, 3])` |
| `mathbr.statistics.weighted_covariance(x, y, weights)` | Weighted population covariance. | Float. | Aligned sequences and valid weights required. | `mathbr.statistics.weighted_covariance([0, 2], [1, 5], [1, 3])` |
| `mathbr.statistics.weighted_correlation(x, y, weights)` | Weighted Pearson correlation. | Float. | Positive weighted variance required. | `mathbr.statistics.weighted_correlation([0, 2], [1, 5], [1, 3])` |

These are descriptive, in-sample calculations. They do not by themselves provide hypothesis tests or missing-data handling.

## Hypothesis tests

`mathbr.hypothesis` provides two-sided classical t-tests. Each function returns a `TTestResult` object with read-only float attributes `statistic`, `degrees_of_freedom`, and `p_value`. Inputs must contain finite observations, with at least two observations per sample and positive relevant variance; invalid input raises `ValueError`. Exact one-sample and paired p-values assume independent Gaussian observations or Gaussian paired differences. Welch's two-sample p-value uses a Satterthwaite degrees-of-freedom approximation and assumes independent samples; it does not require equal variances. These functions do not correct for multiple testing.

| Signature | Description and parameters | Returns | Raises / limitations | Minimal example |
| --- | --- | --- | --- | --- |
| `mathbr.hypothesis.one_sample_t_test(x, null_mean=0.0)` | Test a sample mean against finite `null_mean`. | `TTestResult`. | `ValueError` for fewer than two values or zero variance. | `mathbr.hypothesis.one_sample_t_test([1, 2, 3])` |
| `mathbr.hypothesis.paired_t_test(before, after)` | Test the mean of aligned `before - after` differences against zero. | `TTestResult`. | `ValueError` for unequal lengths or zero difference variance. | `mathbr.hypothesis.paired_t_test([3, 4, 5], [1, 2, 4])` |
| `mathbr.hypothesis.welch_t_test(x, y)` | Test the difference of two independent means with unequal-variance standard error. | `TTestResult`. | `ValueError` for short or non-finite samples or zero combined variance. | `mathbr.hypothesis.welch_t_test([1, 2, 3], [2, 4, 6])` |

Multiple-comparison functions take a nonempty sequence `p_values` of finite probabilities in [0, 1] and return adjusted probabilities in the original order. Invalid values raise `ValueError`. Bonferroni controls family-wise error under arbitrary dependence; Holm is a step-down family-wise method; Benjamini-Hochberg targets false discovery rate under independence or suitable positive dependence. These adjustments do not repair invalid underlying p-values.

| Signature | Description and parameters | Returns | Raises / limitations | Minimal example |
| --- | --- | --- | --- | --- |
| `mathbr.hypothesis.bonferroni_correction(p_values)` | Multiply each p-value by the number of tests, capped at one. | List of floats. | `ValueError` for empty or invalid probabilities. | `mathbr.hypothesis.bonferroni_correction([0.01, 0.04])` |
| `mathbr.hypothesis.holm_correction(p_values)` | Holm step-down adjusted probabilities. | List of floats. | Same input rules. | `mathbr.hypothesis.holm_correction([0.01, 0.04])` |
| `mathbr.hypothesis.benjamini_hochberg_correction(p_values)` | Benjamini-Hochberg FDR adjusted probabilities. | List of floats. | Same input rules; dependence assumptions apply. | `mathbr.hypothesis.benjamini_hochberg_correction([0.01, 0.04])` |

`proportion_z_test(successes, trials, null_p)` tests a single binomial proportion against a null probability. Its `ZTestResult` has read-only float `statistic` and two-sided `p_value`. Counts must be integers with `0 <= successes <= trials`, `trials > 0`, and finite `null_p` in (0, 1). It requires at least five expected successes and failures under the null; otherwise it raises `ValueError`. The normal approximation is still an approximation, not an exact binomial test. Example: `mathbr.hypothesis.proportion_z_test(60, 100, 0.5)`.

`chi_square_goodness_of_fit(observed, expected)` compares aligned nonnegative observed bin counts with strictly positive expected counts. The totals must match, and there must be at least two bins. It returns a `ChiSquareTestResult` with read-only float `statistic`, `degrees_of_freedom`, and `p_value`. Invalid counts raise `ValueError`. This uses `k-1` degrees of freedom and assumes no parameters were estimated from the data; small expected bin counts can make the chi-squared approximation unreliable. Example: `mathbr.hypothesis.chi_square_goodness_of_fit([10, 20, 30], [20, 20, 20])`.

```python
result = mathbr.hypothesis.one_sample_t_test([1.0, 2.0, 3.0])
print(result.statistic, result.degrees_of_freedom, result.p_value)
```

## Regression and classification

### `mathbr.LinearRegression(n_features)`

Full-batch gradient descent minimizes MSE. `n_features: int` must be positive or construction raises `ValueError`. `fit(X, y, lr=0.01, epochs=1000)` accepts a nonempty matrix of rows of length `n_features`, same-length numeric targets, positive finite learning rate, and positive integer epochs; violations raise `ValueError`. It returns `None`.

`predict(x)` returns one float; `predict_batch(X)` returns a list of floats. Rows with the wrong width raise `ValueError`. `get_weights()` returns a list of slopes, `get_bias()` a float, and `trained()` a bool. Predictions and initial zero coefficients are available before fitting. No statistical inference is provided; feature scale affects gradient-descent convergence.

```python
model = mathbr.LinearRegression(1)
model.fit([[0.0], [1.0]], [1.0, 3.0])
print(model.predict([2.0]), model.get_weights())
```

### `mathbr.LogisticRegression(n_features)`

Full-batch gradient descent minimizes binary log loss. Constructor and `fit(X, y, lr=0.01, epochs=1000)` constraints match `LinearRegression`, except targets must be integer 0 or 1; invalid values raise `ValueError`. `fit` returns `None`.

`predict_proba(x)` returns a float probability, `predict_proba_batch(X)` a list of probabilities, and `predict(x, threshold=0.5)` an integer 0 or 1. The threshold must be finite and in [0, 1]. Wrong-width rows or invalid thresholds raise `ValueError`. `get_weights()` returns slopes, `get_bias()` the intercept, and `trained()` a bool. Initial zero coefficients are usable before fitting. This is a classifier without standard errors, calibration checks, or classification metrics.

```python
model = mathbr.LogisticRegression(1)
model.fit([[-1.0], [1.0]], [0, 1])
print(model.predict_proba([0.5]), model.predict([0.5]))
```

## OLS and WLS

### `mathbr.OLS(n_features)`

Householder QR least squares with an automatic intercept. `n_features: int` must be positive. `fit(X, y)` takes finite numeric rows of that width and targets, requires more observations than coefficients and full column rank, and returns `None`. Invalid shapes, non-finite input, or rank deficiency raise `ValueError`.

`predict(x)` returns a float and `predict_batch(X)` a list; invalid row width raises `ValueError`. `coefficients()` returns `[intercept, slopes...]`; `standard_errors()` returns classical standard errors in the same order. `t_statistics()` and `p_values()` return matching lists; p-values are two-sided Student-t values using `degrees_of_freedom()`. `confidence_intervals(level=0.95)` returns one `[lower, upper]` pair per coefficient; `level` must be finite and in (0, 1). `r_squared()`, `adjusted_r_squared()`, `f_statistic()`, and `residual_variance()` return floats; `degrees_of_freedom()` returns an integer and `trained()` a bool. Invalid interval level or zero/non-finite standard errors raise `ValueError`; zero residual variance makes the F-statistic undefined and raises `ValueError`. Fitted-result methods raise `RuntimeError` before `fit`. The t and F reference distributions are exact under independent Gaussian, homoscedastic errors and a fixed full-rank design; robust inference is absent.

```python
model = mathbr.OLS(1)
model.fit([[0.0], [1.0], [2.0], [3.0]], [1.0, 3.0, 5.0, 7.0])
print(model.coefficients(), model.r_squared())
```

### `mathbr.WLS(n_features)`

Weighted QR least squares with an intercept. Constructor and inherited results match `OLS`. `fit(X, y, weights)` takes one strictly positive finite weight per observation; bad weights raise `ValueError`. It returns `None`. All OLS prediction and result methods above, including t-statistics, p-values, confidence intervals, adjusted R-squared, and F-statistic, are inherited. Classical inference additionally assumes weights proportional to inverse error variance and independent Gaussian errors; the reported R-squared and residual variance are weighted.

```python
model = mathbr.WLS(1)
model.fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 4.0], [1.0, 1.0, 0.1])
print(model.coefficients())
```

## Regularized regression

`mathbr.Ridge(n_features, alpha=1.0, max_iter=1000, tol=1e-8)` and `mathbr.Lasso(n_features, alpha=1.0, max_iter=1000, tol=1e-8)` apply L2 and L1 penalties respectively. `mathbr.ElasticNet(n_features, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-8)` mixes them. `n_features` and `max_iter` must be positive integers, `alpha` finite and nonnegative, `tol` finite and positive, and `l1_ratio` finite in [0, 1]. Invalid parameters raise `ValueError`.

All three inherit `fit(X, y)` (returns `None`), `predict(x)` (float), `predict_batch(X)` (list of floats), `coefficients()` (list of slopes), `intercept()` (float), `iterations()` (int), `converged()` (bool), and `trained()` (bool). Fit needs nonempty, finite data with matching rows and feature width; invalid input raises `ValueError`. Results requested before fit raise `RuntimeError`. The objective is `SSE/(2n) + alpha * (l1_ratio * L1 + (1-l1_ratio) * L2/2)`; the intercept is unpenalized. Features are not standardized, so scale affects penalty strength. There are no standard errors; check `converged()`.

```python
model = mathbr.ElasticNet(1, alpha=0.1, l1_ratio=0.5)
model.fit([[-1.0], [0.0], [1.0]], [-2.0, 0.0, 2.0])
print(model.coefficients(), model.converged())
```

## Econometrics

### `mathbr.IV2SLS(n_exog, n_endog, n_instruments)`

Two-stage least squares with excluded instruments. Integer dimensions set the exogenous, endogenous, and excluded-instrument widths; the number of instruments must be sufficient to identify the endogenous regressors. `fit(exog, endog, instruments, y)` takes aligned finite matrices and targets and returns `None`. Invalid dimensions, rank-deficient stages, or non-finite data raise `ValueError`.

`predict(exog, endog)` returns a float; `coefficients()` returns `[intercept, exogenous slopes..., endogenous slopes...]`; `first_stage_r_squared()` returns one float per endogenous regressor; `trained()` returns a bool. Invalid prediction dimensions raise `ValueError`, and fitted results before fit raise `RuntimeError`. Instrument relevance and exclusion are assumed, not verified. First-stage R-squared is not a weak-instrument test; valid 2SLS standard errors are not provided.

```python
model = mathbr.IV2SLS(0, 1, 1)
model.fit([[], [], [], [], []], [[0], [2], [1], [4], [3]],
          [[0], [1], [2], [3], [4]], [1, 5, 3, 9, 7])
print(model.coefficients())
```

### `mathbr.FixedEffects(n_features)`

Entity-demeaned linear regression. `n_features` must be positive. `fit(X, y, entity_ids)` takes aligned finite observations and integer IDs and returns `None`; invalid shapes or non-identifiable within-entity features raise `ValueError`. `predict(x, entity_id)` returns a float for an entity seen in training. `coefficients()` returns within slopes, `entity_intercept(entity_id)` returns that entity's float intercept, `within_r_squared()` returns a float, and `trained()` a bool. Unknown entities or wrong-width rows raise `ValueError`; fitted results before fit raise `RuntimeError`. There are no time effects or clustered standard errors.

```python
model = mathbr.FixedEffects(1)
model.fit([[0], [1], [2], [0], [1], [2]],
          [1, 3, 5, 10, 12, 14], [10, 10, 10, 20, 20, 20])
print(model.coefficients(), model.entity_intercept(20))
```

## Time series

`mathbr.AR(lags)` fits a univariate autoregression with an intercept. `mathbr.VAR(n_series, lags)` fits one OLS equation per series. Constructor dimensions and `forecast(steps)` require positive integers; invalid inputs raise `ValueError`. `AR.fit(observations)` takes a finite numeric sequence; `VAR.fit(observations)` takes finite rows of width `n_series`. Both require enough observations and full-rank lag matrices; violations raise `ValueError`; fit returns `None`.

`AR.coefficients()` returns `[intercept, lag slopes...]`; `AR.forecast(steps)` returns a list of floats. `VAR.coefficients()` returns one coefficient list per equation, ordered by lag and series; `VAR.forecast(steps)` returns one row per future step; `VAR.residual_covariance()` returns a square matrix. Both expose `trained()` as bool. Fitted results before fit raise `RuntimeError`. Neither selects lag order, tests stationarity, or provides forecast intervals.

```python
ar = mathbr.AR(1)
ar.fit([1, 2, 4, 8, 16, 32])
print(ar.forecast(2))
```

`mathbr.statistics.weighted_quantile(x, weights, p)` returns the smallest sorted observation where cumulative positive weight reaches fraction `p` of total weight. It is a step-function quantile, with `p` in `[0,1]`. `mathbr.statistics.log_sum_exp(x)` computes the log of summed exponentials using max shifting, requiring a nonempty finite vector.

## Nonparametric estimation

`mathbr.nonparametric.empirical_cdf(sample, points)` returns the fraction of sample values at or below each point; it sorts once and evaluates queries by binary search. `gaussian_kde(sample, points, bandwidth)` evaluates a Gaussian kernel density. `nadaraya_watson(x, y, points, bandwidth)` returns Gaussian-kernel weighted predictions, shifting log weights for stability. Inputs must be finite and aligned where applicable; samples must be nonempty and bandwidth positive. The caller selects bandwidth. KDE and kernel regression take O(sample size × number of points) time.

## Regression diagnostics

`mathbr.diagnostics.aic(log_likelihood, n_parameters)`, `bic(log_likelihood, n_parameters, n_observations)`, and `hqic(log_likelihood, n_parameters, n_observations)` compute common likelihood-based information criteria; the caller supplies a maximized finite log likelihood and nonnegative parameter count. BIC needs positive sample size and HQIC at least three observations. `residual_standard_error(residuals, n_parameters)` uses squared residuals divided by `n-k` and needs positive residual degrees of freedom. `durbin_watson(residuals)` uses ordered residuals and needs a nonzero squared sum. `jarque_bera_statistic(residuals)` returns the skewness/kurtosis statistic and `jarque_bera_p_value(residuals)` its asymptotic chi-square(2) tail probability. Jarque-Bera needs at least three finite, nonconstant residuals. Invalid inputs raise `ValueError`.

## Binary classification evaluation

`mathbr.evaluation.confusion_matrix(y_true, y_pred)` returns `[[TN, FP], [FN, TP]]`. `precision`, `recall`, and `f1_score` take the same aligned binary-label inputs and return a float; undefined denominators yield zero. Inputs must be nonempty and labels must be 0 or 1. `roc_curve(y_true, scores)` returns `RocCurve` with read-only `fpr`, `tpr`, and `thresholds` lists. The initial ROC point is `(0, 0)` with an infinite threshold. `roc_auc(y_true, scores)` returns trapezoidal area. `precision_recall_curve(y_true, scores)` returns `PrecisionRecallCurve` with read-only `precision`, `recall`, and `thresholds` lists. Thresholds descend and equal scores are grouped. Both classes and finite scores are required; invalid input raises `ValueError`. Curves have one point per unique score, except for the ROC initial point. Sorting dominates runtime at O(n log n).

```python
e = mathbr.evaluation
labels = [0, 0, 1, 1]
scores = [0.1, 0.4, 0.35, 0.8]
print(e.roc_auc(labels, scores))  # 0.75
print(e.confusion_matrix(labels, [0, 0, 0, 1]))
```

`rmse(y_true, y_pred)`, `mae(y_true, y_pred)`, and `mape(y_true, y_pred)` are regression evaluation metrics on nonempty aligned finite vectors. MAPE is a fraction and rejects zero targets. `log_loss(y_true, probabilities)` uses binary labels and probabilities in `[0,1]`; confidently wrong probabilities at the boundary yield infinity. `k_fold_indices(n_samples, n_splits, seed=0)` returns reproducibly shuffled validation index folds, balancing sizes within one sample. `calibration_curve(y_true, probabilities, n_bins=10)` returns a `CalibrationCurve` with read-only `mean_predicted`, `fraction_positive`, and `counts` arrays for nonempty equal-width bins.

## Time-series diagnostics

`mathbr.time_series_diagnostics.acf(x, max_lag)` returns biased sample autocorrelations from lag 0 through `max_lag`. `pacf(x, max_lag)` uses Durbin–Levinson recursion on those autocorrelations. `ljung_box(x, lags)` returns `LjungBoxResult` with read-only `statistic`, `p_value`, and `lags` attributes; its chi-square reference uses `lags` degrees of freedom, with no fitted-parameter correction. Inputs must be finite, nonconstant, and have more observations than the requested maximum lag. ACF is direct O(n × lags), while PACF adds O(lags²) work.

## Conjugate Bayesian updating

`mathbr.bayesian.beta_binomial_update(alpha, beta, successes, trials)` returns `BetaPosterior` with read-only `alpha` and `beta`, a `mean()` method, and `credible_interval(level=0.95)` equal-tailed beta posterior quantiles. The shapes must be positive and `0 <= successes <= trials`. `normal_normal_update(prior_mean, prior_sd, observation_sd, observations)` returns `NormalPosterior` with read-only `mean` and `standard_deviation` plus `credible_interval(level=0.95)`. It assumes independent normal observations with known observation standard deviation; prior and observation standard deviations must be positive, observations nonempty and finite. Credible levels must lie in `(0,1)`.

## Survival analysis

`mathbr.survival.kaplan_meier(times, events)` estimates the right-censored Kaplan–Meier curve and returns `KaplanMeierResult` with read-only `times`, `survival`, `at_risk`, and `events` arrays at all unique observed times. `events=1` means the event occurred; `0` means right censored. `log_rank_test(times, events, groups)` compares binary groups and returns `LogRankResult` with read-only `statistic` and asymptotic `p_value`. Times must be nonnegative and finite, and inputs aligned. The log-rank test requires both groups and positive variance. These routines assume independent censoring, no delayed entry, and no competing risks.

## Volatility

`mathbr.GARCH(max_iter=2000, tol=1e-7, arch_only=False)` estimates GARCH(1,1) by Gaussian quasi-likelihood; `arch_only=True` fixes beta to zero. `mathbr.ARCH(max_iter=2000, tol=1e-7)` is the ARCH(1) subclass. `max_iter` must be positive, `tol` finite and positive. `fit(observations)` needs at least 20 finite observations with positive variance; invalid input raises `ValueError` and fit returns `None`.

Both expose `mean()`, `omega()`, `alpha()`, `beta()`, and `log_likelihood()` as floats; `conditional_variance()` as a list of floats; `forecast_variance(steps)` as a list of floats for positive integer steps; and `converged()` and `trained()` as bools. Fitted results before fit raise `RuntimeError`; invalid forecast horizon raises `ValueError`. The constant-mean, symmetric Gaussian model has no standard errors or non-Gaussian innovations. Check convergence before interpreting parameters.

```python
model = mathbr.GARCH()
model.fit([-2.0, -1.0, 0.5, 1.0, 2.0] * 20)
print(model.forecast_variance(3), model.converged())
```

The module also exposes `mathbr.version`, currently the string `"0.6.0"`. `_RegularizedRegression` is a binding base class for the three regularized models; it has no public constructor.
