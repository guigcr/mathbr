#include "mathbr/ols.hpp"
#include "mathbr/distributions.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mathbr {

OLS::OLS(int n_features) : n_features_(n_features) {
    if (n_features <= 0) throw std::invalid_argument("n_features must be positive");
}

void OLS::require_trained() const {
    if (!trained_) throw std::logic_error("fit must be called first");
}

void OLS::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    fit_weighted(X, y, std::vector<double>(X.size(), 1.0));
}

void OLS::fit_weighted(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
                       const std::vector<double>& weights) {
    const size_t n = X.size();
    const size_t p = static_cast<size_t>(n_features_) + 1;
    if (n != y.size() || n != weights.size() || n <= p) {
        throw std::invalid_argument("OLS requires matching X and y with more rows than parameters");
    }
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(weights[i]) || weights[i] <= 0.0)
            throw std::invalid_argument("weights must be finite and positive");
        if (X[i].size() != static_cast<size_t>(n_features_))
            throw std::invalid_argument("each X row must match n_features");
        if (!std::isfinite(y[i])) throw std::invalid_argument("y must contain finite values");
        for (double value : X[i])
            if (!std::isfinite(value)) throw std::invalid_argument("X must contain finite values");
    }

    std::vector<double> sqrt_weights(n);
    for (size_t i = 0; i < n; ++i) sqrt_weights[i] = std::sqrt(weights[i]);

    // Column-major Householder QR applies one stable reflection per feature.
    // The transformed response is Q'Wy, so Q itself is never materialized.
    std::vector<std::vector<double>> columns(p, std::vector<double>(n));
    std::vector<double> original_norms(p, 0.0);
    for (size_t j = 0; j < p; ++j) {
        double squared_norm = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double value = sqrt_weights[i] * (j == 0 ? 1.0 : X[i][j - 1]);
            columns[j][i] = value;
            squared_norm += value * value;
        }
        original_norms[j] = std::sqrt(squared_norm);
    }
    std::vector<double> transformed_y(n);
    for (size_t i = 0; i < n; ++i) transformed_y[i] = sqrt_weights[i] * y[i];
    std::vector<double> r(p * p, 0.0);
    for (size_t j = 0; j < p; ++j) {
        auto& column = columns[j];
        double squared_norm = 0.0;
        for (size_t i = j; i < n; ++i) squared_norm += column[i] * column[i];
        const double norm = std::sqrt(squared_norm);
        if (!std::isfinite(original_norms[j]) || !std::isfinite(norm) ||
            norm <= 1e-12 * original_norms[j]) {
            throw std::invalid_argument("design matrix has linearly dependent columns");
        }
        const double diagonal = -std::copysign(norm, column[j]);
        column[j] -= diagonal;
        double reflector_norm = 0.0;
        for (size_t i = j; i < n; ++i) reflector_norm += column[i] * column[i];
        const double tau = 2.0 / reflector_norm;
        for (size_t k = j + 1; k < p; ++k) {
            double projection = 0.0;
            for (size_t i = j; i < n; ++i) projection += column[i] * columns[k][i];
            projection *= tau;
            for (size_t i = j; i < n; ++i) columns[k][i] -= projection * column[i];
            r[j * p + k] = columns[k][j];
        }
        double response_projection = 0.0;
        for (size_t i = j; i < n; ++i) response_projection += column[i] * transformed_y[i];
        response_projection *= tau;
        for (size_t i = j; i < n; ++i) transformed_y[i] -= response_projection * column[i];
        r[j * p + j] = diagonal;
    }

    std::vector<double> beta(p);
    for (size_t j = p; j-- > 0;) {
        double value = transformed_y[j];
        for (size_t k = j + 1; k < p; ++k) value -= r[j * p + k] * beta[k];
        beta[j] = value / r[j * p + j];
    }

    double weighted_sum = 0.0, total_weight = 0.0;
    for (size_t i = 0; i < n; ++i) {
        weighted_sum += weights[i] * y[i];
        total_weight += weights[i];
    }
    const double mean = weighted_sum / total_weight;
    double sse = 0.0, sst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double fitted = beta[0];
        for (size_t j = 1; j < p; ++j) fitted += beta[j] * X[i][j - 1];
        const double error = y[i] - fitted;
        sse += weights[i] * error * error;
        const double centered = y[i] - mean;
        sst += weights[i] * centered * centered;
    }
    const double sigma2 = sse / static_cast<double>(n - p);
    std::vector<double> errors(p);
    for (size_t j = 0; j < p; ++j) {
        // Diagonal of (X'X)^-1 = R^-1 (R^-1)' via R' u = e_j.
        std::vector<double> u(p, 0.0);
        for (size_t k = 0; k < p; ++k) {
            double value = k == j ? 1.0 : 0.0;
            for (size_t l = 0; l < k; ++l) value -= r[l * p + k] * u[l];
            u[k] = value / r[k * p + k];
        }
        errors[j] = std::sqrt(sigma2 * std::inner_product(u.begin(), u.end(), u.begin(), 0.0));
    }
    coefficients_ = std::move(beta);
    standard_errors_ = std::move(errors);
    residual_variance_ = sigma2;
    r_squared_ = sst == 0.0 ? (sse == 0.0 ? 1.0 : 0.0) : 1.0 - sse / sst;
    degrees_of_freedom_ = static_cast<int>(n - p);
    trained_ = true;
}

double OLS::predict(const std::vector<double>& x) const {
    require_trained();
    if (x.size() != static_cast<size_t>(n_features_))
        throw std::invalid_argument("x size doesn't match n_features");
    double result = coefficients_[0];
    for (size_t j = 0; j < x.size(); ++j) result += coefficients_[j + 1] * x[j];
    return result;
}

std::vector<double> OLS::predict_batch(const std::vector<std::vector<double>>& X) const {
    std::vector<double> result;
    result.reserve(X.size());
    for (const auto& row : X) result.push_back(predict(row));
    return result;
}

std::vector<double> OLS::coefficients() const { require_trained(); return coefficients_; }
std::vector<double> OLS::standard_errors() const { require_trained(); return standard_errors_; }
std::vector<double> OLS::t_statistics() const {
    require_trained();
    std::vector<double> result(coefficients_.size());
    for (size_t i = 0; i < result.size(); ++i) {
        if (!(standard_errors_[i] > 0.0) || !std::isfinite(standard_errors_[i]))
            throw std::domain_error("t-statistics require positive finite standard errors");
        result[i] = coefficients_[i] / standard_errors_[i];
    }
    return result;
}
std::vector<double> OLS::p_values() const {
    const auto t = t_statistics();
    std::vector<double> result(t.size());
    for (size_t i = 0; i < t.size(); ++i)
        result[i] = 2.0 * distributions::student_t_cdf(-std::abs(t[i]), degrees_of_freedom_);
    return result;
}
std::vector<std::vector<double>> OLS::confidence_intervals(double level) const {
    require_trained();
    if (!std::isfinite(level) || level <= 0.0 || level >= 1.0)
        throw std::invalid_argument("level must be finite and in (0, 1)");
    t_statistics();  // Reject zero or non-finite standard errors.
    const double critical = distributions::student_t_ppf(0.5 + level / 2.0,
                                                          degrees_of_freedom_);
    std::vector<std::vector<double>> result(coefficients_.size(), std::vector<double>(2));
    for (size_t i = 0; i < result.size(); ++i) {
        const double margin = critical * standard_errors_[i];
        result[i][0] = coefficients_[i] - margin;
        result[i][1] = coefficients_[i] + margin;
    }
    return result;
}
double OLS::r_squared() const { require_trained(); return r_squared_; }
double OLS::adjusted_r_squared() const {
    require_trained();
    const double n_minus_one = degrees_of_freedom_ + n_features_;
    return 1.0 - (1.0 - r_squared_) * n_minus_one / degrees_of_freedom_;
}
double OLS::f_statistic() const {
    require_trained();
    if (residual_variance_ == 0.0)
        throw std::domain_error("F-statistic is undefined with zero residual variance");
    return (r_squared_ / n_features_) * degrees_of_freedom_ / (1.0 - r_squared_);
}
double OLS::residual_variance() const { require_trained(); return residual_variance_; }
int OLS::degrees_of_freedom() const { require_trained(); return degrees_of_freedom_; }
bool OLS::trained() const noexcept { return trained_; }

}  // namespace mathbr
