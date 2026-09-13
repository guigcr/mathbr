#include "mathbr/ols.hpp"

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

    // Modified Gram-Schmidt with a second orthogonalization pass.
    std::vector<std::vector<double>> q(p, std::vector<double>(n));
    std::vector<double> r(p * p, 0.0);
    for (size_t j = 0; j < p; ++j) {
        std::vector<double> column(n);
        for (size_t i = 0; i < n; ++i)
            column[i] = std::sqrt(weights[i]) * (j == 0 ? 1.0 : X[i][j - 1]);
        const double original_norm = std::sqrt(std::inner_product(column.begin(), column.end(), column.begin(), 0.0));
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t k = 0; k < j; ++k) {
                const double projection = std::inner_product(q[k].begin(), q[k].end(), column.begin(), 0.0);
                r[k * p + j] += projection;
                for (size_t i = 0; i < n; ++i) column[i] -= projection * q[k][i];
            }
        }
        const double norm = std::sqrt(std::inner_product(column.begin(), column.end(), column.begin(), 0.0));
        if (!std::isfinite(original_norm) || !std::isfinite(norm) ||
            norm <= 1e-12 * original_norm) {
            throw std::invalid_argument("design matrix has linearly dependent columns");
        }
        r[j * p + j] = norm;
        for (size_t i = 0; i < n; ++i) q[j][i] = column[i] / norm;
    }

    std::vector<double> qty(p), beta(p);
    for (size_t j = 0; j < p; ++j)
        for (size_t i = 0; i < n; ++i) qty[j] += q[j][i] * std::sqrt(weights[i]) * y[i];
    for (size_t j = p; j-- > 0;) {
        double value = qty[j];
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
double OLS::r_squared() const { require_trained(); return r_squared_; }
double OLS::residual_variance() const { require_trained(); return residual_variance_; }
int OLS::degrees_of_freedom() const { require_trained(); return degrees_of_freedom_; }
bool OLS::trained() const noexcept { return trained_; }

}  // namespace mathbr
