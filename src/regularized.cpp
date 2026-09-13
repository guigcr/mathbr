#include "mathbr/regularized.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mathbr {

RegularizedRegression::RegularizedRegression(int n_features, double alpha, double l1_ratio,
                                             int max_iter, double tol)
    : n_features_(n_features), alpha_(alpha), l1_ratio_(l1_ratio),
      max_iter_(max_iter), tol_(tol) {
    if (n_features <= 0 || !std::isfinite(alpha) || alpha < 0.0 ||
        !std::isfinite(l1_ratio) || l1_ratio < 0.0 || l1_ratio > 1.0 ||
        max_iter <= 0 || !std::isfinite(tol) || tol <= 0.0)
        throw std::invalid_argument("invalid regularization or optimization parameters");
}

void RegularizedRegression::require_trained() const {
    if (!trained_) throw std::logic_error("fit must be called first");
}

void RegularizedRegression::fit(const std::vector<std::vector<double>>& X,
                                const std::vector<double>& y) {
    const size_t n = X.size(), p = static_cast<size_t>(n_features_);
    if (n == 0 || n != y.size())
        throw std::invalid_argument("X and y must have the same nonzero length");
    std::vector<double> means(p, 0.0);
    double mean_y = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (X[i].size() != p) throw std::invalid_argument("each X row must match n_features");
        if (!std::isfinite(y[i])) throw std::invalid_argument("y must contain finite values");
        mean_y += y[i];
        for (size_t j = 0; j < p; ++j) {
            if (!std::isfinite(X[i][j])) throw std::invalid_argument("X must contain finite values");
            means[j] += X[i][j];
        }
    }
    mean_y /= static_cast<double>(n);
    for (double& value : means) value /= static_cast<double>(n);

    std::vector<std::vector<double>> columns(p, std::vector<double>(n));
    std::vector<double> squared_norms(p, 0.0);
    for (size_t j = 0; j < p; ++j) {
        for (size_t i = 0; i < n; ++i) {
            columns[j][i] = X[i][j] - means[j];
            squared_norms[j] += columns[j][i] * columns[j][i];
        }
        squared_norms[j] /= static_cast<double>(n);
    }
    std::vector<double> beta(p, 0.0), residual(n);
    for (size_t i = 0; i < n; ++i) residual[i] = y[i] - mean_y;
    bool did_converge = false;
    int used_iterations = 0;
    for (int iteration = 0; iteration < max_iter_; ++iteration) {
        double max_change = 0.0, max_coefficient = 0.0;
        for (size_t j = 0; j < p; ++j) {
            const double old = beta[j];
            double numerator = 0.0;
            for (size_t i = 0; i < n; ++i)
                numerator += columns[j][i] * (residual[i] + columns[j][i] * old);
            numerator /= static_cast<double>(n);
            const double shrink = alpha_ * l1_ratio_;
            const double denominator = squared_norms[j] + alpha_ * (1.0 - l1_ratio_);
            beta[j] = denominator == 0.0 ? 0.0 :
                std::copysign(std::max(std::abs(numerator) - shrink, 0.0), numerator) / denominator;
            const double change = beta[j] - old;
            for (size_t i = 0; i < n; ++i) residual[i] -= columns[j][i] * change;
            max_change = std::max(max_change, std::abs(change));
            max_coefficient = std::max(max_coefficient, std::abs(beta[j]));
        }
        used_iterations = iteration + 1;
        if (max_change <= tol_ * std::max(1.0, max_coefficient)) {
            did_converge = true;
            break;
        }
    }
    double bias = mean_y;
    for (size_t j = 0; j < p; ++j) bias -= means[j] * beta[j];
    coefficients_ = std::move(beta);
    intercept_ = bias;
    iterations_ = used_iterations;
    converged_ = did_converge;
    trained_ = true;
}

double RegularizedRegression::predict(const std::vector<double>& x) const {
    require_trained();
    if (x.size() != static_cast<size_t>(n_features_))
        throw std::invalid_argument("x size doesn't match n_features");
    double result = intercept_;
    for (size_t j = 0; j < x.size(); ++j) result += coefficients_[j] * x[j];
    return result;
}

std::vector<double> RegularizedRegression::predict_batch(const std::vector<std::vector<double>>& X) const {
    std::vector<double> result;
    result.reserve(X.size());
    for (const auto& row : X) result.push_back(predict(row));
    return result;
}

std::vector<double> RegularizedRegression::coefficients() const { require_trained(); return coefficients_; }
double RegularizedRegression::intercept() const { require_trained(); return intercept_; }
int RegularizedRegression::iterations() const { require_trained(); return iterations_; }
bool RegularizedRegression::converged() const { require_trained(); return converged_; }
bool RegularizedRegression::trained() const noexcept { return trained_; }

}  // namespace mathbr
