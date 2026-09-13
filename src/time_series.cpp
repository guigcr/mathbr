#include "mathbr/time_series.hpp"
#include "mathbr/ols.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mathbr {

VAR::VAR(int n_series, int lags) : n_series_(n_series), lags_(lags) {
    if (n_series <= 0 || lags <= 0) throw std::invalid_argument("n_series and lags must be positive");
}
void VAR::require_trained() const {
    if (!trained_) throw std::logic_error("fit must be called first");
}
void VAR::fit(const std::vector<std::vector<double>>& observations) {
    const size_t t = observations.size();
    const size_t p = static_cast<size_t>(n_series_) * lags_;
    if (t <= static_cast<size_t>(lags_) + p + 1)
        throw std::invalid_argument("VAR needs more effective observations than coefficients");
    for (const auto& row : observations) {
        if (row.size() != static_cast<size_t>(n_series_))
            throw std::invalid_argument("each observation must match n_series");
        for (double value : row)
            if (!std::isfinite(value)) throw std::invalid_argument("observations must be finite");
    }
    const size_t n = t - lags_;
    std::vector<std::vector<double>> X(n, std::vector<double>(p));
    for (size_t i = 0; i < n; ++i)
        for (int lag = 1; lag <= lags_; ++lag)
            for (int j = 0; j < n_series_; ++j)
                X[i][(lag - 1) * n_series_ + j] = observations[i + lags_ - lag][j];
    std::vector<std::vector<double>> fitted_coefficients(n_series_);
    for (int equation = 0; equation < n_series_; ++equation) {
        std::vector<double> y(n);
        for (size_t i = 0; i < n; ++i) y[i] = observations[i + lags_][equation];
        OLS model(static_cast<int>(p));
        model.fit(X, y);
        fitted_coefficients[equation] = model.coefficients();
    }
    std::vector<std::vector<double>> cov(n_series_, std::vector<double>(n_series_, 0.0));
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> residual(n_series_);
        for (int equation = 0; equation < n_series_; ++equation) {
            double prediction = fitted_coefficients[equation][0];
            for (size_t j = 0; j < p; ++j) prediction += fitted_coefficients[equation][j + 1] * X[i][j];
            residual[equation] = observations[i + lags_][equation] - prediction;
        }
        for (int a = 0; a < n_series_; ++a)
            for (int b = 0; b < n_series_; ++b)
                cov[a][b] += residual[a] * residual[b];
    }
    for (auto& row : cov)
        for (double& value : row) value /= static_cast<double>(n - p - 1);
    coefficients_ = std::move(fitted_coefficients);
    covariance_ = std::move(cov);
    history_.assign(observations.end() - lags_, observations.end());
    trained_ = true;
}
std::vector<std::vector<double>> VAR::coefficients() const { require_trained(); return coefficients_; }
std::vector<std::vector<double>> VAR::residual_covariance() const { require_trained(); return covariance_; }
std::vector<std::vector<double>> VAR::forecast(int steps) const {
    require_trained();
    if (steps <= 0) throw std::invalid_argument("steps must be positive");
    std::vector<std::vector<double>> history = history_, result;
    result.reserve(steps);
    for (int step = 0; step < steps; ++step) {
        std::vector<double> next(n_series_);
        for (int eq = 0; eq < n_series_; ++eq) {
            next[eq] = coefficients_[eq][0];
            for (int lag = 1; lag <= lags_; ++lag)
                for (int j = 0; j < n_series_; ++j)
                    next[eq] += coefficients_[eq][1 + (lag - 1) * n_series_ + j]
                                * history[history.size() - lag][j];
        }
        history.push_back(next);
        result.push_back(std::move(next));
    }
    return result;
}
bool VAR::trained() const noexcept { return trained_; }

AR::AR(int lags) : model_(1, lags) {}
void AR::fit(const std::vector<double>& observations) {
    std::vector<std::vector<double>> rows;
    rows.reserve(observations.size());
    for (double value : observations) rows.push_back({value});
    model_.fit(rows);
}
std::vector<double> AR::coefficients() const { return model_.coefficients()[0]; }
std::vector<double> AR::forecast(int steps) const {
    auto rows = model_.forecast(steps);
    std::vector<double> result;
    result.reserve(rows.size());
    for (const auto& row : rows) result.push_back(row[0]);
    return result;
}
bool AR::trained() const noexcept { return model_.trained(); }

GARCH::GARCH(int max_iter, double tol, bool arch_only)
    : max_iter_(max_iter), tol_(tol), arch_only_(arch_only) {
    if (max_iter <= 0 || !std::isfinite(tol) || tol <= 0.0)
        throw std::invalid_argument("max_iter and tol must be positive");
}
void GARCH::require_trained() const {
    if (!trained_) throw std::logic_error("fit must be called first");
}
void GARCH::fit(const std::vector<double>& observations) {
    const size_t n = observations.size();
    if (n < 20) throw std::invalid_argument("ARCH/GARCH requires at least 20 observations");
    double sample_mean = 0.0;
    for (double value : observations) {
        if (!std::isfinite(value)) throw std::invalid_argument("observations must be finite");
        sample_mean += value;
    }
    sample_mean /= static_cast<double>(n);
    std::vector<double> squared(n);
    double sample_variance = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double error = observations[i] - sample_mean;
        squared[i] = error * error;
        sample_variance += squared[i];
    }
    sample_variance /= static_cast<double>(n);
    if (!std::isfinite(sample_variance) || sample_variance <= 0.0)
        throw std::invalid_argument("observations must have nonzero finite variance");

    auto evaluate = [&](double log_omega, double alpha, double beta, std::vector<double>* path) {
        if (alpha < 0.0 || beta < 0.0 || alpha + beta >= 0.999 ||
            (arch_only_ && beta != 0.0)) return std::numeric_limits<double>::infinity();
        const double omega = std::exp(log_omega);
        if (!std::isfinite(omega) || omega <= 0.0) return std::numeric_limits<double>::infinity();
        double previous = sample_variance, loss = 0.0;
        if (path) path->clear();
        for (size_t i = 0; i < n; ++i) {
            const double h = i == 0 ? previous : omega + alpha * squared[i - 1] + beta * previous;
            if (!std::isfinite(h) || h <= 0.0) return std::numeric_limits<double>::infinity();
            loss += std::log(h) + squared[i] / h;
            if (path) path->push_back(h);
            previous = h;
        }
        return 0.5 * (static_cast<double>(n) * std::log(2.0 * 3.14159265358979323846) + loss);
    };
    std::array<double, 3> parameter{std::log(sample_variance * (arch_only_ ? 0.8 : 0.1)),
                                    arch_only_ ? 0.2 : 0.1, arch_only_ ? 0.0 : 0.8};
    std::array<double, 3> step{0.5, 0.1, 0.1};
    double best = evaluate(parameter[0], parameter[1], parameter[2], nullptr);
    bool did_converge = false;
    for (int iteration = 0; iteration < max_iter_; ++iteration) {
        bool improved = false;
        for (int axis = 0; axis < (arch_only_ ? 2 : 3); ++axis) {
            for (double sign : {-1.0, 1.0}) {
                auto candidate = parameter;
                candidate[axis] += sign * step[axis];
                const double loss = evaluate(candidate[0], candidate[1], candidate[2], nullptr);
                if (loss < best) {
                    best = loss;
                    parameter = candidate;
                    improved = true;
                }
            }
        }
        if (!improved) {
            for (double& size : step) size *= 0.5;
            if (std::max({step[0], step[1], arch_only_ ? 0.0 : step[2]}) < tol_) {
                did_converge = true;
                break;
            }
        }
    }
    std::vector<double> path;
    evaluate(parameter[0], parameter[1], parameter[2], &path);
    mean_ = sample_mean;
    omega_ = std::exp(parameter[0]);
    alpha_ = parameter[1];
    beta_ = parameter[2];
    log_likelihood_ = -best;
    last_error_squared_ = squared.back();
    variance_ = std::move(path);
    converged_ = did_converge;
    trained_ = true;
}
double GARCH::mean() const { require_trained(); return mean_; }
double GARCH::omega() const { require_trained(); return omega_; }
double GARCH::alpha() const { require_trained(); return alpha_; }
double GARCH::beta() const { require_trained(); return beta_; }
double GARCH::log_likelihood() const { require_trained(); return log_likelihood_; }
std::vector<double> GARCH::conditional_variance() const { require_trained(); return variance_; }
std::vector<double> GARCH::forecast_variance(int steps) const {
    require_trained();
    if (steps <= 0) throw std::invalid_argument("steps must be positive");
    std::vector<double> result;
    result.reserve(steps);
    double next = omega_ + alpha_ * last_error_squared_ + beta_ * variance_.back();
    for (int step = 0; step < steps; ++step) {
        result.push_back(next);
        next = omega_ + (alpha_ + beta_) * next;
    }
    return result;
}
bool GARCH::converged() const { require_trained(); return converged_; }
bool GARCH::trained() const noexcept { return trained_; }

}  // namespace mathbr
