#include "mathbr/multivariate.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace mathbr::multivariate {
namespace {
constexpr double log_2pi = 1.83787706640934548356;
}

MultivariateNormal::MultivariateNormal(
    const std::vector<double>& mean,
    const std::vector<std::vector<double>>& covariance)
    : mean_(mean), lower_(mean.size(), std::vector<double>(mean.size())), log_normalizer_(0.0) {
    const std::size_t p = mean.size();
    if (p == 0 || covariance.size() != p)
        throw std::invalid_argument("mean and covariance must have matching positive dimensions");
    for (double value : mean)
        if (!std::isfinite(value)) throw std::invalid_argument("mean must be finite");
    for (const auto& row : covariance) {
        if (row.size() != p) throw std::invalid_argument("covariance must be square");
        for (double value : row)
            if (!std::isfinite(value)) throw std::invalid_argument("covariance must be finite");
    }
    for (std::size_t i = 0; i < p; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            const double a = covariance[i][j], b = covariance[j][i];
            if (std::abs(a - b) > 1e-12 * std::max({1.0, std::abs(a), std::abs(b)}))
                throw std::invalid_argument("covariance must be symmetric");
            double pivot = a + 0.5 * (b - a);
            for (std::size_t k = 0; k < j; ++k) pivot -= lower_[i][k] * lower_[j][k];
            if (i == j) {
                if (!(pivot > 0.0) || !std::isfinite(pivot))
                    throw std::invalid_argument("covariance must be positive definite");
                lower_[i][j] = std::sqrt(pivot);
                log_normalizer_ += std::log(lower_[i][j]);
            } else {
                lower_[i][j] = pivot / lower_[j][j];
                if (!std::isfinite(lower_[i][j]))
                    throw std::invalid_argument("covariance factor is not finite");
            }
        }
    }
    log_normalizer_ += 0.5 * p * log_2pi;
}

double MultivariateNormal::squared_distance(const std::vector<double>& x) const {
    std::vector<double> scratch(mean_.size());
    return squared_distance(x, scratch);
}

double MultivariateNormal::squared_distance(const std::vector<double>& x,
                                            std::vector<double>& solved) const {
    const std::size_t p = mean_.size();
    if (x.size() != p) throw std::invalid_argument("observation has wrong dimension");
    double squared = 0.0;
    for (std::size_t i = 0; i < p; ++i) {
        if (!std::isfinite(x[i])) throw std::invalid_argument("observation must be finite");
        double residual = x[i] - mean_[i];
        for (std::size_t j = 0; j < i; ++j) residual -= lower_[i][j] * solved[j];
        solved[i] = residual / lower_[i][i];
        squared += solved[i] * solved[i];
    }
    return squared;
}

double MultivariateNormal::logpdf(const std::vector<double>& x) const {
    return -log_normalizer_ - 0.5 * squared_distance(x);
}

double MultivariateNormal::pdf(const std::vector<double>& x) const {
    return std::exp(logpdf(x));
}

double MultivariateNormal::mahalanobis_distance(const std::vector<double>& x) const {
    return std::sqrt(squared_distance(x));
}

std::vector<double> MultivariateNormal::logpdf_batch(
    const std::vector<std::vector<double>>& data) const {
    std::vector<double> result;
    result.reserve(data.size());
    std::vector<double> scratch(mean_.size());
    for (const auto& row : data)
        result.push_back(-log_normalizer_ - 0.5 * squared_distance(row, scratch));
    return result;
}

std::vector<double> MultivariateNormal::pdf_batch(
    const std::vector<std::vector<double>>& data) const {
    std::vector<double> result;
    result.reserve(data.size());
    std::vector<double> scratch(mean_.size());
    for (const auto& row : data)
        result.push_back(std::exp(-log_normalizer_ - 0.5 * squared_distance(row, scratch)));
    return result;
}

std::vector<std::vector<double>> MultivariateNormal::sample(
    std::size_t count, std::uint64_t seed) const {
    std::mt19937_64 generator(seed);
    std::normal_distribution<double> standard;
    const std::size_t p = mean_.size();
    std::vector<std::vector<double>> result(count, std::vector<double>(p));
    std::vector<double> normals(p);
    for (auto& row : result) {
        for (double& value : normals) value = standard(generator);
        for (std::size_t i = 0; i < p; ++i) {
            double value = mean_[i];
            for (std::size_t j = 0; j <= i; ++j) value += lower_[i][j] * normals[j];
            if (!std::isfinite(value)) throw std::overflow_error("sample is not finite");
            row[i] = value;
        }
    }
    return result;
}

double mahalanobis_distance(const std::vector<double>& x,
                            const std::vector<double>& mean,
                            const std::vector<std::vector<double>>& covariance) {
    return MultivariateNormal(mean, covariance).mahalanobis_distance(x);
}

}  // namespace mathbr::multivariate
