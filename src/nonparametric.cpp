#include "mathbr/nonparametric.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mathbr::nonparametric {
namespace {
constexpr double inv_sqrt_2pi = 0.39894228040143267794;
void validate(const std::vector<double>& values, bool allow_empty = false) {
    if (!allow_empty && values.empty()) throw std::invalid_argument("sample must be nonempty");
    for (double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("values must be finite");
}
void validate_bandwidth(double bandwidth) {
    if (!std::isfinite(bandwidth) || bandwidth <= 0)
        throw std::invalid_argument("bandwidth must be finite and positive");
}
}  // namespace

std::vector<double> empirical_cdf(const std::vector<double>& sample,
                                  const std::vector<double>& points) {
    validate(sample);
    validate(points, true);
    auto sorted = sample;
    std::sort(sorted.begin(), sorted.end());
    std::vector<double> result;
    result.reserve(points.size());
    for (double point : points)
        result.push_back(static_cast<double>(std::upper_bound(sorted.begin(), sorted.end(), point)
                                             - sorted.begin()) / sorted.size());
    return result;
}

std::vector<double> gaussian_kde(const std::vector<double>& sample,
                                 const std::vector<double>& points, double bandwidth) {
    validate(sample);
    validate(points, true);
    validate_bandwidth(bandwidth);
    std::vector<double> result;
    result.reserve(points.size());
    for (double point : points) {
        double sum = 0;
        for (double observation : sample) {
            const double z = (point - observation) / bandwidth;
            sum += std::exp(-0.5 * z * z);
        }
        result.push_back(inv_sqrt_2pi * sum / (sample.size() * bandwidth));
    }
    return result;
}

std::vector<double> nadaraya_watson(const std::vector<double>& x,
                                    const std::vector<double>& y,
                                    const std::vector<double>& points,
                                    double bandwidth) {
    validate(x);
    validate(y);
    validate(points, true);
    validate_bandwidth(bandwidth);
    if (x.size() != y.size()) throw std::invalid_argument("x and y lengths must match");
    std::vector<double> result;
    result.reserve(points.size());
    for (double point : points) {
        double weighted_sum = 0, weight_sum = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double z = (point - x[i]) / bandwidth;
            const double weight = std::exp(-0.5 * z * z);
            weighted_sum += weight * y[i];
            weight_sum += weight;
        }
        if (weight_sum > 1e-250) {
            result.push_back(weighted_sum / weight_sum);
            continue;
        }
        // Extremely distant queries need max shifting to avoid losing all
        // relative weights to underflow. This path is normally uncommon.
        double best_log_weight = -std::numeric_limits<double>::infinity();
        weighted_sum = 0;
        weight_sum = 0;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const double z = (point - x[i]) / bandwidth;
            const double log_weight = -0.5 * z * z;
            if (!std::isfinite(log_weight)) continue;
            if (log_weight > best_log_weight) {
                const double scale = std::exp(best_log_weight - log_weight);
                weighted_sum = weighted_sum * scale + y[i];
                weight_sum = weight_sum * scale + 1;
                best_log_weight = log_weight;
            } else {
                const double weight = std::exp(log_weight - best_log_weight);
                weighted_sum += weight * y[i];
                weight_sum += weight;
            }
        }
        if (weight_sum == 0)
            throw std::invalid_argument("point is too far from the sample at this bandwidth");
        result.push_back(weighted_sum / weight_sum);
    }
    return result;
}
}  // namespace mathbr::nonparametric
