#include "mathbr/hypothesis.hpp"
#include "mathbr/distributions.hpp"
#include "mathbr/statistics.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace mathbr::hypothesis {
namespace {
TTestResult finish(double statistic, double df) {
    if (!std::isfinite(statistic) || !std::isfinite(df) || df <= 0.0)
        throw std::invalid_argument("t-test requires finite statistic and positive degrees of freedom");
    return {statistic, df, 2.0 * distributions::student_t_cdf(-std::abs(statistic), df)};
}
void validate_p_values(const std::vector<double>& p_values) {
    if (p_values.empty()) throw std::invalid_argument("p_values must be nonempty");
    for (double p : p_values)
        if (!std::isfinite(p) || p < 0.0 || p > 1.0)
            throw std::invalid_argument("p_values must be finite and in [0, 1]");
}

std::vector<size_t> sorted_indices(const std::vector<double>& p_values) {
    std::vector<size_t> order(p_values.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return p_values[a] < p_values[b]; });
    return order;
}
}  // namespace

TTestResult one_sample_t_test(const std::vector<double>& x, double null_mean) {
    if (!std::isfinite(null_mean)) throw std::invalid_argument("null_mean must be finite");
    if (x.size() < 2) throw std::invalid_argument("t-test requires at least two observations");
    const double center = statistics::mean(x);
    const double sample_variance = statistics::variance(x);
    if (!(sample_variance > 0.0))
        throw std::invalid_argument("t-test requires positive sample variance");
    return finish((center - null_mean) / std::sqrt(sample_variance / x.size()),
                  static_cast<double>(x.size() - 1));
}

TTestResult paired_t_test(const std::vector<double>& before,
                          const std::vector<double>& after) {
    if (before.size() != after.size())
        throw std::invalid_argument("paired samples must have equal lengths");
    std::vector<double> differences(before.size());
    for (size_t i = 0; i < before.size(); ++i) differences[i] = before[i] - after[i];
    return one_sample_t_test(differences, 0.0);
}

TTestResult welch_t_test(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() < 2 || y.size() < 2)
        throw std::invalid_argument("each sample must have at least two observations");
    const double mx = statistics::mean(x), my = statistics::mean(y);
    const double vx = statistics::variance(x) / x.size();
    const double vy = statistics::variance(y) / y.size();
    const double squared_se = vx + vy;
    if (!(squared_se > 0.0))
        throw std::invalid_argument("Welch test requires positive combined variance");
    const double df = squared_se * squared_se /
        (vx * vx / (x.size() - 1) + vy * vy / (y.size() - 1));
    return finish((mx - my) / std::sqrt(squared_se), df);
}

std::vector<double> bonferroni_correction(const std::vector<double>& p_values) {
    validate_p_values(p_values);
    std::vector<double> result(p_values.size());
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = std::min(1.0, p_values[i] * p_values.size());
    return result;
}

std::vector<double> holm_correction(const std::vector<double>& p_values) {
    validate_p_values(p_values);
    const auto order = sorted_indices(p_values);
    std::vector<double> result(p_values.size());
    double adjusted = 0.0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        adjusted = std::max(adjusted, (order.size() - rank) * p_values[order[rank]]);
        result[order[rank]] = std::min(1.0, adjusted);
    }
    return result;
}

std::vector<double> benjamini_hochberg_correction(const std::vector<double>& p_values) {
    validate_p_values(p_values);
    const auto order = sorted_indices(p_values);
    std::vector<double> result(p_values.size());
    double adjusted = 1.0;
    for (size_t rank = order.size(); rank-- > 0;) {
        adjusted = std::min(adjusted,
                            p_values[order[rank]] * order.size() / (rank + 1.0));
        result[order[rank]] = adjusted;
    }
    return result;
}

ZTestResult proportion_z_test(int successes, int trials, double null_p) {
    if (trials <= 0 || successes < 0 || successes > trials || !std::isfinite(null_p)
        || null_p <= 0.0 || null_p >= 1.0)
        throw std::invalid_argument("proportion test requires valid counts and null_p in (0, 1)");
    if (trials * null_p < 5.0 || trials * (1.0 - null_p) < 5.0)
        throw std::invalid_argument("normal approximation requires at least five expected outcomes per class");
    const double observed = static_cast<double>(successes) / trials;
    const double statistic = (observed - null_p) /
        std::sqrt(null_p * (1.0 - null_p) / trials);
    return {statistic, 2.0 * distributions::normal_cdf(-std::abs(statistic))};
}

ChiSquareTestResult chi_square_goodness_of_fit(const std::vector<double>& observed,
                                                const std::vector<double>& expected) {
    if (observed.size() < 2 || observed.size() != expected.size())
        throw std::invalid_argument("goodness-of-fit requires at least two matching bins");
    double total_observed = 0.0, total_expected = 0.0, statistic = 0.0;
    for (size_t i = 0; i < observed.size(); ++i) {
        if (!std::isfinite(observed[i]) || observed[i] < 0.0
            || !std::isfinite(expected[i]) || expected[i] <= 0.0)
            throw std::invalid_argument("observed counts must be nonnegative and expected counts positive");
        total_observed += observed[i];
        total_expected += expected[i];
        const double difference = observed[i] - expected[i];
        statistic += difference * difference / expected[i];
    }
    if (!std::isfinite(total_observed) || !std::isfinite(total_expected)
        || total_observed <= 0.0 || !std::isfinite(statistic)
        || std::abs(total_observed - total_expected) > 1e-9 * total_observed)
        throw std::invalid_argument("observed and expected totals must be finite, positive, and equal");
    const double df = static_cast<double>(observed.size() - 1);
    return {statistic, df, 1.0 - distributions::chi_square_cdf(statistic, df)};
}

}  // namespace mathbr::hypothesis
