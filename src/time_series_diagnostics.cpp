#include "mathbr/time_series_diagnostics.hpp"
#include "mathbr/distributions.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mathbr::time_series_diagnostics {
namespace {
void validate(const std::vector<double>& x, int max_lag) {
    if (max_lag < 0 || x.size() < 2 || static_cast<std::size_t>(max_lag) >= x.size())
        throw std::invalid_argument("require at least two observations and 0 <= lag < n");
    for (double value : x)
        if (!std::isfinite(value)) throw std::invalid_argument("observations must be finite");
}
}  // namespace

std::vector<double> acf(const std::vector<double>& x, int max_lag) {
    validate(x, max_lag);
    const double center = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double denominator = 0;
    for (double value : x) {
        const double deviation = value - center;
        denominator += deviation * deviation;
    }
    if (denominator == 0) throw std::invalid_argument("ACF requires varying observations");
    std::vector<double> result(static_cast<std::size_t>(max_lag) + 1, 1.0);
    for (int lag = 1; lag <= max_lag; ++lag) {
        double numerator = 0;
        for (std::size_t i = static_cast<std::size_t>(lag); i < x.size(); ++i)
            numerator += (x[i] - center) * (x[i - lag] - center);
        result[static_cast<std::size_t>(lag)] = numerator / denominator;
    }
    return result;
}

std::vector<double> pacf(const std::vector<double>& x, int max_lag) {
    const auto correlations = acf(x, max_lag);
    std::vector<double> result(static_cast<std::size_t>(max_lag) + 1, 1.0);
    std::vector<double> previous;
    double residual_variance = 1.0;
    for (int lag = 1; lag <= max_lag; ++lag) {
        double numerator = correlations[static_cast<std::size_t>(lag)];
        for (int j = 1; j < lag; ++j)
            numerator -= previous[static_cast<std::size_t>(j - 1)]
                       * correlations[static_cast<std::size_t>(lag - j)];
        if (residual_variance <= 0)
            throw std::invalid_argument("PACF recurrence requires positive variance");
        const double current = numerator / residual_variance;
        result[static_cast<std::size_t>(lag)] = current;
        std::vector<double> next(static_cast<std::size_t>(lag));
        for (int j = 1; j < lag; ++j)
            next[static_cast<std::size_t>(j - 1)] = previous[static_cast<std::size_t>(j - 1)]
                - current * previous[static_cast<std::size_t>(lag - j - 1)];
        next.back() = current;
        previous = std::move(next);
        residual_variance *= 1 - current * current;
    }
    return result;
}

LjungBoxResult ljung_box(const std::vector<double>& x, int lags) {
    if (lags < 1) throw std::invalid_argument("lags must be positive");
    const auto correlations = acf(x, lags);
    const double n = static_cast<double>(x.size());
    double sum = 0;
    for (int lag = 1; lag <= lags; ++lag) {
        const double r = correlations[static_cast<std::size_t>(lag)];
        sum += r * r / (n - lag);
    }
    const double statistic = n * (n + 2) * sum;
    return {statistic, 1 - mathbr::distributions::chi_square_cdf(statistic,
                                                                  static_cast<double>(lags)), lags};
}
}  // namespace mathbr::time_series_diagnostics
