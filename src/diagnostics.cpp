#include "mathbr/diagnostics.hpp"
#include "mathbr/statistics.hpp"

#include <cmath>
#include <stdexcept>

namespace mathbr::diagnostics {
namespace {
void validate_likelihood(double log_likelihood, int n_parameters) {
    if (!std::isfinite(log_likelihood) || n_parameters < 0)
        throw std::invalid_argument("log likelihood must be finite and parameter count nonnegative");
}
void validate_residuals(const std::vector<double>& residuals, std::size_t minimum) {
    if (residuals.size() < minimum)
        throw std::invalid_argument("insufficient residuals");
    for (double value : residuals)
        if (!std::isfinite(value)) throw std::invalid_argument("residuals must be finite");
}
}  // namespace

double aic(double log_likelihood, int n_parameters) {
    validate_likelihood(log_likelihood, n_parameters);
    return -2 * log_likelihood + 2 * n_parameters;
}
double bic(double log_likelihood, int n_parameters, int n_observations) {
    validate_likelihood(log_likelihood, n_parameters);
    if (n_observations < 1) throw std::invalid_argument("n_observations must be positive");
    return -2 * log_likelihood + n_parameters * std::log(n_observations);
}
double hqic(double log_likelihood, int n_parameters, int n_observations) {
    validate_likelihood(log_likelihood, n_parameters);
    if (n_observations < 3) throw std::invalid_argument("HQIC requires at least three observations");
    return -2 * log_likelihood + 2 * n_parameters * std::log(std::log(n_observations));
}
double residual_standard_error(const std::vector<double>& residuals, int n_parameters) {
    validate_residuals(residuals, 1);
    if (n_parameters < 0 || static_cast<std::size_t>(n_parameters) >= residuals.size())
        throw std::invalid_argument("residual degrees of freedom must be positive");
    double sum = 0;
    for (double value : residuals) sum += value * value;
    return std::sqrt(sum / (residuals.size() - n_parameters));
}
double durbin_watson(const std::vector<double>& residuals) {
    validate_residuals(residuals, 2);
    double numerator = 0, denominator = 0;
    for (std::size_t i = 0; i < residuals.size(); ++i) {
        denominator += residuals[i] * residuals[i];
        if (i > 0) {
            const double difference = residuals[i] - residuals[i - 1];
            numerator += difference * difference;
        }
    }
    if (denominator == 0) throw std::invalid_argument("Durbin-Watson requires nonzero residuals");
    return numerator / denominator;
}
double jarque_bera_statistic(const std::vector<double>& residuals) {
    validate_residuals(residuals, 3);
    const double skew = mathbr::statistics::skewness(residuals);
    const double excess = mathbr::statistics::excess_kurtosis(residuals);
    return residuals.size() * (skew * skew + excess * excess / 4) / 6;
}
double jarque_bera_p_value(const std::vector<double>& residuals) {
    // A chi-square distribution with two degrees of freedom has survival exp(-x/2).
    return std::exp(-jarque_bera_statistic(residuals) / 2);
}
}  // namespace mathbr::diagnostics
