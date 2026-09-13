#include "mathbr/bayesian.hpp"
#include "mathbr/distributions.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace mathbr::bayesian {
namespace {
void positive(double value) {
    if (!std::isfinite(value) || value <= 0)
        throw std::invalid_argument("shape and standard deviation parameters must be finite and positive");
}
void interval_level(double level) {
    if (!std::isfinite(level) || level <= 0 || level >= 1)
        throw std::invalid_argument("credible level must be in (0, 1)");
}
}  // namespace

double BetaPosterior::mean() const { return alpha / (alpha + beta); }
std::vector<double> BetaPosterior::credible_interval(double level) const {
    interval_level(level);
    const double tail = (1 - level) / 2;
    return {mathbr::distributions::beta_ppf(tail, alpha, beta),
            mathbr::distributions::beta_ppf(1 - tail, alpha, beta)};
}
std::vector<double> NormalPosterior::credible_interval(double level) const {
    interval_level(level);
    const double tail = (1 - level) / 2;
    return {mean + standard_deviation * mathbr::distributions::normal_ppf(tail),
            mean + standard_deviation * mathbr::distributions::normal_ppf(1 - tail)};
}
BetaPosterior beta_binomial_update(double alpha, double beta, int successes, int trials) {
    positive(alpha); positive(beta);
    if (trials < 0 || successes < 0 || successes > trials)
        throw std::invalid_argument("require 0 <= successes <= trials");
    const double posterior_alpha = alpha + successes;
    const double posterior_beta = beta + (trials - successes);
    if (!std::isfinite(posterior_alpha) || !std::isfinite(posterior_beta))
        throw std::invalid_argument("posterior parameters must be finite");
    return {posterior_alpha, posterior_beta};
}
NormalPosterior normal_normal_update(double prior_mean, double prior_sd,
                                     double observation_sd,
                                     const std::vector<double>& observations) {
    if (!std::isfinite(prior_mean) || observations.empty())
        throw std::invalid_argument("prior mean must be finite and observations nonempty");
    positive(prior_sd); positive(observation_sd);
    for (double value : observations)
        if (!std::isfinite(value)) throw std::invalid_argument("observations must be finite");
    const double prior_precision = 1 / (prior_sd * prior_sd);
    const double observation_precision = 1 / (observation_sd * observation_sd);
    const double posterior_precision = prior_precision + observations.size() * observation_precision;
    if (!std::isfinite(posterior_precision) || posterior_precision <= 0)
        throw std::invalid_argument("posterior precision must be positive and finite");
    const double sum = std::accumulate(observations.begin(), observations.end(), 0.0);
    const double posterior_mean = (prior_precision * prior_mean + observation_precision * sum)
                                / posterior_precision;
    if (!std::isfinite(posterior_mean))
        throw std::invalid_argument("posterior mean must be finite");
    return {posterior_mean, std::sqrt(1 / posterior_precision)};
}
}  // namespace mathbr::bayesian
