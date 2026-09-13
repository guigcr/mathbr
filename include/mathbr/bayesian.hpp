#pragma once

#include <vector>

namespace mathbr::bayesian {
struct BetaPosterior {
    double alpha;
    double beta;
    double mean() const;
    std::vector<double> credible_interval(double level = 0.95) const;
};
struct NormalPosterior {
    double mean;
    double standard_deviation;
    std::vector<double> credible_interval(double level = 0.95) const;
};
BetaPosterior beta_binomial_update(double alpha, double beta, int successes, int trials);
NormalPosterior normal_normal_update(double prior_mean, double prior_sd,
                                     double observation_sd,
                                     const std::vector<double>& observations);
}  // namespace mathbr::bayesian
