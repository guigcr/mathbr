#pragma once

#include <vector>

namespace mathbr::diagnostics {
double aic(double log_likelihood, int n_parameters);
double bic(double log_likelihood, int n_parameters, int n_observations);
double hqic(double log_likelihood, int n_parameters, int n_observations);
double residual_standard_error(const std::vector<double>& residuals, int n_parameters);
double durbin_watson(const std::vector<double>& residuals);
double jarque_bera_statistic(const std::vector<double>& residuals);
double jarque_bera_p_value(const std::vector<double>& residuals);
}  // namespace mathbr::diagnostics
