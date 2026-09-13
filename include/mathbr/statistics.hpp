#pragma once

#include <vector>

namespace mathbr::statistics {

double mean(const std::vector<double>& x);
double median(const std::vector<double>& x);
double mode(const std::vector<double>& x);
double variance(const std::vector<double>& x, int ddof = 1);
double standard_deviation(const std::vector<double>& x, int ddof = 1);
double skewness(const std::vector<double>& x);
double excess_kurtosis(const std::vector<double>& x);
double quantile(const std::vector<double>& x, double p);
double percentile(const std::vector<double>& x, double p);
double interquartile_range(const std::vector<double>& x);
std::vector<double> five_number_summary(const std::vector<double>& x);
double covariance(const std::vector<double>& x, const std::vector<double>& y, int ddof = 1);
double pearson_correlation(const std::vector<double>& x, const std::vector<double>& y);
double spearman_correlation(const std::vector<double>& x, const std::vector<double>& y);
double weighted_mean(const std::vector<double>& x, const std::vector<double>& weights);
double weighted_variance(const std::vector<double>& x, const std::vector<double>& weights);
double weighted_covariance(const std::vector<double>& x, const std::vector<double>& y,
                           const std::vector<double>& weights);
double weighted_correlation(const std::vector<double>& x, const std::vector<double>& y,
                            const std::vector<double>& weights);
double weighted_quantile(const std::vector<double>& x,
                         const std::vector<double>& weights, double p);
double log_sum_exp(const std::vector<double>& x);

}  // namespace mathbr::statistics
