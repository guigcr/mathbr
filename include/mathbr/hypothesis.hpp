#pragma once

#include <vector>

namespace mathbr::hypothesis {

struct TTestResult {
    double statistic;
    double degrees_of_freedom;
    double p_value;
};

struct ZTestResult {
    double statistic;
    double p_value;
};

struct ChiSquareTestResult {
    double statistic;
    double degrees_of_freedom;
    double p_value;
};

TTestResult one_sample_t_test(const std::vector<double>& x, double null_mean = 0.0);
TTestResult paired_t_test(const std::vector<double>& before,
                          const std::vector<double>& after);
TTestResult welch_t_test(const std::vector<double>& x, const std::vector<double>& y);
std::vector<double> bonferroni_correction(const std::vector<double>& p_values);
std::vector<double> holm_correction(const std::vector<double>& p_values);
std::vector<double> benjamini_hochberg_correction(const std::vector<double>& p_values);
ZTestResult proportion_z_test(int successes, int trials, double null_p);
ChiSquareTestResult chi_square_goodness_of_fit(const std::vector<double>& observed,
                                                const std::vector<double>& expected);

}  // namespace mathbr::hypothesis
