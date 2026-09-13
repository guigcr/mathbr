#pragma once

#include <vector>

namespace mathbr::nonparametric {
std::vector<double> empirical_cdf(const std::vector<double>& sample,
                                  const std::vector<double>& points);
std::vector<double> gaussian_kde(const std::vector<double>& sample,
                                 const std::vector<double>& points, double bandwidth);
std::vector<double> nadaraya_watson(const std::vector<double>& x,
                                    const std::vector<double>& y,
                                    const std::vector<double>& points,
                                    double bandwidth);
}  // namespace mathbr::nonparametric
