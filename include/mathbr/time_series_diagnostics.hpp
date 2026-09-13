#pragma once

#include <vector>

namespace mathbr::time_series_diagnostics {
struct LjungBoxResult {
    double statistic;
    double p_value;
    int lags;
};
std::vector<double> acf(const std::vector<double>& x, int max_lag);
std::vector<double> pacf(const std::vector<double>& x, int max_lag);
LjungBoxResult ljung_box(const std::vector<double>& x, int lags);
}  // namespace mathbr::time_series_diagnostics
