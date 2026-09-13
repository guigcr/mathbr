#pragma once

#include <vector>

namespace mathbr::survival {
struct KaplanMeierResult {
    std::vector<double> times;
    std::vector<double> survival;
    std::vector<int> at_risk;
    std::vector<int> events;
};
struct LogRankResult {
    double statistic;
    double p_value;
};
KaplanMeierResult kaplan_meier(const std::vector<double>& times,
                              const std::vector<int>& events);
LogRankResult log_rank_test(const std::vector<double>& times,
                            const std::vector<int>& events,
                            const std::vector<int>& groups);
}  // namespace mathbr::survival
