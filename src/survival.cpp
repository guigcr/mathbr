#include "mathbr/survival.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mathbr::survival {
namespace {
struct Observation { double time; int event; int group; };
std::vector<Observation> observations(const std::vector<double>& times,
                                       const std::vector<int>& events,
                                       const std::vector<int>* groups = nullptr) {
    if (times.empty() || events.size() != times.size()
        || (groups != nullptr && groups->size() != times.size()))
        throw std::invalid_argument("inputs must have the same nonzero length");
    std::vector<Observation> rows;
    rows.reserve(times.size());
    for (std::size_t i = 0; i < times.size(); ++i) {
        if (!std::isfinite(times[i]) || times[i] < 0 || (events[i] != 0 && events[i] != 1))
            throw std::invalid_argument("times must be nonnegative finite and events binary");
        const int group = groups == nullptr ? 0 : (*groups)[i];
        if (group != 0 && group != 1) throw std::invalid_argument("groups must be 0 or 1");
        rows.push_back({times[i], events[i], group});
    }
    std::sort(rows.begin(), rows.end(),
              [](const Observation& a, const Observation& b) { return a.time < b.time; });
    return rows;
}
}  // namespace

KaplanMeierResult kaplan_meier(const std::vector<double>& times,
                              const std::vector<int>& events) {
    const auto rows = observations(times, events);
    KaplanMeierResult result;
    int risk = static_cast<int>(rows.size());
    double survival = 1;
    for (std::size_t i = 0; i < rows.size();) {
        const double time = rows[i].time;
        int at_time = 0, observed_events = 0;
        do {
            ++at_time;
            observed_events += rows[i].event;
            ++i;
        } while (i < rows.size() && rows[i].time == time);
        survival *= 1 - static_cast<double>(observed_events) / risk;
        result.times.push_back(time);
        result.survival.push_back(survival);
        result.at_risk.push_back(risk);
        result.events.push_back(observed_events);
        risk -= at_time;
    }
    return result;
}

LogRankResult log_rank_test(const std::vector<double>& times,
                            const std::vector<int>& events,
                            const std::vector<int>& groups) {
    const auto rows = observations(times, events, &groups);
    int risk1 = 0;
    for (const auto& row : rows) risk1 += row.group;
    int risk0 = static_cast<int>(rows.size()) - risk1;
    if (risk0 == 0 || risk1 == 0)
        throw std::invalid_argument("both groups are required");
    double observed_minus_expected = 0, variance = 0;
    for (std::size_t i = 0; i < rows.size();) {
        const double time = rows[i].time;
        int removed0 = 0, removed1 = 0, events0 = 0, events1 = 0;
        do {
            if (rows[i].group == 0) { ++removed0; events0 += rows[i].event; }
            else { ++removed1; events1 += rows[i].event; }
            ++i;
        } while (i < rows.size() && rows[i].time == time);
        const int n = risk0 + risk1;
        const int d = events0 + events1;
        if (d > 0) {
            observed_minus_expected += events1 - static_cast<double>(risk1) * d / n;
            if (n > 1)
                variance += static_cast<double>(risk1) * risk0 * d * (n - d)
                          / (static_cast<double>(n) * n * (n - 1));
        }
        risk0 -= removed0;
        risk1 -= removed1;
    }
    if (variance <= 0) throw std::invalid_argument("log-rank variance must be positive");
    const double statistic = observed_minus_expected * observed_minus_expected / variance;
    return {statistic, std::erfc(std::sqrt(statistic / 2))};
}
}  // namespace mathbr::survival
