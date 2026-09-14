#include "mathbr/hypothesis.hpp"
#include "mathbr/distributions.hpp"
#include "mathbr/statistics.hpp"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace mathbr::hypothesis {
namespace {
TTestResult finish(double statistic, double df) {
    if (!std::isfinite(statistic) || !std::isfinite(df) || df <= 0.0)
        throw std::invalid_argument("t-test requires finite statistic and positive degrees of freedom");
    return {statistic, df, 2.0 * distributions::student_t_cdf(-std::abs(statistic), df)};
}
void validate_p_values(const std::vector<double>& p_values) {
    if (p_values.empty()) throw std::invalid_argument("p_values must be nonempty");
    for (double p : p_values)
        if (!std::isfinite(p) || p < 0.0 || p > 1.0)
            throw std::invalid_argument("p_values must be finite and in [0, 1]");
}

std::vector<size_t> sorted_indices(const std::vector<double>& p_values) {
    std::vector<size_t> order(p_values.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return p_values[a] < p_values[b]; });
    return order;
}

size_t validate_groups(const std::vector<std::vector<double>>& groups) {
    if (groups.size() < 2) throw std::invalid_argument("at least two groups are required");
    size_t total = 0;
    for (const auto& group : groups) {
        if (group.empty()) throw std::invalid_argument("groups must be nonempty");
        total += group.size();
        for (double value : group)
            if (!std::isfinite(value)) throw std::invalid_argument("group values must be finite");
    }
    return total;
}
}  // namespace

TTestResult one_sample_t_test(const std::vector<double>& x, double null_mean) {
    if (!std::isfinite(null_mean)) throw std::invalid_argument("null_mean must be finite");
    if (x.size() < 2) throw std::invalid_argument("t-test requires at least two observations");
    const double center = statistics::mean(x);
    const double sample_variance = statistics::variance(x);
    if (!(sample_variance > 0.0))
        throw std::invalid_argument("t-test requires positive sample variance");
    return finish((center - null_mean) / std::sqrt(sample_variance / x.size()),
                  static_cast<double>(x.size() - 1));
}

TTestResult paired_t_test(const std::vector<double>& before,
                          const std::vector<double>& after) {
    if (before.size() != after.size())
        throw std::invalid_argument("paired samples must have equal lengths");
    std::vector<double> differences(before.size());
    for (size_t i = 0; i < before.size(); ++i) differences[i] = before[i] - after[i];
    return one_sample_t_test(differences, 0.0);
}

TTestResult welch_t_test(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() < 2 || y.size() < 2)
        throw std::invalid_argument("each sample must have at least two observations");
    const double mx = statistics::mean(x), my = statistics::mean(y);
    const double vx = statistics::variance(x) / x.size();
    const double vy = statistics::variance(y) / y.size();
    const double squared_se = vx + vy;
    if (!(squared_se > 0.0))
        throw std::invalid_argument("Welch test requires positive combined variance");
    const double df = squared_se * squared_se /
        (vx * vx / (x.size() - 1) + vy * vy / (y.size() - 1));
    return finish((mx - my) / std::sqrt(squared_se), df);
}

std::vector<double> bonferroni_correction(const std::vector<double>& p_values) {
    validate_p_values(p_values);
    std::vector<double> result(p_values.size());
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = std::min(1.0, p_values[i] * p_values.size());
    return result;
}

std::vector<double> holm_correction(const std::vector<double>& p_values) {
    validate_p_values(p_values);
    const auto order = sorted_indices(p_values);
    std::vector<double> result(p_values.size());
    double adjusted = 0.0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        adjusted = std::max(adjusted, (order.size() - rank) * p_values[order[rank]]);
        result[order[rank]] = std::min(1.0, adjusted);
    }
    return result;
}

std::vector<double> benjamini_hochberg_correction(const std::vector<double>& p_values) {
    validate_p_values(p_values);
    const auto order = sorted_indices(p_values);
    std::vector<double> result(p_values.size());
    double adjusted = 1.0;
    for (size_t rank = order.size(); rank-- > 0;) {
        adjusted = std::min(adjusted,
                            p_values[order[rank]] * order.size() / (rank + 1.0));
        result[order[rank]] = adjusted;
    }
    return result;
}

ZTestResult proportion_z_test(int successes, int trials, double null_p) {
    if (trials <= 0 || successes < 0 || successes > trials || !std::isfinite(null_p)
        || null_p <= 0.0 || null_p >= 1.0)
        throw std::invalid_argument("proportion test requires valid counts and null_p in (0, 1)");
    if (trials * null_p < 5.0 || trials * (1.0 - null_p) < 5.0)
        throw std::invalid_argument("normal approximation requires at least five expected outcomes per class");
    const double observed = static_cast<double>(successes) / trials;
    const double statistic = (observed - null_p) /
        std::sqrt(null_p * (1.0 - null_p) / trials);
    return {statistic, 2.0 * distributions::normal_cdf(-std::abs(statistic))};
}

ChiSquareTestResult chi_square_goodness_of_fit(const std::vector<double>& observed,
                                                const std::vector<double>& expected) {
    if (observed.size() < 2 || observed.size() != expected.size())
        throw std::invalid_argument("goodness-of-fit requires at least two matching bins");
    double total_observed = 0.0, total_expected = 0.0, statistic = 0.0;
    for (size_t i = 0; i < observed.size(); ++i) {
        if (!std::isfinite(observed[i]) || observed[i] < 0.0
            || !std::isfinite(expected[i]) || expected[i] <= 0.0)
            throw std::invalid_argument("observed counts must be nonnegative and expected counts positive");
        total_observed += observed[i];
        total_expected += expected[i];
        const double difference = observed[i] - expected[i];
        statistic += difference * difference / expected[i];
    }
    if (!std::isfinite(total_observed) || !std::isfinite(total_expected)
        || total_observed <= 0.0 || !std::isfinite(statistic)
        || std::abs(total_observed - total_expected) > 1e-9 * total_observed)
        throw std::invalid_argument("observed and expected totals must be finite, positive, and equal");
    const double df = static_cast<double>(observed.size() - 1);
    return {statistic, df, 1.0 - distributions::chi_square_cdf(statistic, df)};
}

AnovaResult one_way_anova(const std::vector<std::vector<double>>& groups) {
    const size_t n = validate_groups(groups), k = groups.size();
    if (n <= k) throw std::invalid_argument("ANOVA requires positive within-group degrees of freedom");
    std::vector<double> means(k);
    double grand_mean = 0.0;
    for (size_t j = 0; j < k; ++j) {
        const double reference = groups[j].front();
        double offset = 0.0;
        for (double value : groups[j])
            offset += value / groups[j].size() - reference / groups[j].size();
        means[j] = reference + offset;
        grand_mean += (static_cast<double>(groups[j].size()) / n) * means[j];
    }
    double between = 0.0, within = 0.0;
    for (size_t j = 0; j < k; ++j) {
        const double delta = means[j] - grand_mean;
        between += groups[j].size() * delta * delta;
        for (double value : groups[j]) {
            const double residual = value - means[j];
            within += residual * residual;
        }
    }
    if (!(within > 0.0) || !std::isfinite(within) || !std::isfinite(between))
        throw std::invalid_argument("ANOVA requires finite positive within-group variation");
    const double df_between = static_cast<double>(k - 1);
    const double df_within = static_cast<double>(n - k);
    const double statistic = (between / df_between) / (within / df_within);
    if (!std::isfinite(statistic)) throw std::invalid_argument("ANOVA statistic is not finite");
    return {statistic, df_between, df_within,
            1.0 - distributions::f_cdf(statistic, df_between, df_within)};
}

ChiSquareTestResult kruskal_wallis(const std::vector<std::vector<double>>& groups) {
    const size_t n = validate_groups(groups), k = groups.size();
    std::vector<std::pair<double, size_t>> values;
    values.reserve(n);
    for (size_t j = 0; j < k; ++j)
        for (double value : groups[j]) values.emplace_back(value, j);
    std::sort(values.begin(), values.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<double> rank_sums(k, 0.0);
    long double tie_sum = 0.0;
    for (size_t i = 0; i < n;) {
        size_t end = i + 1;
        while (end < n && values[end].first == values[i].first) ++end;
        const double average_rank = 0.5 * (static_cast<double>(i + 1) + end);
        for (size_t index = i; index < end; ++index)
            rank_sums[values[index].second] += average_rank;
        const long double ties = static_cast<long double>(end - i);
        tie_sum += ties * ties * ties - ties;
        i = end;
    }
    const long double count = static_cast<long double>(n);
    const long double correction = 1.0L - tie_sum / (count * count * count - count);
    if (!(correction > 0.0L))
        throw std::invalid_argument("Kruskal-Wallis requires varying pooled data");
    double centered = 0.0;
    for (size_t j = 0; j < k; ++j) {
        const double deviation = rank_sums[j] - groups[j].size() * (n + 1.0) / 2.0;
        centered += deviation * deviation / groups[j].size();
    }
    const double statistic = 12.0 * centered / (n * (n + 1.0)) /
                             static_cast<double>(correction);
    if (!std::isfinite(statistic)) throw std::invalid_argument("Kruskal-Wallis statistic is not finite");
    const double df = static_cast<double>(k - 1);
    return {statistic, df, 1.0 - distributions::chi_square_cdf(statistic, df)};
}

ChiSquareTestResult chi_square_independence(
    const std::vector<std::vector<double>>& table) {
    if (table.size() < 2 || table.front().size() < 2)
        throw std::invalid_argument("contingency table needs at least two rows and columns");
    const size_t rows = table.size(), columns = table.front().size();
    std::vector<double> row_totals(rows, 0.0), column_totals(columns, 0.0);
    double total = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        if (table[i].size() != columns)
            throw std::invalid_argument("contingency table must be rectangular");
        for (size_t j = 0; j < columns; ++j) {
            const double observed = table[i][j];
            if (!std::isfinite(observed) || observed < 0.0)
                throw std::invalid_argument("contingency counts must be finite and nonnegative");
            row_totals[i] += observed;
            column_totals[j] += observed;
            total += observed;
        }
    }
    if (!(total > 0.0) || !std::isfinite(total))
        throw std::invalid_argument("contingency total must be finite and positive");
    for (double value : row_totals)
        if (!(value > 0.0) || !std::isfinite(value))
            throw std::invalid_argument("every row must have a finite positive total");
    for (double value : column_totals)
        if (!(value > 0.0) || !std::isfinite(value))
            throw std::invalid_argument("every column must have a finite positive total");
    double statistic = 0.0;
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < columns; ++j) {
            const double expected = (row_totals[i] / total) * column_totals[j];
            if (!(expected > 0.0))
                throw std::invalid_argument("expected counts must be positive");
            const double difference = table[i][j] - expected;
            statistic += difference * difference / expected;
        }
    if (!std::isfinite(statistic))
        throw std::invalid_argument("chi-square statistic is not finite");
    const double df = static_cast<double>((rows - 1) * (columns - 1));
    return {statistic, df, 1.0 - distributions::chi_square_cdf(statistic, df)};
}

MannWhitneyResult mann_whitney_u(const std::vector<double>& x,
                                 const std::vector<double>& y) {
    if (x.empty() || y.empty())
        throw std::invalid_argument("Mann-Whitney requires two nonempty groups");
    std::vector<std::pair<double, bool>> values;
    values.reserve(x.size() + y.size());
    for (double value : x) {
        if (!std::isfinite(value)) throw std::invalid_argument("samples must be finite");
        values.emplace_back(value, true);
    }
    for (double value : y) {
        if (!std::isfinite(value)) throw std::invalid_argument("samples must be finite");
        values.emplace_back(value, false);
    }
    std::sort(values.begin(), values.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    double rank_sum_x = 0.0;
    long double tie_sum = 0.0;
    for (size_t i = 0; i < values.size();) {
        size_t end = i + 1;
        while (end < values.size() && values[end].first == values[i].first) ++end;
        const double rank = 0.5 * (static_cast<double>(i + 1) + end);
        for (size_t index = i; index < end; ++index)
            if (values[index].second) rank_sum_x += rank;
        const long double tied = static_cast<long double>(end - i);
        tie_sum += tied * tied * tied - tied;
        i = end;
    }
    const double nx = static_cast<double>(x.size()), ny = static_cast<double>(y.size());
    const double n = nx + ny;
    const double u = rank_sum_x - nx * (nx + 1.0) / 2.0;
    const double center = nx * ny / 2.0;
    const double variance = nx * ny / 12.0 *
        (n + 1.0 - static_cast<double>(tie_sum / (n * (n - 1.0))));
    if (!(variance > 0.0) || !std::isfinite(variance))
        throw std::invalid_argument("Mann-Whitney requires varying pooled data");
    const double difference = u - center;
    const double adjusted = std::copysign(std::max(0.0, std::abs(difference) - 0.5),
                                          difference);
    const double z = adjusted / std::sqrt(variance);
    return {u, z, 2.0 * distributions::normal_cdf(-std::abs(z))};
}

}  // namespace mathbr::hypothesis
