#include "mathbr/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <limits>
#include <stdexcept>

namespace mathbr::statistics {
namespace {
void validate(const std::vector<double>& x) {
    if (x.empty()) throw std::invalid_argument("data must be nonempty");
    for (double value : x)
        if (!std::isfinite(value)) throw std::invalid_argument("data must be finite");
}

void validate_pair(const std::vector<double>& x, const std::vector<double>& y) {
    validate(x);
    validate(y);
    if (x.size() != y.size()) throw std::invalid_argument("data lengths must match");
}

double quantile_sorted(const std::vector<double>& sorted, double p) {
    const double index = p * (sorted.size() - 1);
    const auto lower = static_cast<size_t>(index);
    const auto upper = std::min(lower + 1, sorted.size() - 1);
    return sorted[lower] + (index - lower) * (sorted[upper] - sorted[lower]);
}

double weight_sum(const std::vector<double>& x, const std::vector<double>& weights) {
    validate_pair(x, weights);
    double total = 0.0;
    for (double weight : weights) {
        if (weight <= 0.0) throw std::invalid_argument("weights must be positive and finite");
        total += weight;
    }
    if (!std::isfinite(total)) throw std::invalid_argument("weight sum must be finite");
    return total;
}

std::vector<double> average_ranks(const std::vector<double>& x) {
    std::vector<size_t> order(x.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return x[a] < x[b]; });
    std::vector<double> ranks(x.size());
    for (size_t i = 0; i < order.size();) {
        size_t j = i + 1;
        while (j < order.size() && x[order[j]] == x[order[i]]) ++j;
        const double average = 0.5 * (i + j - 1) + 1.0;
        for (size_t k = i; k < j; ++k) ranks[order[k]] = average;
        i = j;
    }
    return ranks;
}

void validate_matrix(const std::vector<std::vector<double>>& data) {
    if (data.empty() || data.front().empty())
        throw std::invalid_argument("data matrix must have rows and columns");
    const size_t columns = data.front().size();
    for (const auto& row : data) {
        if (row.size() != columns) throw std::invalid_argument("data matrix must be rectangular");
        for (double value : row)
            if (!std::isfinite(value)) throw std::invalid_argument("data matrix must be finite");
    }
}

std::vector<std::vector<double>> matrix_covariance(
    const std::vector<std::vector<double>>& data, int ddof) {
    validate_matrix(data);
    const size_t n = data.size(), p = data.front().size();
    if (ddof < 0 || static_cast<size_t>(ddof) >= n)
        throw std::invalid_argument("ddof must be nonnegative and less than row count");
    std::vector<std::vector<double>> columns(p, std::vector<double>(n));
    std::vector<double> means(p, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < p; ++j) means[j] += data[i][j];
    for (double& value : means) value /= n;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < p; ++j) columns[j][i] = data[i][j] - means[j];
    std::vector<std::vector<double>> result(p, std::vector<double>(p));
    for (size_t j = 0; j < p; ++j) {
        for (size_t k = j; k < p; ++k) {
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i) sum += columns[j][i] * columns[k][i];
            result[j][k] = result[k][j] = sum / (n - ddof);
        }
    }
    return result;
}

std::vector<std::vector<double>> matrix_weighted_covariance(
    const std::vector<std::vector<double>>& data, const std::vector<double>& weights) {
    validate_matrix(data);
    const size_t n = data.size(), p = data.front().size();
    if (weights.size() != n) throw std::invalid_argument("weight count must match row count");
    double total = 0.0;
    for (double weight : weights) {
        if (!std::isfinite(weight) || weight <= 0.0)
            throw std::invalid_argument("weights must be positive and finite");
        total += weight;
    }
    if (!std::isfinite(total)) throw std::invalid_argument("weight sum must be finite");
    std::vector<double> means(p, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < p; ++j)
            means[j] += (weights[i] / total) * (data[i][j] - data[0][j]);
    for (size_t j = 0; j < p; ++j) means[j] += data[0][j];
    std::vector<std::vector<double>> columns(p, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i) {
        const double scale = std::sqrt(weights[i] / total);
        for (size_t j = 0; j < p; ++j)
            columns[j][i] = scale * (data[i][j] - means[j]);
    }
    std::vector<std::vector<double>> result(p, std::vector<double>(p));
    for (size_t j = 0; j < p; ++j)
        for (size_t k = j; k < p; ++k) {
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i) sum += columns[j][i] * columns[k][i];
            result[j][k] = result[k][j] = sum;
        }
    return result;
}

std::vector<std::vector<double>> normalize_covariance(std::vector<std::vector<double>> result) {
    const size_t p = result.size();
    std::vector<double> scales(p);
    for (size_t j = 0; j < p; ++j) {
        if (!(result[j][j] > 0.0))
            throw std::invalid_argument("correlation matrix requires varying columns");
        scales[j] = std::sqrt(result[j][j]);
    }
    for (size_t j = 0; j < p; ++j) {
        result[j][j] = 1.0;
        for (size_t k = j + 1; k < p; ++k)
            result[j][k] = result[k][j] = result[j][k] / (scales[j] * scales[k]);
    }
    return result;
}
}  // namespace

double mean(const std::vector<double>& x) {
    validate(x);
    return std::accumulate(x.begin(), x.end(), 0.0) / x.size();
}

double median(const std::vector<double>& x) { return quantile(x, 0.5); }

double mode(const std::vector<double>& x) {
    validate(x);
    auto sorted = x;
    std::sort(sorted.begin(), sorted.end());
    double best = sorted[0];
    size_t best_count = 0;
    for (size_t i = 0; i < sorted.size();) {
        size_t j = i + 1;
        while (j < sorted.size() && sorted[j] == sorted[i]) ++j;
        if (j - i > best_count) { best_count = j - i; best = sorted[i]; }
        i = j;
    }
    return best;
}

double variance(const std::vector<double>& x, int ddof) {
    validate(x);
    if (ddof < 0 || static_cast<size_t>(ddof) >= x.size())
        throw std::invalid_argument("ddof must be nonnegative and less than data length");
    const double center = mean(x);
    double sum = 0.0;
    for (double value : x) { const double d = value - center; sum += d * d; }
    return sum / (x.size() - ddof);
}

double standard_deviation(const std::vector<double>& x, int ddof) {
    return std::sqrt(variance(x, ddof));
}

double skewness(const std::vector<double>& x) {
    validate(x);
    const double center = mean(x);
    double m2 = 0.0, m3 = 0.0;
    for (double value : x) {
        const double d = value - center;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= x.size();
    m3 /= x.size();
    if (m2 == 0.0) throw std::invalid_argument("skewness requires positive variance");
    return m3 / std::pow(m2, 1.5);
}

double excess_kurtosis(const std::vector<double>& x) {
    validate(x);
    const double center = mean(x);
    double m2 = 0.0, m4 = 0.0;
    for (double value : x) {
        const double d = value - center;
        const double d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= x.size();
    m4 /= x.size();
    if (m2 == 0.0) throw std::invalid_argument("kurtosis requires positive variance");
    return m4 / (m2 * m2) - 3.0;
}

double quantile(const std::vector<double>& x, double p) {
    validate(x);
    if (!std::isfinite(p) || p < 0.0 || p > 1.0)
        throw std::invalid_argument("p must be finite and in [0, 1]");
    auto sorted = x;
    std::sort(sorted.begin(), sorted.end());
    return quantile_sorted(sorted, p);
}

double percentile(const std::vector<double>& x, double p) {
    if (!std::isfinite(p) || p < 0.0 || p > 100.0)
        throw std::invalid_argument("p must be finite and in [0, 100]");
    return quantile(x, p / 100.0);
}

double interquartile_range(const std::vector<double>& x) {
    validate(x);
    auto sorted = x;
    std::sort(sorted.begin(), sorted.end());
    return quantile_sorted(sorted, 0.75) - quantile_sorted(sorted, 0.25);
}

std::vector<double> five_number_summary(const std::vector<double>& x) {
    validate(x);
    auto sorted = x;
    std::sort(sorted.begin(), sorted.end());
    return {sorted.front(), quantile_sorted(sorted, 0.25),
            quantile_sorted(sorted, 0.5), quantile_sorted(sorted, 0.75), sorted.back()};
}

double covariance(const std::vector<double>& x, const std::vector<double>& y, int ddof) {
    validate_pair(x, y);
    if (ddof < 0 || static_cast<size_t>(ddof) >= x.size())
        throw std::invalid_argument("ddof must be nonnegative and less than data length");
    const double mx = mean(x), my = mean(y);
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) sum += (x[i] - mx) * (y[i] - my);
    return sum / (x.size() - ddof);
}

double pearson_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    validate_pair(x, y);
    const double denominator = std::sqrt(variance(x, 0) * variance(y, 0));
    if (denominator == 0.0) throw std::invalid_argument("correlation requires varying data");
    return covariance(x, y, 0) / denominator;
}

double spearman_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    validate_pair(x, y);
    return pearson_correlation(average_ranks(x), average_ranks(y));
}

double kendall_tau(const std::vector<double>& x, const std::vector<double>& y) {
    validate_pair(x, y);
    const size_t n = x.size();
    std::vector<std::pair<double, double>> pairs(n);
    for (size_t i = 0; i < n; ++i) pairs[i] = {x[i], y[i]};
    std::sort(pairs.begin(), pairs.end());

    std::vector<double> levels = y;
    std::sort(levels.begin(), levels.end());
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    std::vector<size_t> tree(levels.size() + 1, 0);
    auto prefix = [&](size_t end) {
        size_t count = 0;
        for (; end > 0; end -= end & (~end + 1)) count += tree[end];
        return count;
    };
    auto add = [&](size_t index) {
        for (++index; index < tree.size(); index += index & (~index + 1)) ++tree[index];
    };

    long double concordant = 0, discordant = 0, x_ties = 0;
    for (size_t i = 0; i < n;) {
        size_t j = i + 1;
        while (j < n && pairs[j].first == pairs[i].first) ++j;
        const long double group = static_cast<long double>(j - i);
        x_ties += group * (group - 1) / 2;
        for (size_t k = i; k < j; ++k) {
            const size_t rank = std::lower_bound(levels.begin(), levels.end(), pairs[k].second)
                                - levels.begin();
            concordant += prefix(rank);
            discordant += i - prefix(rank + 1);
        }
        for (size_t k = i; k < j; ++k) {
            const size_t rank = std::lower_bound(levels.begin(), levels.end(), pairs[k].second)
                                - levels.begin();
            add(rank);
        }
        i = j;
    }
    long double y_ties = 0;
    for (size_t i = 0; i < levels.size(); ++i) {
        const long double count = static_cast<long double>(prefix(i + 1) - prefix(i));
        y_ties += count * (count - 1) / 2;
    }
    const long double total = static_cast<long double>(n) * (n - 1) / 2;
    const long double denominator = std::sqrt((total - x_ties) * (total - y_ties));
    if (denominator == 0) throw std::invalid_argument("Kendall tau requires varying data");
    return static_cast<double>((concordant - discordant) / denominator);
}

std::vector<std::vector<double>> covariance_matrix(
    const std::vector<std::vector<double>>& data, int ddof) {
    return matrix_covariance(data, ddof);
}

std::vector<std::vector<double>> correlation_matrix(
    const std::vector<std::vector<double>>& data) {
    return normalize_covariance(matrix_covariance(data, 0));
}

std::vector<std::vector<double>> spearman_correlation_matrix(
    const std::vector<std::vector<double>>& data) {
    validate_matrix(data);
    const size_t n = data.size(), p = data.front().size();
    std::vector<std::vector<double>> ranked(n, std::vector<double>(p));
    for (size_t j = 0; j < p; ++j) {
        std::vector<double> column(n);
        for (size_t i = 0; i < n; ++i) column[i] = data[i][j];
        auto ranks = average_ranks(column);
        for (size_t i = 0; i < n; ++i) ranked[i][j] = ranks[i];
    }
    return normalize_covariance(matrix_covariance(ranked, 0));
}

std::vector<std::vector<double>> kendall_correlation_matrix(
    const std::vector<std::vector<double>>& data) {
    validate_matrix(data);
    const size_t n = data.size(), p = data.front().size();
    std::vector<std::vector<double>> columns(p, std::vector<double>(n));
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < p; ++j) columns[j][i] = data[i][j];
    std::vector<std::vector<double>> result(p, std::vector<double>(p, 1.0));
    for (size_t j = 0; j < p; ++j) {
        // Validate diagonal as a varying column, even for a single-column matrix.
        if (*std::min_element(columns[j].begin(), columns[j].end()) ==
            *std::max_element(columns[j].begin(), columns[j].end()))
            throw std::invalid_argument("correlation matrix requires varying columns");
        for (size_t k = j + 1; k < p; ++k)
            result[j][k] = result[k][j] = kendall_tau(columns[j], columns[k]);
    }
    return result;
}

std::vector<std::vector<double>> weighted_covariance_matrix(
    const std::vector<std::vector<double>>& data, const std::vector<double>& weights) {
    return matrix_weighted_covariance(data, weights);
}

std::vector<std::vector<double>> weighted_correlation_matrix(
    const std::vector<std::vector<double>>& data, const std::vector<double>& weights) {
    return normalize_covariance(matrix_weighted_covariance(data, weights));
}

double weighted_mean(const std::vector<double>& x, const std::vector<double>& weights) {
    const double total = weight_sum(x, weights);
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) sum += weights[i] * x[i];
    return sum / total;
}

double weighted_covariance(const std::vector<double>& x, const std::vector<double>& y,
                           const std::vector<double>& weights) {
    validate_pair(x, y);
    const double total = weight_sum(x, weights);
    const double mx = weighted_mean(x, weights), my = weighted_mean(y, weights);
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) sum += weights[i] * (x[i] - mx) * (y[i] - my);
    return sum / total;
}

double weighted_variance(const std::vector<double>& x, const std::vector<double>& weights) {
    return weighted_covariance(x, x, weights);
}

double weighted_correlation(const std::vector<double>& x, const std::vector<double>& y,
                            const std::vector<double>& weights) {
    const double denominator = std::sqrt(weighted_variance(x, weights)
                                         * weighted_variance(y, weights));
    if (denominator == 0.0) throw std::invalid_argument("correlation requires varying data");
    return weighted_covariance(x, y, weights) / denominator;
}

double weighted_quantile(const std::vector<double>& x,
                         const std::vector<double>& weights, double p) {
    const double total = weight_sum(x, weights);
    if (!std::isfinite(p) || p < 0 || p > 1)
        throw std::invalid_argument("p must be finite and in [0, 1]");
    std::vector<std::size_t> order(x.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return x[a] < x[b]; });
    double cumulative = 0;
    for (std::size_t index : order) {
        cumulative += weights[index];
        if (cumulative / total >= p) return x[index];
    }
    return x[order.back()];
}

double log_sum_exp(const std::vector<double>& x) {
    validate(x);
    const double maximum = *std::max_element(x.begin(), x.end());
    double sum = 0;
    for (double value : x) sum += std::exp(value - maximum);
    return maximum + std::log(sum);
}

double log_empirical_mgf(const std::vector<double>& x, double t) {
    validate(x);
    if (!std::isfinite(t)) throw std::invalid_argument("t must be finite");
    if (t == 0.0) return 0.0;
    double maximum = -std::numeric_limits<double>::infinity();
    for (double value : x) {
        const double product = t * value;
        if (!std::isfinite(product))
            throw std::invalid_argument("t times data must be finite");
        maximum = std::max(maximum, product);
    }
    double sum = 0.0;
    for (double value : x) sum += std::exp(t * value - maximum);
    return maximum + std::log(sum / x.size());
}

double empirical_mgf(const std::vector<double>& x, double t) {
    const double result = std::exp(log_empirical_mgf(x, t));
    if (!std::isfinite(result)) throw std::invalid_argument("MGF exceeds finite range");
    return result;
}

}  // namespace mathbr::statistics
