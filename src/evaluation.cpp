#include "mathbr/evaluation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace mathbr::evaluation {
namespace {
void validate_labels(const std::vector<int>& y_true, std::size_t other_size) {
    if (y_true.empty() || y_true.size() != other_size)
        throw std::invalid_argument("inputs must have the same nonzero length");
    for (int label : y_true)
        if (label != 0 && label != 1)
            throw std::invalid_argument("labels must be 0 or 1");
}

struct Counts { double tn = 0, fp = 0, fn = 0, tp = 0; };
Counts count_predictions(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    validate_labels(y_true, y_pred.size());
    Counts c;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_pred[i] != 0 && y_pred[i] != 1)
            throw std::invalid_argument("predictions must be 0 or 1");
        if (y_true[i] == 1) {
            if (y_pred[i] == 1) ++c.tp; else ++c.fn;
        } else {
            if (y_pred[i] == 1) ++c.fp; else ++c.tn;
        }
    }
    return c;
}

struct RankedScores {
    std::vector<std::size_t> order;
    double positives = 0;
    double negatives = 0;
};
RankedScores rank_scores(const std::vector<int>& y_true, const std::vector<double>& scores) {
    validate_labels(y_true, scores.size());
    RankedScores ranked;
    ranked.order.resize(scores.size());
    std::iota(ranked.order.begin(), ranked.order.end(), 0);
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (!std::isfinite(scores[i])) throw std::invalid_argument("scores must be finite");
        ranked.positives += y_true[i];
    }
    ranked.negatives = static_cast<double>(scores.size()) - ranked.positives;
    if (ranked.positives == 0 || ranked.negatives == 0)
        throw std::invalid_argument("both classes are required");
    std::sort(ranked.order.begin(), ranked.order.end(),
              [&scores](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
    return ranked;
}
void validate_regression(const std::vector<double>& y_true,
                         const std::vector<double>& y_pred) {
    if (y_true.empty() || y_true.size() != y_pred.size())
        throw std::invalid_argument("inputs must have the same nonzero length");
    for (std::size_t i = 0; i < y_true.size(); ++i)
        if (!std::isfinite(y_true[i]) || !std::isfinite(y_pred[i]))
            throw std::invalid_argument("inputs must be finite");
}
void validate_probabilities(const std::vector<int>& labels,
                            const std::vector<double>& probabilities) {
    validate_labels(labels, probabilities.size());
    for (double p : probabilities)
        if (!std::isfinite(p) || p < 0 || p > 1)
            throw std::invalid_argument("probabilities must be finite and in [0, 1]");
}
}  // namespace

std::vector<std::vector<int>> confusion_matrix(const std::vector<int>& y_true,
                                                const std::vector<int>& y_pred) {
    const auto c = count_predictions(y_true, y_pred);
    return {{static_cast<int>(c.tn), static_cast<int>(c.fp)},
            {static_cast<int>(c.fn), static_cast<int>(c.tp)}};
}
double precision(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    const auto c = count_predictions(y_true, y_pred);
    return c.tp + c.fp == 0 ? 0 : c.tp / (c.tp + c.fp);
}
double recall(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    const auto c = count_predictions(y_true, y_pred);
    return c.tp + c.fn == 0 ? 0 : c.tp / (c.tp + c.fn);
}
double f1_score(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    const auto c = count_predictions(y_true, y_pred);
    const double denominator = 2 * c.tp + c.fp + c.fn;
    return denominator == 0 ? 0 : 2 * c.tp / denominator;
}
RocCurve roc_curve(const std::vector<int>& y_true, const std::vector<double>& scores) {
    const auto ranked = rank_scores(y_true, scores);
    RocCurve result{{0.0}, {0.0}, {std::numeric_limits<double>::infinity()}};
    double tp = 0, fp = 0;
    for (std::size_t i = 0; i < ranked.order.size();) {
        const double threshold = scores[ranked.order[i]];
        do {
            if (y_true[ranked.order[i]] == 1) ++tp; else ++fp;
            ++i;
        } while (i < ranked.order.size() && scores[ranked.order[i]] == threshold);
        result.fpr.push_back(fp / ranked.negatives);
        result.tpr.push_back(tp / ranked.positives);
        result.thresholds.push_back(threshold);
    }
    return result;
}
double roc_auc(const std::vector<int>& y_true, const std::vector<double>& scores) {
    validate_labels(y_true, scores.size());
    struct ScoredLabel { double score; int label; };
    std::vector<ScoredLabel> ranked;
    ranked.reserve(scores.size());
    double positives = 0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (!std::isfinite(scores[i])) throw std::invalid_argument("scores must be finite");
        positives += y_true[i];
        ranked.push_back({scores[i], y_true[i]});
    }
    const double negatives = static_cast<double>(scores.size()) - positives;
    if (positives == 0 || negatives == 0)
        throw std::invalid_argument("both classes are required");
    std::sort(ranked.begin(), ranked.end(),
              [](const ScoredLabel& a, const ScoredLabel& b) { return a.score > b.score; });
    double concordant = 0, negatives_above = 0;
    for (std::size_t i = 0; i < ranked.size();) {
        const double score = ranked[i].score;
        double group_positives = 0, group_negatives = 0;
        do {
            if (ranked[i].label == 1) ++group_positives;
            else ++group_negatives;
            ++i;
        } while (i < ranked.size() && ranked[i].score == score);
        concordant += group_positives * (negatives - negatives_above - group_negatives)
                    + 0.5 * group_positives * group_negatives;
        negatives_above += group_negatives;
    }
    return concordant / (positives * negatives);
}
PrecisionRecallCurve precision_recall_curve(const std::vector<int>& y_true,
                                            const std::vector<double>& scores) {
    const auto ranked = rank_scores(y_true, scores);
    PrecisionRecallCurve result;
    double tp = 0, fp = 0;
    for (std::size_t i = 0; i < ranked.order.size();) {
        const double threshold = scores[ranked.order[i]];
        do {
            if (y_true[ranked.order[i]] == 1) ++tp; else ++fp;
            ++i;
        } while (i < ranked.order.size() && scores[ranked.order[i]] == threshold);
        result.precision.push_back(tp / (tp + fp));
        result.recall.push_back(tp / ranked.positives);
        result.thresholds.push_back(threshold);
    }
    return result;
}
double rmse(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    validate_regression(y_true, y_pred);
    double sum = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double error = y_true[i] - y_pred[i];
        sum += error * error;
    }
    return std::sqrt(sum / y_true.size());
}
double mae(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    validate_regression(y_true, y_pred);
    double sum = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i)
        sum += std::abs(y_true[i] - y_pred[i]);
    return sum / y_true.size();
}
double mape(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    validate_regression(y_true, y_pred);
    double sum = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == 0) throw std::invalid_argument("MAPE is undefined for zero targets");
        sum += std::abs((y_true[i] - y_pred[i]) / y_true[i]);
    }
    return sum / y_true.size();
}
double log_loss(const std::vector<int>& y_true, const std::vector<double>& probabilities) {
    validate_probabilities(y_true, probabilities);
    double sum = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double p = probabilities[i];
        if ((y_true[i] == 1 && p == 0) || (y_true[i] == 0 && p == 1))
            return std::numeric_limits<double>::infinity();
        sum -= y_true[i] == 1 ? std::log(p) : std::log1p(-p);
    }
    return sum / y_true.size();
}
std::vector<std::vector<int>> k_fold_indices(int n_samples, int n_splits, int seed) {
    if (n_samples < 2 || n_splits < 2 || n_splits > n_samples)
        throw std::invalid_argument("require 2 <= n_splits <= n_samples");
    std::vector<int> indices(static_cast<std::size_t>(n_samples));
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::shuffle(indices.begin(), indices.end(), rng);
    std::vector<std::vector<int>> folds(static_cast<std::size_t>(n_splits));
    for (int i = 0; i < n_samples; ++i)
        folds[static_cast<std::size_t>(i % n_splits)].push_back(indices[static_cast<std::size_t>(i)]);
    return folds;
}
CalibrationCurve calibration_curve(const std::vector<int>& y_true,
                                   const std::vector<double>& probabilities, int n_bins) {
    validate_probabilities(y_true, probabilities);
    if (n_bins < 1) throw std::invalid_argument("n_bins must be positive");
    std::vector<double> predicted(static_cast<std::size_t>(n_bins), 0);
    std::vector<double> positive(static_cast<std::size_t>(n_bins), 0);
    std::vector<int> counts(static_cast<std::size_t>(n_bins), 0);
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const auto bin = std::min(n_bins - 1, static_cast<int>(probabilities[i] * n_bins));
        predicted[static_cast<std::size_t>(bin)] += probabilities[i];
        positive[static_cast<std::size_t>(bin)] += y_true[i];
        ++counts[static_cast<std::size_t>(bin)];
    }
    CalibrationCurve result;
    for (int bin = 0; bin < n_bins; ++bin) {
        if (counts[static_cast<std::size_t>(bin)] == 0) continue;
        result.mean_predicted.push_back(predicted[static_cast<std::size_t>(bin)] / counts[static_cast<std::size_t>(bin)]);
        result.fraction_positive.push_back(positive[static_cast<std::size_t>(bin)] / counts[static_cast<std::size_t>(bin)]);
        result.counts.push_back(counts[static_cast<std::size_t>(bin)]);
    }
    return result;
}
}  // namespace mathbr::evaluation
