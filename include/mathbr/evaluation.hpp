#pragma once

#include <vector>

namespace mathbr::evaluation {

struct RocCurve {
    std::vector<double> fpr;
    std::vector<double> tpr;
    std::vector<double> thresholds;
};

struct PrecisionRecallCurve {
    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> thresholds;
};

struct CalibrationCurve {
    std::vector<double> mean_predicted;
    std::vector<double> fraction_positive;
    std::vector<int> counts;
};

std::vector<std::vector<int>> confusion_matrix(const std::vector<int>& y_true,
                                                const std::vector<int>& y_pred);
double precision(const std::vector<int>& y_true, const std::vector<int>& y_pred);
double recall(const std::vector<int>& y_true, const std::vector<int>& y_pred);
double f1_score(const std::vector<int>& y_true, const std::vector<int>& y_pred);
RocCurve roc_curve(const std::vector<int>& y_true, const std::vector<double>& scores);
double roc_auc(const std::vector<int>& y_true, const std::vector<double>& scores);
PrecisionRecallCurve precision_recall_curve(const std::vector<int>& y_true,
                                            const std::vector<double>& scores);
double rmse(const std::vector<double>& y_true, const std::vector<double>& y_pred);
double mae(const std::vector<double>& y_true, const std::vector<double>& y_pred);
double mape(const std::vector<double>& y_true, const std::vector<double>& y_pred);
double log_loss(const std::vector<int>& y_true, const std::vector<double>& probabilities);
std::vector<std::vector<int>> k_fold_indices(int n_samples, int n_splits, int seed = 0);
CalibrationCurve calibration_curve(const std::vector<int>& y_true,
                                   const std::vector<double>& probabilities, int n_bins = 10);

}  // namespace mathbr::evaluation
