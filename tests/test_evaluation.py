import math

import pytest

import mathbr


def test_binary_metrics():
    e = mathbr.evaluation
    actual = [0, 0, 1, 1]
    predicted = [0, 1, 0, 1]
    assert e.confusion_matrix(actual, predicted) == [[1, 1], [1, 1]]
    assert e.precision(actual, predicted) == 0.5
    assert e.recall(actual, predicted) == 0.5
    assert e.f1_score(actual, predicted) == 0.5
    assert e.precision([1, 0], [0, 0]) == 0
    assert e.recall([0, 0], [0, 0]) == 0
    assert e.f1_score([0, 0], [0, 0]) == 0


def test_roc_and_precision_recall_curves():
    e = mathbr.evaluation
    actual = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    roc = e.roc_curve(actual, scores)
    assert math.isinf(roc.thresholds[0])
    assert roc.fpr == [0, 0, 0.5, 0.5, 1]
    assert roc.tpr == [0, 0.5, 0.5, 1, 1]
    assert e.roc_auc(actual, scores) == pytest.approx(0.75)
    pr = e.precision_recall_curve(actual, scores)
    assert pr.thresholds == [0.8, 0.4, 0.35, 0.1]
    assert pr.precision == pytest.approx([1, 0.5, 2 / 3, 0.5])
    assert pr.recall == [0.5, 0.5, 1, 1]


def test_tied_scores_are_grouped_before_rates():
    e = mathbr.evaluation
    assert e.roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    roc = e.roc_curve([0, 1], [0.5, 0.5])
    assert roc.fpr == [0, 1]
    assert roc.tpr == [0, 1]
    pr = e.precision_recall_curve([0, 1], [0.5, 0.5])
    assert pr.precision == [0.5]
    assert pr.recall == [1]


@pytest.mark.parametrize("actual,predicted", [([], []), ([0], []), ([2], [0]), ([0], [2])])
def test_invalid_binary_labels(actual, predicted):
    with pytest.raises(ValueError):
        mathbr.evaluation.confusion_matrix(actual, predicted)


@pytest.mark.parametrize("actual,scores", [([0, 0], [0.1, 0.2]),
                                               ([0, 1], [0.1, math.nan]),
                                               ([0, 1], [0.1]),
                                               ([0, 2], [0.1, 0.2])])
def test_invalid_curve_inputs(actual, scores):
    with pytest.raises(ValueError):
        mathbr.evaluation.roc_curve(actual, scores)
    with pytest.raises(ValueError):
        mathbr.evaluation.precision_recall_curve(actual, scores)


def test_out_of_sample_metrics_and_extreme_probabilities():
    e = mathbr.evaluation
    actual, predicted = [2.0, 4.0], [1.0, 6.0]
    assert e.rmse(actual, predicted) == pytest.approx(math.sqrt(2.5))
    assert e.mae(actual, predicted) == 1.5
    assert e.mape(actual, predicted) == 0.5
    assert e.log_loss([0, 1], [0.2, 0.8]) == pytest.approx(-math.log(0.8))
    assert e.log_loss([0, 1], [0.0, 1.0]) == 0
    assert math.isinf(e.log_loss([1], [0.0]))
    with pytest.raises(ValueError):
        e.mape([0.0], [1.0])
    with pytest.raises(ValueError):
        e.rmse([1.0], [math.inf])
    with pytest.raises(ValueError):
        e.log_loss([1], [1.1])


def test_kfold_reproducibility_and_coverage():
    e = mathbr.evaluation
    folds = e.k_fold_indices(11, 3, seed=42)
    assert folds == e.k_fold_indices(11, 3, seed=42)
    assert sorted(index for fold in folds for index in fold) == list(range(11))
    assert sorted(map(len, folds)) == [3, 4, 4]
    with pytest.raises(ValueError):
        e.k_fold_indices(3, 4)


def test_calibration_curve_bin_boundaries():
    e = mathbr.evaluation
    result = e.calibration_curve([0, 1, 1, 0], [0.0, 0.2, 0.5, 1.0], n_bins=2)
    assert result.mean_predicted == pytest.approx([0.1, 0.75])
    assert result.fraction_positive == [0.5, 0.5]
    assert result.counts == [2, 2]
    with pytest.raises(ValueError):
        e.calibration_curve([0], [0.5], n_bins=0)
