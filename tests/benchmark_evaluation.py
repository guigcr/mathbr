"""Manual end-to-end classification metric benchmark against NumPy.

Run: python tests/benchmark_evaluation.py
NumPy is used only for a reference and is not a runtime dependency.
"""

from statistics import median
from time import perf_counter

import mathbr
import numpy as np


def elapsed(call, repetitions=15):
    call()
    durations = []
    for _ in range(repetitions):
        start = perf_counter()
        call()
        durations.append((perf_counter() - start) * 1000)
    return median(durations)


def numpy_auc(labels, scores):
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    tp = np.r_[0, np.cumsum(sorted_labels)[ends]]
    fp = np.r_[0, ends + 1 - tp[1:]]
    return np.trapezoid(tp / labels.sum(), fp / (len(labels) - labels.sum()))


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    print(f"NumPy {np.__version__}, mathbr {mathbr.version}")
    print("Rows  mathbr AUC ms  NumPy AUC ms  abs error")
    for n in (100, 10_000, 100_000):
        labels = rng.integers(0, 2, size=n)
        scores = rng.random(n)
        label_list, score_list = labels.tolist(), scores.tolist()
        observed = mathbr.evaluation.roc_auc(label_list, score_list)
        reference = numpy_auc(labels, scores)
        assert abs(observed - reference) < 1e-12
        print(f"{n:>6} {elapsed(lambda: mathbr.evaluation.roc_auc(label_list, score_list)):>14.3f} "
              f"{elapsed(lambda: numpy_auc(labels, scores)):>13.3f} "
              f"{abs(observed-reference):>10.3e}")
