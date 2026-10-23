"""Downstream utility psi (Sec. 4): AUROC, weighted one-vs-rest AUROC, RMSE."""

import numpy as np
from sklearn.metrics import roc_auc_score

HIGHER_IS_BETTER = {"binclass": True, "multiclass": True, "regression": False}
METRIC_NAME = {"binclass": "AUROC", "multiclass": "weighted-AUROC", "regression": "RMSE"}


def utility(task: str, y_true: np.ndarray, pred: np.ndarray) -> float:
    if task == "regression":
        return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(pred)) ** 2)))
    if task == "binclass":
        return float(roc_auc_score(y_true, pred))
    present = np.unique(y_true)  # classes absent from y_true carry zero weight
    p = pred[:, present]
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    if len(present) == 2:
        return float(roc_auc_score(y_true == present[1], p[:, 1]))
    return float(roc_auc_score(y_true, p, multi_class="ovr", average="weighted", labels=present))


def score(task: str, value: float) -> float:
    """Utility oriented so that higher is better (the search maximises it)."""
    return value if HIGHER_IS_BETTER[task] else -value


def delta_percent(task: str, value: float, reference: float) -> float:
    """Sign-corrected relative change (positive = better than the reference)."""
    d = (value - reference) / abs(reference) * 100
    return d if HIGHER_IS_BETTER[task] else -d
