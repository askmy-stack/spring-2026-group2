from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_with_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    return (y_prob >= threshold).astype(np.int32)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_pred = predict_with_threshold(y_prob, threshold)

    metrics = {
        "threshold": float(threshold),
        "aucpr": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def sweep_thresholds_for_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> tuple[float, list[dict]]:
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    rows: list[dict] = []
    best_threshold = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        y_pred = predict_with_threshold(y_prob, float(thr))
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        row = {
            "threshold": float(thr),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }
        rows.append(row)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)

    return best_threshold, rows
