from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0 or y_prob.size == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0, "f1": 0.0, "auroc": float("nan"), "tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    finite_mask = np.isfinite(y_prob)
    if not finite_mask.all():
        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]
        y_prob = y_prob[finite_mask]
        if y_true.size == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "specificity": 0.0, "f1": 0.0, "auroc": float("nan"), "tp": 0.0, "tn": 0.0, "fp": 0.0, "fn": 0.0}
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    metrics = {
        "accuracy": safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "specificity": safe_div(tn, tn + fp),
        "f1": safe_div(2 * precision * recall, precision + recall),
        "auroc": float(roc_auc_score(y_true, y_prob)) if SKLEARN_AVAILABLE and len(np.unique(y_true)) > 1 else float("nan"),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
    return metrics


def smooth_probabilities(y_prob: np.ndarray, mode: str, window: int) -> np.ndarray:
    if mode == "none" or y_prob.size == 0:
        return y_prob
    if mode == "moving_average":
        window = max(1, int(window))
        if window <= 1:
            return y_prob
        kernel = np.ones(window, dtype=float) / window
        return np.convolve(y_prob, kernel, mode="same")
    return y_prob


def smooth_predictions_consecutive(y_pred: np.ndarray, min_run: int) -> np.ndarray:
    if y_pred.size == 0:
        return y_pred
    min_run = max(1, int(min_run))
    if min_run <= 1:
        return y_pred
    smoothed = np.zeros_like(y_pred)
    start = None
    for idx, value in enumerate(y_pred):
        if value == 1 and start is None:
            start = idx
        elif value == 0 and start is not None:
            if idx - start >= min_run:
                smoothed[start:idx] = 1
            start = None
    if start is not None and len(y_pred) - start >= min_run:
        smoothed[start:] = 1
    return smoothed


def compute_metrics_with_smoothing(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    smoothing_mode: str,
    smoothing_window: int,
    min_positive_run: int,
) -> dict[str, float]:
    effective_prob = smooth_probabilities(y_prob, smoothing_mode, smoothing_window)
    y_pred = (effective_prob >= threshold).astype(int)
    if smoothing_mode == "consecutive":
        y_pred = smooth_predictions_consecutive(y_pred, min_positive_run)
    metrics = compute_binary_metrics(y_true, y_pred, effective_prob)
    metrics["threshold"] = float(threshold)
    return metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    smoothing_mode: str,
    smoothing_window: int,
    min_positive_run: int,
) -> tuple[float, dict[str, float]]:
    thresholds = np.round(np.arange(0.05, 0.951, 0.05), 2)
    best_threshold = 0.5
    best_metrics = compute_metrics_with_smoothing(y_true, y_prob, best_threshold, smoothing_mode, smoothing_window, min_positive_run)
    for threshold in thresholds:
        metrics = compute_metrics_with_smoothing(y_true, y_prob, float(threshold), smoothing_mode, smoothing_window, min_positive_run)
        if metrics["f1"] > best_metrics["f1"] or (metrics["f1"] == best_metrics["f1"] and metrics["precision"] > best_metrics["precision"]):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


__all__ = [
    "compute_binary_metrics",
    "compute_metrics_with_smoothing",
    "find_best_threshold",
    "safe_div",
]
