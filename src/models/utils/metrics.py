"""
Shared evaluation metrics for EEG seizure detection.

All training and evaluation scripts import from here. Never define metrics locally.
"""
import logging
from typing import Optional

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

logger = logging.getLogger(__name__)


def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute binary F1-score.

    Args:
        y_true: Ground-truth binary labels, shape (n,)
        y_pred: Predicted binary labels, shape (n,)

    Returns:
        F1-score in [0, 1]. Returns 0.0 if no positive predictions or empty input.

    Example:
        >>> compute_f1_score(np.array([1,0,1]), np.array([1,0,0]))
        0.6666...
    """
    if len(y_true) == 0:
        logger.warning("Empty y_true in compute_f1_score; returning 0.0")
        return 0.0
    return float(f1_score(y_true, y_pred, zero_division=0))


def compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AUC-ROC score.

    Args:
        y_true: Ground-truth binary labels, shape (n,)
        y_score: Predicted probabilities for positive class, shape (n,)

    Returns:
        AUC-ROC in [0, 1]. Returns 0.5 if only one class present.

    Raises:
        ValueError: If y_true and y_score have different lengths.

    Example:
        >>> compute_auc_roc(np.array([1,0,1]), np.array([0.9,0.1,0.8]))
        1.0
    """
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class in y_true; AUC-ROC undefined, returning 0.5")
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def compute_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute sensitivity (recall for the positive/seizure class).

    Args:
        y_true: Ground-truth binary labels, shape (n,)
        y_pred: Predicted binary labels, shape (n,)

    Returns:
        Sensitivity in [0, 1]. Returns 0.0 if no true positives exist.

    Example:
        >>> compute_sensitivity(np.array([1,1,0]), np.array([1,0,0]))
        0.5
    """
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_positives = conf_matrix[1, 1]
    false_negatives = conf_matrix[1, 0]
    denominator = true_positives + false_negatives
    if denominator == 0:
        return 0.0
    return float(true_positives / denominator)


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute specificity (true negative rate).

    Args:
        y_true: Ground-truth binary labels, shape (n,)
        y_pred: Predicted binary labels, shape (n,)

    Returns:
        Specificity in [0, 1]. Returns 0.0 if no true negatives exist.

    Example:
        >>> compute_specificity(np.array([0,0,1]), np.array([0,1,1]))
        0.5
    """
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negatives = conf_matrix[0, 0]
    false_positives = conf_matrix[0, 1]
    denominator = true_negatives + false_positives
    if denominator == 0:
        return 0.0
    return float(true_negatives / denominator)


def find_optimal_threshold(
    y_true: np.ndarray, y_score: np.ndarray, objective: str = "f1",
) -> float:
    """
    Find classification threshold maximising the given objective on the provided split.

    Args:
        y_true:    Ground-truth binary labels, shape (n,)
        y_score:   Predicted probabilities,    shape (n,)
        objective: 'f1' (default) or 'youden'. F1 is more robust than
                   Youden's J when the val → test distribution shifts, which
                   is typical in subject-independent EEG splits.

    Returns:
        Optimal threshold in [0.01, 0.99].
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_score = -1.0
    best_thresh = 0.5
    for thresh in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_score >= thresh).astype(int)
        if objective == "youden":
            score = compute_sensitivity(y_true, y_pred) + compute_specificity(y_true, y_pred) - 1.0
        else:  # 'f1'
            score = compute_f1_score(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)
    return best_thresh
