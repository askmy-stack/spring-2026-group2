"""Tests for src/models/utils/metrics."""
import numpy as np
import pytest
from src.models.utils.metrics import compute_f1_score, compute_auc_roc, compute_sensitivity


def test_f1_score_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert compute_f1_score(y_true, y_pred) == 1.0


def test_f1_score_all_wrong():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    assert compute_f1_score(y_true, y_pred) == 0.0


def test_f1_score_handles_no_positive_predictions():
    y_true = np.array([1, 0, 0])
    y_pred = np.array([0, 0, 0])
    result = compute_f1_score(y_true, y_pred)
    assert isinstance(result, float)


def test_compute_sensitivity_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert compute_sensitivity(y_true, y_pred) == 1.0


def test_compute_sensitivity_missed_all():
    y_true = np.array([1, 1])
    y_pred = np.array([0, 0])
    assert compute_sensitivity(y_true, y_pred) == 0.0
