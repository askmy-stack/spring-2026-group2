"""
Tests for :mod:`src.models.meta_ensemble`.

Covers the pure numpy combine step for all four strategies plus the
label-mismatch safety check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.meta_ensemble import (
    combine, evaluate, _load_families, _rank_average, _shared_labels,
)


# ---------------------------------------------------------------------------
# combine() — per-strategy numerics
# ---------------------------------------------------------------------------


def _toy_probs():
    """Three families, five samples. Designed so the mean gives ~0.5 for all."""
    val = np.array([
        [0.9, 0.8, 0.1, 0.2, 0.7],   # family A: strong positives on [0,1,4]
        [0.7, 0.6, 0.3, 0.1, 0.8],   # family B: similar but weaker
        [0.4, 0.5, 0.5, 0.5, 0.6],   # family C: near-random
    ])
    test = val.copy()  # identical test stack for simplicity
    return val, test


def test_combine_mean_is_simple_average():
    val, test = _toy_probs()
    val_ens, test_ens = combine("mean", val, test, np.array([0.6, 0.5, 0.4]), np.zeros(5, dtype=int))
    np.testing.assert_allclose(val_ens, val.mean(axis=0))
    np.testing.assert_allclose(test_ens, test.mean(axis=0))


def test_combine_weighted_prefers_high_f1_family():
    val, test = _toy_probs()
    # Family A weight 0.9, others 0.05 each — result should be close to family A.
    weights = np.array([0.9, 0.05, 0.05])
    val_ens, _ = combine("weighted", val, test, weights, np.zeros(5, dtype=int))
    np.testing.assert_allclose(val_ens, np.average(val, axis=0, weights=weights), rtol=1e-6)


def test_combine_weighted_with_zero_weights_falls_back_to_mean():
    val, test = _toy_probs()
    weights = np.zeros(3)
    val_ens, _ = combine("weighted", val, test, weights, np.zeros(5, dtype=int))
    np.testing.assert_allclose(val_ens, val.mean(axis=0))


def test_combine_rank_average_is_in_unit_interval():
    val, test = _toy_probs()
    val_ens, test_ens = combine("rank_average", val, test, np.ones(3), np.zeros(5, dtype=int))
    assert np.all((val_ens >= 0.0) & (val_ens <= 1.0))
    assert np.all((test_ens >= 0.0) & (test_ens <= 1.0))


def test_rank_average_preserves_order_of_probs():
    # With one family, rank-average is just a rank-normalised version of the
    # input. Order must be preserved.
    stack = np.array([[0.3, 0.9, 0.1, 0.5]])
    out = _rank_average(stack)
    # rank of 0.3=1, 0.9=3, 0.1=0, 0.5=2 ; normalised by (N-1)=3
    np.testing.assert_allclose(out, np.array([1/3, 1.0, 0.0, 2/3]))


def test_combine_unknown_strategy_raises():
    val, test = _toy_probs()
    with pytest.raises(ValueError, match="Unknown strategy"):
        combine("bogus", val, test, np.ones(3), np.zeros(5, dtype=int))


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def test_evaluate_perfect_classifier():
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    m = evaluate(y_true, y_prob, threshold=0.5)
    assert m["f1"] == pytest.approx(1.0)
    assert m["sens"] == pytest.approx(1.0)
    assert m["spec"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _load_families + _shared_labels
# ---------------------------------------------------------------------------


def _write_family(d: Path, val_probs, test_probs, val_labels, test_labels, f1: float = 0.6) -> None:
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "val_probs.npy", val_probs)
    np.save(d / "test_probs.npy", test_probs)
    np.save(d / "val_labels.npy", val_labels)
    np.save(d / "test_labels.npy", test_labels)
    with open(d / "ensemble_metrics.json", "w") as fh:
        json.dump({"f1": f1}, fh)


def test_shared_labels_raises_on_mismatch(tmp_path):
    v = np.array([0, 1, 0])
    t = np.array([1, 0])
    _write_family(tmp_path / "fam_a", [0.1, 0.9, 0.2], [0.7, 0.3], v, t)
    _write_family(tmp_path / "fam_b", [0.2, 0.8, 0.1], [0.6, 0.4], v, t[::-1])  # scrambled test labels

    families = _load_families([tmp_path / "fam_a", tmp_path / "fam_b"])
    with pytest.raises(ValueError, match="test labels disagree"):
        _shared_labels(families)


def test_load_families_skips_missing_artifacts(tmp_path, caplog):
    v = np.array([0, 1, 0])
    t = np.array([1, 0])
    _write_family(tmp_path / "good", [0.1, 0.9, 0.2], [0.7, 0.3], v, t)
    (tmp_path / "broken").mkdir()  # no npy files
    families = _load_families([tmp_path / "good", tmp_path / "broken"])
    assert [f["name"] for f in families] == ["good"]


def test_load_families_reads_nested_results_f1(tmp_path):
    """improved_lstm_models/ensemble.py writes results.weighted.f1 \u2014 support it."""
    v = np.array([0, 1, 0])
    t = np.array([1, 0])
    d = tmp_path / "fam"
    _write_family(d, [0.1, 0.9, 0.2], [0.7, 0.3], v, t)
    with open(d / "ensemble_metrics.json", "w") as fh:
        json.dump({"results": {"weighted": {"f1": 0.777}}}, fh)
    families = _load_families([d])
    assert families[0]["val_f1"] == pytest.approx(0.777)
