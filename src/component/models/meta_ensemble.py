"""
Cross-family Meta Ensemble.

Combines the per-family ensemble outputs (LSTM, Mamba, HF) into a single
decision by operating directly on the cached probability arrays written by
each family's ensemble script. No model reloading, no GPU needed — this is a
pure numpy combine step run on the bundled artifacts::

    <family_dir>/
        ensemble_metrics.json   # read for the val F1 weighting
        val_probs.npy           # (n_val,)
        val_labels.npy          # (n_val,) — identical across families
        test_probs.npy          # (n_test,)
        test_labels.npy         # (n_test,) — identical across families

Usage::

    python -m src.models.meta_ensemble \\
        --family_dir outputs/ensemble/lstm \\
        --family_dir outputs/ensemble/mamba \\
        --family_dir outputs/ensemble/hf \\
        --strategy weighted \\
        --out_dir outputs/ensemble/meta

Supported strategies:

- ``mean``:               equal-weighted average of family probs.
- ``weighted``:           weighted by each family's val F1 (read from
                          ``ensemble_metrics.json``'s ``results`` map if
                          present, else falls back to ``f1`` at the top
                          level, else 1.0).
- ``rank_average``:       average of rank-normalised probabilities (robust to
                          miscalibrated individual families).
- ``logistic_stacking``:  fits a scalar logistic regression on val probs,
                          applies it to test probs. Falls back to ``mean``
                          if scikit-learn is unavailable.

Rationale for the two-tier design is documented in
``.windsurf/plans/boost-metrics-and-meta-ensemble-c3a10f.md``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.models.utils.metrics import (
    compute_f1_score, compute_auc_roc, compute_sensitivity,
    compute_specificity, find_optimal_threshold,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--family_dir", action="append", required=True, type=Path,
        help="Path to a family ensemble output dir. Pass multiple times.",
    )
    parser.add_argument(
        "--strategy", default="weighted",
        choices=("mean", "weighted", "rank_average", "logistic_stacking"),
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("outputs/ensemble/meta"),
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Decision threshold (0 = tune on val; otherwise fixed).",
    )
    return parser.parse_args()


def main() -> None:
    """Read family artifacts, combine, write meta metrics."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    families = _load_families(args.family_dir)
    if not families:
        logger.error("No valid family dirs among %s; aborting.", args.family_dir)
        return

    labels_val, labels_test = _shared_labels(families)
    val_stack = np.stack([f["val_probs"] for f in families])   # (F, N_val)
    test_stack = np.stack([f["test_probs"] for f in families]) # (F, N_test)
    val_f1s = np.array([f["val_f1"] for f in families], dtype=np.float64)

    val_ens, test_ens = combine(args.strategy, val_stack, test_stack, val_f1s, labels_val)

    threshold = (
        float(args.threshold) if args.threshold > 0
        else float(find_optimal_threshold(labels_val, val_ens))
    )
    metrics = evaluate(labels_test, test_ens, threshold)

    logger.info("=" * 60)
    logger.info("META ENSEMBLE — %s (%d families):", args.strategy, len(families))
    for f, w in zip(families, val_f1s):
        logger.info("  %-30s val_f1=%.4f", f["name"], w)
    logger.info("  F1=%.4f  AUROC=%.4f  sens=%.4f  spec=%.4f  t=%.3f",
                metrics["f1"], metrics["auroc"], metrics["sens"],
                metrics["spec"], threshold)
    logger.info("=" * 60)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "meta_metrics.json", "w") as fh:
        json.dump({
            **metrics,
            "threshold": threshold,
            "strategy": args.strategy,
            "families": [f["name"] for f in families],
            "val_f1_weights": val_f1s.tolist(),
        }, fh, indent=2)
    np.save(args.out_dir / "val_probs.npy", val_ens)
    np.save(args.out_dir / "test_probs.npy", test_ens)
    logger.info("Wrote meta artifacts -> %s", args.out_dir)


# ---------------------------------------------------------------------------
# Core library functions (importable for tests)
# ---------------------------------------------------------------------------


def combine(
    strategy: str, val_stack: np.ndarray, test_stack: np.ndarray,
    val_f1s: np.ndarray, labels_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply ``strategy`` to val + test probability stacks.

    Args:
        val_stack:  shape (F, N_val) — per-family val probs, one row each.
        test_stack: shape (F, N_test).
        val_f1s:    shape (F,) — per-family val F1, used by ``weighted``.
        labels_val: shape (N_val,) — used by ``logistic_stacking`` to fit.

    Returns:
        ``(val_ensemble, test_ensemble)`` as 1-D arrays in [0, 1].
    """
    if strategy == "mean":
        return val_stack.mean(axis=0), test_stack.mean(axis=0)
    if strategy == "weighted":
        weights = val_f1s / val_f1s.sum() if val_f1s.sum() > 0 else None
        if weights is None:
            return val_stack.mean(axis=0), test_stack.mean(axis=0)
        return (
            np.average(val_stack, axis=0, weights=weights),
            np.average(test_stack, axis=0, weights=weights),
        )
    if strategy == "rank_average":
        return _rank_average(val_stack), _rank_average(test_stack)
    if strategy == "logistic_stacking":
        return _logistic_stacking(val_stack, test_stack, labels_val)
    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute the standard F1/AUROC/sens/spec quartet at ``threshold``."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "f1": float(compute_f1_score(y_true, y_pred)),
        "auroc": float(compute_auc_roc(y_true, y_prob)),
        "sens": float(compute_sensitivity(y_true, y_pred)),
        "spec": float(compute_specificity(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_families(dirs: List[Path]) -> List[Dict]:
    """Read each family dir's npy + json; skip dirs that are missing artifacts."""
    out: List[Dict] = []
    for d in dirs:
        val_p = d / "val_probs.npy"
        test_p = d / "test_probs.npy"
        val_l = d / "val_labels.npy"
        test_l = d / "test_labels.npy"
        if not all(p.exists() for p in (val_p, test_p, val_l, test_l)):
            logger.warning("Skip %s: missing one of val/test probs/labels.npy", d)
            continue
        val_f1 = _read_val_f1(d / "ensemble_metrics.json")
        out.append({
            "name": d.name,
            "dir": d,
            "val_probs": np.load(val_p),
            "test_probs": np.load(test_p),
            "val_labels": np.load(val_l),
            "test_labels": np.load(test_l),
            "val_f1": val_f1,
        })
    return out


def _read_val_f1(path: Path) -> float:
    """Read val F1 from a family's metrics JSON. Supports both the nested
    (``results.weighted.f1``) and flat (``f1``) conventions used by the
    improved-LSTM and HF/Mamba ensemble writers respectively.
    """
    if not path.exists():
        return 1.0
    try:
        with open(path) as fh:
            blob = json.load(fh)
    except json.JSONDecodeError:
        logger.warning("Could not parse %s; weighting this family as 1.0.", path)
        return 1.0
    if "results" in blob and isinstance(blob["results"], dict):
        # improved_lstm_models/ensemble.py style: results.<strategy>.f1
        for strat in ("weighted", "mean"):
            block = blob["results"].get(strat)
            if isinstance(block, dict) and "f1" in block:
                return float(block["f1"])
    return float(blob.get("f1", 1.0))


def _shared_labels(families: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (val_labels, test_labels) after asserting all families agree.

    Mismatched labels imply the families were evaluated on different splits,
    which would silently corrupt the ensemble (off-by-one at best, completely
    wrong at worst). Fail loud instead of combining them.
    """
    base_val = families[0]["val_labels"]
    base_test = families[0]["test_labels"]
    for f in families[1:]:
        if not np.array_equal(f["val_labels"], base_val):
            raise ValueError(
                f"Family '{f['name']}' val labels disagree with '{families[0]['name']}'. "
                "Re-run each family on the same val split before meta-ensembling."
            )
        if not np.array_equal(f["test_labels"], base_test):
            raise ValueError(
                f"Family '{f['name']}' test labels disagree with '{families[0]['name']}'."
            )
    return base_val, base_test


def _rank_average(stack: np.ndarray) -> np.ndarray:
    """Average of per-family rank-normalised probabilities.

    Robust to calibration differences (e.g. one family's probabilities are
    concentrated near 0.5 while another's saturate to 0/1). Ranks are mapped
    to [0, 1] via ``(r - 1) / (N - 1)``.
    """
    n = stack.shape[0]          # number of families
    if n < 2:
        # Single family: rank-normalise across the samples dimension instead.
        m = stack.shape[1]
        if m < 2:
            return stack.mean(axis=0)
        ranks = np.argsort(np.argsort(stack, axis=1), axis=1).astype(np.float64)
        return (ranks / (m - 1)).squeeze(0)
    # Multiple families: rank each sample's probability across the family axis.
    ranks = np.argsort(np.argsort(stack, axis=0), axis=0).astype(np.float64)
    return (ranks / (n - 1)).mean(axis=0)


def _logistic_stacking(
    val_stack: np.ndarray, test_stack: np.ndarray, labels_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a logistic regression on val probs, apply it on test.

    Columns are family probabilities. Uses scikit-learn if available; falls
    back to mean-averaging to avoid adding a hard runtime dependency.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        logger.warning("sklearn unavailable; logistic_stacking falls back to mean.")
        return val_stack.mean(axis=0), test_stack.mean(axis=0)
    x_val = val_stack.T   # (N_val, F)
    x_test = test_stack.T # (N_test, F)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(x_val, labels_val)
    val_prob = clf.predict_proba(x_val)[:, 1]
    test_prob = clf.predict_proba(x_test)[:, 1]
    return val_prob, test_prob


if __name__ == "__main__":
    main()
