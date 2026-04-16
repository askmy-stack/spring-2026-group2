"""
Ensemble evaluator for trained HF EEG models.

Loads best checkpoints from all trained models, generates soft predictions
on the test set, and combines them via averaging or weighted voting.

Usage:
    python -m src.models.hugging_face_mamba_moe.ensemble_hf \
        --data_path src/results/tensors/chbmit
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .architectures.hf_factory import create_hf_model, list_hf_models

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble HF EEG model predictions on test set.")
    parser.add_argument("--data_path", required=True, help="Path to tensor splits (train/val/test).")
    parser.add_argument("--results_dir", default="", help="Override results root (default: auto-detect).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--strategy", default="mean", choices=("mean", "weighted"),
                        help="Ensemble strategy: 'mean' = equal averaging, 'weighted' = weight by val AUROC.")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Decision threshold for ensemble. 0 = auto-tune on val set.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    # Locate results root
    src_dir = Path(__file__).parent.parent.parent
    results_root = Path(args.results_dir) if args.results_dir else src_dir.parent / "results" / "model"

    # Load test data
    data_path = Path(args.data_path)
    test_dl, val_dl, channels, samples = _load_data(data_path, args.batch_size)

    # Discover trained models
    models_info = _discover_models(results_root, channels, samples, device)
    if not models_info:
        logger.error("No trained model checkpoints found in %s", results_root)
        return
    logger.info("Found %d trained models: %s", len(models_info), [m["name"] for m in models_info])

    # Collect predictions on val + test
    val_probs_all, test_probs_all, weights = [], [], []
    val_true = _collect_labels(val_dl)
    test_true = _collect_labels(test_dl)

    for info in models_info:
        name, model = info["name"], info["model"]
        logger.info("Generating predictions: %s", name)
        val_probs = _predict(model, val_dl, device)
        test_probs = _predict(model, test_dl, device)
        val_probs_all.append(val_probs)
        test_probs_all.append(test_probs)

        # Read val AUROC for weighting
        auroc = info.get("val_auroc", 0.5)
        weights.append(max(auroc - 0.5, 0.01))  # weight = excess AUROC above random

    # Ensemble
    val_probs_stack = np.stack(val_probs_all)    # (n_models, n_samples)
    test_probs_stack = np.stack(test_probs_all)

    if args.strategy == "weighted":
        w = np.array(weights) / np.sum(weights)
        logger.info("Model weights: %s", {m["name"]: f"{wi:.3f}" for m, wi in zip(models_info, w)})
        val_ensemble = np.average(val_probs_stack, axis=0, weights=w)
        test_ensemble = np.average(test_probs_stack, axis=0, weights=w)
    else:
        val_ensemble = val_probs_stack.mean(axis=0)
        test_ensemble = test_probs_stack.mean(axis=0)

    # Tune threshold on val set
    if args.threshold > 0:
        best_t = args.threshold
    else:
        best_t = _tune_threshold(val_true, val_ensemble)
    logger.info("Using threshold: %.2f", best_t)

    # Final test metrics
    test_metrics = _compute_metrics(test_true, test_ensemble, best_t)
    logger.info("=" * 60)
    logger.info("ENSEMBLE Test Results (%d models, strategy=%s):", len(models_info), args.strategy)
    logger.info("  F1:        %.4f", test_metrics["f1"])
    logger.info("  Precision: %.4f", test_metrics["precision"])
    logger.info("  Recall:    %.4f", test_metrics["recall"])
    logger.info("  AUROC:     %.4f", test_metrics["auroc"])
    logger.info("  Threshold: %.2f", best_t)
    logger.info("  TP=%d  TN=%d  FP=%d  FN=%d",
                test_metrics["tp"], test_metrics["tn"], test_metrics["fp"], test_metrics["fn"])
    logger.info("=" * 60)

    # Also log individual model test metrics for comparison
    logger.info("\nIndividual model test results:")
    for info, probs in zip(models_info, test_probs_all):
        m = _compute_metrics(test_true, probs, best_t)
        logger.info("  %-30s F1=%.4f  AUROC=%.4f  Recall=%.4f", info["name"], m["f1"], m["auroc"], m["recall"])

    # Save ensemble results
    out_dir = results_root / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ensemble_metrics.json", "w") as f:
        json.dump({**test_metrics, "threshold": best_t, "strategy": args.strategy,
                   "models": [m["name"] for m in models_info]}, f, indent=2)
    logger.info("Saved to %s", out_dir / "ensemble_metrics.json")


def _load_data(data_path: Path, batch_size: int):
    """Load val and test splits."""
    pin = torch.cuda.is_available()

    def _split(name: str) -> TensorDataset:
        d = torch.load(data_path / name / "data.pt", weights_only=True).float()
        l = torch.load(data_path / name / "labels.pt", weights_only=True).long().squeeze()
        return TensorDataset(d, l)

    val_ds = _split("val")
    test_ds = _split("test")
    channels = val_ds.tensors[0].shape[1]
    samples = val_ds.tensors[0].shape[2]
    return (
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin),
        channels,
        samples,
    )


def _discover_models(results_root: Path, channels: int, samples: int,
                     device: torch.device) -> list[dict[str, Any]]:
    """Scan results_root for trained model checkpoints and load them."""
    models_info = []
    for model_name in list_hf_models():
        ckpt = results_root / model_name / "checkpoints" / "best_model.pt"
        if not ckpt.exists():
            continue
        try:
            model = create_hf_model(
                model_name, in_channels=channels, num_classes=2,
                n_times=samples, sfreq=256, dropout=0.0,
            )
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            model.to(device).eval()

            # Try to read val AUROC from logs
            val_auroc = _read_best_val_auroc(results_root / model_name / "logs")

            models_info.append({"name": model_name, "model": model, "val_auroc": val_auroc})
            logger.info("Loaded %s (val_auroc=%.4f)", model_name, val_auroc)
        except Exception as e:
            logger.warning("Skipping %s: %s", model_name, e)
    return models_info


def _read_best_val_auroc(logs_dir: Path) -> float:
    """Read best val AUROC from training history."""
    try:
        import csv
        history_path = logs_dir / "history.csv"
        if not history_path.exists():
            return 0.5
        with open(history_path) as f:
            reader = csv.DictReader(f)
            best = 0.5
            for row in reader:
                auroc = float(row.get("val_auroc", 0.5))
                if auroc > best:
                    best = auroc
            return best
    except Exception:
        return 0.5


def _predict(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    """Generate probability predictions for positive class."""
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.append(p.cpu().numpy())
    return np.concatenate(probs)


def _collect_labels(dataloader: DataLoader) -> np.ndarray:
    """Collect all labels from a dataloader."""
    labels = []
    for _, y in dataloader:
        labels.append(y.numpy())
    return np.concatenate(labels)


def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 on validation set."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute binary classification metrics."""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = float("nan")
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc": auroc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


if __name__ == "__main__":
    main()
