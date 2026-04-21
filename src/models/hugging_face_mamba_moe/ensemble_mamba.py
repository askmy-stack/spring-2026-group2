"""
Mamba family ensemble — combines ``eeg_mamba`` and ``eeg_mamba_moe`` on test.

Mirrors the structure of ``ensemble_hf.py`` but unpacks the ``(logits, lb_loss)``
tuple returned by ``eeg_mamba_moe``'s forward so both members write sigmoid
probabilities in the same shape.

Outputs (in ``--out_dir``):
    ensemble_metrics.json
    val_probs.npy / val_labels.npy
    test_probs.npy / test_labels.npy

The ``*_probs.npy`` files feed the cross-family meta ensemble at
:mod:`src.models.meta_ensemble`.

Usage::

    python -m src.models.hugging_face_mamba_moe.ensemble_mamba \\
        --data_path src/results/tensors/chbmit \\
        --ckpt_dir outputs/models \\
        --out_dir outputs/ensemble/mamba
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.utils.checkpoint import load_checkpoint
from src.models.utils.metrics import (
    compute_f1_score, compute_auc_roc, compute_sensitivity,
    compute_specificity, find_optimal_threshold,
)

logger = logging.getLogger(__name__)

MAMBA_MEMBERS = ("eeg_mamba", "eeg_mamba_moe")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble the Mamba family on test.")
    parser.add_argument("--data_path", required=True, type=Path,
                        help="Tensor splits dir (train/val/test with data.pt + labels.pt).")
    parser.add_argument("--ckpt_dir", type=Path, default=Path("outputs/models"),
                        help="Dir holding ``eeg_mamba_best.pt`` / ``eeg_mamba_moe_best.pt``.")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/ensemble/mamba"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--strategy", default="weighted", choices=("mean", "weighted"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    """CLI entry point: load both Mamba members, combine, write metrics + probs."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    members = _load_members(args.ckpt_dir, device)
    if not members:
        logger.error("No Mamba checkpoints found under %s; nothing to ensemble.", args.ckpt_dir)
        return
    logger.info("Loaded %d Mamba members: %s", len(members), [m["name"] for m in members])

    val_dl, test_dl = _build_dataloaders(args.data_path, args.batch_size)
    val_labels = _collect_labels(val_dl)
    test_labels = _collect_labels(test_dl)

    val_probs_per_model: List[np.ndarray] = []
    test_probs_per_model: List[np.ndarray] = []
    weights: List[float] = []
    for info in members:
        logger.info("Predicting: %s", info["name"])
        val_probs_per_model.append(_predict(info["model"], info["name"], val_dl, device))
        test_probs_per_model.append(_predict(info["model"], info["name"], test_dl, device))
        # Weight by val F1 — same convention as the LSTM family ensemble.
        weights.append(max(info["val_f1"], 1e-3))

    weights_arr = np.asarray(weights, dtype=np.float64)
    if args.strategy == "weighted" and weights_arr.sum() > 0:
        val_ens = np.average(np.stack(val_probs_per_model), axis=0, weights=weights_arr)
        test_ens = np.average(np.stack(test_probs_per_model), axis=0, weights=weights_arr)
    else:
        val_ens = np.stack(val_probs_per_model).mean(axis=0)
        test_ens = np.stack(test_probs_per_model).mean(axis=0)

    threshold = float(find_optimal_threshold(val_labels, val_ens))
    metrics = _compute_metrics(test_labels, test_ens, threshold)
    metrics.update(threshold=threshold, strategy=args.strategy,
                   members=[m["name"] for m in members])

    logger.info("=" * 60)
    logger.info("MAMBA ENSEMBLE test — %s (%d members):", args.strategy, len(members))
    logger.info("  F1 %.4f  AUROC %.4f  sens %.4f  spec %.4f  t=%.3f",
                metrics["f1"], metrics["auroc"], metrics["sens"],
                metrics["spec"], threshold)
    logger.info("=" * 60)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "ensemble_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    np.save(args.out_dir / "val_probs.npy", val_ens)
    np.save(args.out_dir / "val_labels.npy", val_labels)
    np.save(args.out_dir / "test_probs.npy", test_ens)
    np.save(args.out_dir / "test_labels.npy", test_labels)
    logger.info("Wrote family artifacts -> %s", args.out_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_members(ckpt_dir: Path, device: torch.device) -> List[Dict]:
    """Rebuild each Mamba checkpoint if present; skip missing ones gracefully."""
    out: List[Dict] = []
    for name in MAMBA_MEMBERS:
        path = ckpt_dir / f"{name}_best.pt"
        if not path.exists():
            logger.info("Skip %s: no checkpoint at %s", name, path)
            continue
        try:
            model, payload = load_checkpoint(path, map_location=device, build_model=True)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("Skip %s: %s", name, exc)
            continue
        if model is None:
            logger.warning("Skip %s: auto-rebuild failed (see load_checkpoint warnings).", name)
            continue
        model = model.to(device).eval()
        out.append({
            "name": name,
            "model": model,
            "val_f1": float(payload.get("val_metrics", {}).get("f1", 0.0)),
            "threshold": float(payload.get("optimal_threshold", 0.5)),
        })
    return out


def _build_dataloaders(data_path: Path, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Load val + test splits; shuffle=False to keep label/prob order stable."""
    pin = torch.cuda.is_available()

    def _split(name: str) -> TensorDataset:
        d = torch.load(data_path / name / "data.pt", weights_only=True).float()
        l = torch.load(data_path / name / "labels.pt", weights_only=True).long().squeeze()
        return TensorDataset(d, l)

    return (
        DataLoader(_split("val"), batch_size=batch_size, shuffle=False, pin_memory=pin),
        DataLoader(_split("test"), batch_size=batch_size, shuffle=False, pin_memory=pin),
    )


def _predict(
    model: nn.Module, model_name: str, loader: DataLoader, device: torch.device,
) -> np.ndarray:
    """Return positive-class probs; unpack tuple output for eeg_mamba_moe."""
    is_moe = model_name == "eeg_mamba_moe"
    probs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            logits = out[0] if is_moe else out
            p = torch.sigmoid(logits.squeeze(-1)).detach().cpu().numpy()
            probs.append(np.atleast_1d(p))
    return np.concatenate(probs)


def _collect_labels(loader: DataLoader) -> np.ndarray:
    """Concatenate labels across a DataLoader into a flat 1-D array."""
    out: List[np.ndarray] = []
    for _, y in loader:
        out.append(np.atleast_1d(y.numpy()))
    return np.concatenate(out)


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict:
    """Compute the standard F1/AUROC/sens/spec quartet at ``threshold``."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "f1": float(compute_f1_score(y_true, y_pred)),
        "auroc": float(compute_auc_roc(y_true, y_prob)),
        "sens": float(compute_sensitivity(y_true, y_pred)),
        "spec": float(compute_specificity(y_true, y_pred)),
    }


if __name__ == "__main__":
    main()
