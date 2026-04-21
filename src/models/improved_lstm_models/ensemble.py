"""
Ensemble Prediction for Improved LSTM Models.

Combines multiple model predictions via averaging, weighted averaging,
voting, max probability, or stacking.

Import from here — never define ensemble logic inline in a training script.

CLI usage (auto-discovers every unified-schema .pt in the given dirs;
pass multiple ``--ckpt_dir`` to combine benchmark + improved checkpoints):

    python -m src.models.improved_lstm_models.ensemble \\
        --data_path src/results/tensors/chbmit \\
        --ckpt_dir src/models/lstm_benchmark_models/checkpoints \\
                    src/models/improved_lstm_models/checkpoints

Reports F1 / AUROC / sens / spec for both ``average`` and ``weighted`` voting
strategies with threshold tuned (for F1) on the validation split. Handles
both 3-D (m1..m7 benchmarks) and 4-D (HierarchicalLSTM) input contracts.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EnsemblePredictor(nn.Module):
    """
    Ensemble predictor combining multiple trained LSTM models.

    Args:
        models: List of trained nn.Module instances.
        weights: Optional per-model weights for 'weighted' strategy.
        strategy: One of 'average', 'weighted', 'voting', 'max', 'stacking'.
        threshold: Classification threshold for 'voting' strategy (default: 0.5).

    Example:
        >>> ens = EnsemblePredictor([model_a, model_b], strategy='average')
        >>> logits = ens(torch.randn(4, 16, 256))
        >>> assert logits.shape == (4, 1)
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        strategy: str = "weighted",
        threshold: float = 0.5,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.strategy = strategy
        self.threshold = threshold
        for model in self.models:
            model.eval()
        self.register_buffer("weights", _normalise_weights(weights, len(models)))
        if strategy == "stacking":
            self.meta_learner = _build_meta_learner(len(models))

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Compute ensemble prediction.

        Args:
            eeg_input: EEG batch, shape (batch, n_channels, time_steps).

        Returns:
            Ensemble logits, shape (batch, 1).
        """
        all_probs = self._collect_probs(eeg_input)
        return self._combine(all_probs)

    def predict_proba(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Return ensemble probabilities.

        Args:
            eeg_input: EEG batch, shape (batch, n_channels, time_steps).

        Returns:
            Probabilities in [0, 1], shape (batch,).
        """
        logits = self.forward(eeg_input)
        return torch.sigmoid(logits).squeeze(-1)

    def _collect_probs(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """Run all base models and stack their sigmoid probabilities."""
        all_probs = []
        for model in self.models:
            with torch.no_grad():
                logits = model(eeg_input)
            all_probs.append(torch.sigmoid(logits))
        return torch.stack(all_probs, dim=-1)

    def _combine(self, all_probs: torch.Tensor) -> torch.Tensor:
        """Combine per-model probabilities using the selected strategy."""
        if self.strategy == "average":
            ensemble_prob = all_probs.mean(dim=-1)
        elif self.strategy == "weighted":
            w = self.weights.to(all_probs.device).view(1, 1, -1)
            ensemble_prob = (all_probs * w).sum(dim=-1)
        elif self.strategy == "voting":
            ensemble_prob = (all_probs > self.threshold).float().mean(dim=-1)
        elif self.strategy == "max":
            ensemble_prob = all_probs.max(dim=-1).values
        elif self.strategy == "stacking":
            stacked = all_probs.squeeze(1)
            return self.meta_learner(stacked)
        else:
            raise ValueError(f"Unknown ensemble strategy: '{self.strategy}'")
        ensemble_prob = torch.clamp(ensemble_prob, 1e-7, 1.0 - 1e-7)
        return torch.log(ensemble_prob / (1.0 - ensemble_prob))


def load_ensemble_from_checkpoints(
    model_builders: List[callable],
    checkpoint_paths: List[str],
    device: torch.device,
    strategy: str = "weighted",
    weights: Optional[List[float]] = None,
) -> EnsemblePredictor:
    """
    Load checkpoint files and build an EnsemblePredictor.

    Args:
        model_builders: List of callables that return an nn.Module each.
        checkpoint_paths: Corresponding list of .pt checkpoint file paths.
        device: Device to map checkpoint tensors to.
        strategy: Ensemble combination strategy.
        weights: Optional per-model weights.

    Returns:
        EnsemblePredictor with all checkpoints loaded.

    Raises:
        FileNotFoundError: If any checkpoint path does not exist.
    """
    models = []
    for builder, ckpt_path in zip(model_builders, checkpoint_paths):
        model = builder()
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)
        models.append(model)
    logger.info("Loaded %d model checkpoints for ensemble.", len(models))
    return EnsemblePredictor(models, weights=weights, strategy=strategy)


def _normalise_weights(weights: Optional[List[float]], n_models: int) -> torch.Tensor:
    """Return a normalised weight tensor (uniform if weights is None)."""
    if weights is None:
        return torch.ones(n_models) / n_models
    w = np.array(weights, dtype=np.float32)
    return torch.tensor(w / w.sum(), dtype=torch.float32)


def _build_meta_learner(n_models: int) -> nn.Sequential:
    """Build a small FC meta-learner for stacking ensemble."""
    return nn.Sequential(
        nn.Linear(n_models, 16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, 1),
    )


# ---------------------------------------------------------------------------
# CLI: discover checkpoints → per-model inference → ensemble evaluation
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble over all improved-LSTM checkpoints.")
    parser.add_argument("--data_path", required=True,
                        help="Tensor splits directory (train/val/test/{data,labels}.pt).")
    parser.add_argument(
        "--ckpt_dir",
        nargs="+",
        default=[
            "src/models/lstm_benchmark_models/checkpoints",
            "src/models/improved_lstm_models/checkpoints",
        ],
        help="One or more directories holding unified-schema .pt checkpoints. "
             "Defaults cover both benchmark (m1..m7) and improved LSTM.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="src/models/improved_lstm_models/checkpoints/ensemble",
                        help="Where to write ensemble_meta.pt and ensemble_metrics.json.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point — ensemble every .pt checkpoint in ``--ckpt_dir``."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    ckpt_dirs = [Path(d) for d in args.ckpt_dir]
    ckpts: List[Path] = []
    for d in ckpt_dirs:
        ckpts.extend(_discover_checkpoints(d))
    if not ckpts:
        logger.error("No unified-schema .pt checkpoints found in %s", ckpt_dirs)
        return
    logger.info("Found %d checkpoints: %s", len(ckpts), [p.name for p in ckpts])

    models, weights, thresholds = _load_members(ckpts, device)
    if not models:
        logger.error("Could not rebuild any member model; aborting.")
        return

    val_dl, test_dl = _build_dataloaders(Path(args.data_path), args.batch_size)
    val_true = _collect_labels(val_dl)
    test_true = _collect_labels(test_dl)

    val_probs_per_model = [_predict(m, val_dl, device) for m in models]
    test_probs_per_model = [_predict(m, test_dl, device) for m in models]

    results: Dict[str, Dict[str, float]] = {}
    for strategy in ("mean", "weighted"):
        w = np.asarray(weights) if strategy == "weighted" else None
        val_ens = _combine(val_probs_per_model, w)
        test_ens = _combine(test_probs_per_model, w)
        best_t = _tune_threshold(val_true, val_ens)
        results[strategy] = _metrics(test_true, test_ens, best_t)
        logger.info("[%s] threshold=%.2f  F1=%.4f  AUROC=%.4f  sens=%.4f  spec=%.4f",
                    strategy, best_t, results[strategy]["f1"], results[strategy]["auroc"],
                    results[strategy]["sens"], results[strategy]["spec"])
        results[strategy]["threshold"] = best_t

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "member_paths": [str(p) for p in ckpts],
            "member_weights": [float(w) for w in weights],
            "member_thresholds": [float(t) for t in thresholds],
            "results": results,
        },
        out_dir / "ensemble_meta.pt",
    )
    with open(out_dir / "ensemble_metrics.json", "w") as fh:
        json.dump({"members": [p.name for p in ckpts], "results": results}, fh, indent=2)

    # Dump combined soft probabilities for the meta-ensemble. We write the
    # ``weighted`` strategy's probs by default (higher val F1 in every run so
    # far) plus the corresponding labels; the meta-ensemble treats these as
    # the LSTM family's contribution.
    weighted_w = np.asarray(weights)
    val_final = _combine(val_probs_per_model, weighted_w)
    test_final = _combine(test_probs_per_model, weighted_w)
    np.save(out_dir / "val_probs.npy", val_final)
    np.save(out_dir / "val_labels.npy", val_true)
    np.save(out_dir / "test_probs.npy", test_final)
    np.save(out_dir / "test_labels.npy", test_true)
    logger.info("Wrote ensemble meta -> %s (+val/test probs.npy)", out_dir)


def _discover_checkpoints(ckpt_dir: Path) -> List[Path]:
    """Return sorted list of unified-schema .pt files in ``ckpt_dir``.

    Skips anything inside an ``ensemble/`` subdirectory so re-running the
    ensemble doesn't pick up its own meta file as a member.
    """
    if not ckpt_dir.exists():
        return []
    return sorted(
        p for p in ckpt_dir.glob("*.pt")
        if p.is_file() and p.parent.name != "ensemble"
    )


def _load_members(
    ckpts: List[Path], device: torch.device,
) -> Tuple[List[nn.Module], List[float], List[float]]:
    """Rebuild each checkpoint's model; return (models, weights, thresholds)."""
    from src.models.utils.checkpoint import load_checkpoint  # local to avoid cycles
    models, weights, thresholds = [], [], []
    for path in ckpts:
        try:
            model, payload = load_checkpoint(path, map_location=device, build_model=True)
        except Exception as exc:
            logger.warning("Skip %s: %s", path.name, exc)
            continue
        if model is None:
            logger.warning("Skip %s: auto-rebuild failed (missing model_class/config).", path.name)
            continue
        model = model.to(device).eval()
        val_f1 = float(payload.get("val_metrics", {}).get("f1", 0.0))
        threshold = float(payload.get("optimal_threshold", 0.5))
        models.append(model)
        weights.append(max(val_f1, 1e-3))  # weight proportional to val F1
        thresholds.append(threshold)
        logger.info("Loaded %s (val_f1=%.4f, threshold=%.3f)", path.name, val_f1, threshold)
    if weights:
        total = sum(weights)
        weights = [w / total for w in weights]
    return models, weights, thresholds


def _build_dataloaders(data_path: Path, batch_size: int):
    """Load val and test tensors; return (val_loader, test_loader)."""
    from torch.utils.data import DataLoader, TensorDataset
    pin = torch.cuda.is_available()

    def _split(name: str) -> TensorDataset:
        data = torch.load(data_path / name / "data.pt", weights_only=True).float()
        labels = torch.load(data_path / name / "labels.pt", weights_only=True).long().squeeze()
        return TensorDataset(data, labels)

    return (
        DataLoader(_split("val"), batch_size=batch_size, shuffle=False, pin_memory=pin),
        DataLoader(_split("test"), batch_size=batch_size, shuffle=False, pin_memory=pin),
    )


def _predict(model: nn.Module, loader, device: torch.device) -> np.ndarray:
    """Run model on loader; return 1-D array of positive-class probabilities.

    Dispatches between 3-D and 4-D input contracts automatically: models with
    an ``n_windows`` attribute (e.g. HierarchicalLSTM) receive 4-D tensors,
    everything else (m1..m7 benchmarks) receives 3-D as-is.
    """
    needs_4d = hasattr(model, "n_windows")
    probs = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if needs_4d and x.ndim == 3:
                x = x.unsqueeze(1)  # (B, 1, C, T)
            logits = model(x).squeeze(-1).squeeze()
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(np.atleast_1d(p))
    return np.concatenate(probs)


def _collect_labels(loader) -> np.ndarray:
    """Concatenate labels across a loader."""
    out = []
    for _, y in loader:
        out.append(y.numpy())
    return np.concatenate(out)


def _combine(probs_list: List[np.ndarray], weights: Optional[np.ndarray]) -> np.ndarray:
    """Combine per-model probability arrays via mean or weighted-average."""
    stack = np.stack(probs_list)  # (n_models, n_samples)
    if weights is None:
        return stack.mean(axis=0)
    w = weights / weights.sum()
    return np.average(stack, axis=0, weights=w)


def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return threshold in [0.05, 0.95] that maximises F1 on given data."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.05):
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """Return F1/AUROC/sens/spec at the given threshold."""
    from sklearn.metrics import f1_score, roc_auc_score, recall_score
    y_pred = (y_prob >= threshold).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = float("nan")
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": auroc,
        "sens": float(recall_score(y_true, y_pred, zero_division=0)),
        "spec": float(tn / max(tn + fp, 1)),
    }


if __name__ == "__main__":
    main()
