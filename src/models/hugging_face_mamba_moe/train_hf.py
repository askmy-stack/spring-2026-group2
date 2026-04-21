"""
Hugging Face EEG Model Training — Directory 4 of 4.

Trains any model in HF_MODEL_REGISTRY using the existing cached data loader.
Supports cross-entropy and focal loss, early stopping, gradient clipping,
threshold tuning, and temporal smoothing metrics.

Usage:
    python -m src.models.hugging_face_mamba_moe.train_hf \
        --model baseline_cnn_1d \
        --config_path src/data_loader/config.yaml

All available models:
    baseline_cnn_1d, enhanced_cnn_1d, eegnet_local, eegnet, deepconvnet,
    multiscale_cnn, multiscale_attention_cnn, st_eegformer,
    bendr_pretrained, biot_pretrained, eegpt_pretrained, hf_st_eegformer
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from src.models.utils.checkpoint import save_checkpoint
from .architectures.hf_factory import create_hf_model, list_hf_models

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Build and return CLI argument parser."""
    parser = argparse.ArgumentParser(description="Train HF EEG models via cached dataloader.")
    parser.add_argument("--model", default="baseline_cnn_1d", choices=["all"] + list_hf_models())
    parser.add_argument("--run_name", default="", help="Optional suffix for results directory.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--train_augment", action="store_true")
    parser.add_argument("--loss", default="focal", choices=("ce", "focal"))
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--threshold_mode", default="tune", choices=("fixed", "tune"))
    parser.add_argument("--decision_threshold", type=float, default=0.35)
    parser.add_argument("--smoothing_mode", default="none", choices=("none", "moving_average", "consecutive"))
    parser.add_argument("--smoothing_window", type=int, default=5)
    parser.add_argument("--min_positive_run", type=int, default=2)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--config_path", default="src/data_loader/config.yaml")
    parser.add_argument("--data_path", default="", help="Path to pre-processed tensors (train/val/test with data.pt & labels.pt). If set, bypasses cache loader.")
    return parser.parse_args()


def _train_single_model(args: argparse.Namespace) -> None:
    """Train a single HF model end-to-end."""
    _set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config_path = Path(args.config_path).resolve()
    results_root, ckpt_dir, logs_dir = _setup_dirs(args, config_path)
    train_dl, val_dl, test_dl, channels, samples, sfreq = _load_data(args, config_path)
    model = _build_model(args, channels, samples, sfreq, device)
    criterion = _build_criterion(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    _save_config(logs_dir, args, channels, samples, sfreq, str(device), config_path)
    _training_loop(model, train_dl, val_dl, test_dl, criterion, optimizer,
                   device, args, ckpt_dir, logs_dir, results_root)


def main() -> None:
    """CLI entry point: set up, train, evaluate on test set."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.model == "all":
        incompatible_models = {
            "bendr_pretrained": "requires 20 channels (CHB-MIT has 16)",
            "biot_pretrained": "requires 200 Hz sampling (CHB-MIT is 256 Hz)",
            "eegpt_pretrained": "requires 62 channels (CHB-MIT has 19)",
            "hf_st_eegformer": "requires 128 Hz sampling (CHB-MIT is 256 Hz)",
            "st_eegformer": "requires 128 Hz sampling (CHB-MIT is 256 Hz)"
        }
        for model_name in list_hf_models():
            if model_name in incompatible_models:
                logger.warning("Skipping %s: %s", model_name, incompatible_models[model_name])
                continue
            logger.info("=" * 60)
            logger.info("Training HF model: %s", model_name)
            logger.info("=" * 60)
            args.model = model_name
            try:
                _train_single_model(args)
            except Exception as e:
                logger.error("Model %s FAILED: %s", model_name, e)
    else:
        _train_single_model(args)


def _training_loop(
    model: nn.Module, train_dl, val_dl, test_dl,
    criterion: nn.Module, optimizer: torch.optim.Optimizer,
    device: torch.device, args: argparse.Namespace,
    ckpt_dir: Path, logs_dir: Path, results_root: Path,
) -> None:
    """Full training + validation loop with early stopping."""
    best_val_f1, best_epoch, no_improve = -1.0, 0, 0
    history: list[dict[str, Any]] = []
    threshold = args.decision_threshold
    for epoch in range(1, args.epochs + 1):
        train_m = _run_epoch(model, train_dl, criterion, device, optimizer,
                             threshold=threshold, log_interval=500, split_name="train",
                             grad_clip_norm=args.grad_clip_norm)
        val_m = _run_epoch(model, val_dl, criterion, device, optimizer=None,
                           threshold=threshold, tune_threshold=args.threshold_mode == "tune",
                           smoothing_mode=args.smoothing_mode, smoothing_window=args.smoothing_window,
                           min_positive_run=args.min_positive_run, split_name="val")
        threshold = float(val_m.get("threshold", threshold))
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
               **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        logger.info("Epoch %d  train_loss=%.4f val_f1=%.4f val_auroc=%.4f",
                    epoch, train_m["loss"], val_m.get("f1", 0), val_m.get("auroc", float("nan")))
        train_loss_val = train_m.get("loss", float("nan"))
        if not np.isfinite(train_loss_val):
            logger.error("Non-finite train loss at epoch %d — aborting this model run.", epoch)
            break
        cur_f1 = val_m.get("f1", 0.0)
        if cur_f1 > best_val_f1:
            best_val_f1, best_epoch, no_improve = cur_f1, epoch, 0
            model_config = {
                "name": args.model,
                "in_channels": int(getattr(args, "channels", 16)),
                "num_classes": int(args.num_classes),
                "n_times": int(getattr(args, "samples", 256)),
                "sfreq": int(getattr(args, "sfreq", 256)),
                "dropout": float(args.dropout),
                "freeze_backbone": bool(args.freeze_backbone),
            }
            save_checkpoint(
                ckpt_dir / "best_model.pt", model,
                model_config=model_config,
                model_builder="src.models.hugging_face_mamba_moe.architectures.hf_factory.create_hf_model",
                epoch=epoch,
                val_metrics=val_m,
                optimal_threshold=float(threshold),
            )
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stopping at epoch %d (patience=%d).", epoch, args.patience)
                break
    _save_history(logs_dir / "history.csv", history)
    _run_test(model, test_dl, criterion, device, threshold, args, ckpt_dir, logs_dir)


def _run_test(
    model: nn.Module, test_dl, criterion: nn.Module, device: torch.device,
    threshold: float, args: argparse.Namespace, ckpt_dir: Path, logs_dir: Path,
) -> None:
    """Load best checkpoint, evaluate, and rewrite it in the unified schema."""
    best_ckpt = ckpt_dir / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    test_m = _run_epoch(model, test_dl, criterion, device, optimizer=None,
                        threshold=threshold, smoothing_mode=args.smoothing_mode,
                        smoothing_window=args.smoothing_window, min_positive_run=args.min_positive_run,
                        split_name="test")
    logger.info("Test results: %s", test_m)
    _save_json(logs_dir / "test_metrics.json", test_m)
    # Rewrite best_model.pt with the unified checkpoint schema so downstream
    # code (inference, ensembles, apps) can load it via utils.checkpoint.load_checkpoint.
    # We record a ``model_builder`` pointing to create_hf_model so load_checkpoint
    # rebuilds the HF model via the factory (class-based rebuild wouldn't work
    # because most HF wrappers don't accept the factory's kwargs directly).
    model_config = {
        "name": args.model,
        "in_channels": int(getattr(args, "channels", 16)),
        "num_classes": int(args.num_classes),
        "n_times": int(getattr(args, "samples", 256)),
        "sfreq": int(getattr(args, "sfreq", 256)),
        "dropout": float(args.dropout),
        "freeze_backbone": bool(args.freeze_backbone),
    }
    val_metrics = {
        "f1": float(test_m.get("f1", 0.0)),
        "auroc": float(test_m.get("auroc", float("nan"))),
        "sens": float(test_m.get("recall", 0.0)),
        "spec": float(test_m.get("tn", 0) / max(test_m.get("tn", 0) + test_m.get("fp", 0), 1)),
    }
    save_checkpoint(
        best_ckpt, model,
        model_config=model_config,
        model_builder="src.models.hugging_face_mamba_moe.architectures.hf_factory.create_hf_model",
        optimizer=None,
        epoch=int(args.epochs),
        val_metrics=val_metrics,
        optimal_threshold=float(threshold),
    )


def _run_epoch(
    model: nn.Module, dataloader, criterion: nn.Module, device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float = 0.5, tune_threshold: bool = False,
    smoothing_mode: str = "none", smoothing_window: int = 5,
    min_positive_run: int = 2, log_interval: int = 0,
    split_name: str = "train", grad_clip_norm: float = 0.0,
) -> dict[str, Any]:
    """Run one epoch of training or evaluation; return metric dict."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_items = 0.0, 0
    y_true_all, y_prob_all = [], []
    for step, (x, y) in enumerate(dataloader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        probs = torch.softmax(logits, dim=1)[:, 1]
        total_loss += float(loss.item()) * x.size(0)
        total_items += x.size(0)
        y_true_all.append(y.cpu().numpy())
        y_prob_all.append(probs.detach().cpu().numpy())
        if log_interval > 0 and step % log_interval == 0:
            logger.info("[%s] step=%d avg_loss=%.4f", split_name, step, total_loss / total_items)
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([], dtype=float)
    metrics = _compute_metrics(y_true, y_prob, threshold, tune_threshold,
                               smoothing_mode, smoothing_window, min_positive_run)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


def _compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray,
    threshold: float, tune: bool,
    smoothing_mode: str, smoothing_window: int, min_positive_run: int,
) -> dict[str, Any]:
    """Compute F1/AUROC/precision/recall with optional threshold tuning."""
    if y_true.size == 0:
        return {}
    if tune:
        # Start from the fixed-threshold metrics so we always have a populated dict,
        # even when all tuned thresholds produce zero F1 (e.g., NaN logits).
        best_m = _metrics_at_threshold(y_true, y_prob, threshold, smoothing_mode, smoothing_window, min_positive_run)
        for t in np.round(np.arange(0.05, 0.951, 0.05), 2):
            m = _metrics_at_threshold(y_true, y_prob, float(t), smoothing_mode, smoothing_window, min_positive_run)
            if m.get("f1", 0) > best_m.get("f1", 0):
                best_m = m
        return best_m
    return _metrics_at_threshold(y_true, y_prob, threshold, smoothing_mode, smoothing_window, min_positive_run)


def _metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
    smoothing_mode: str, smoothing_window: int, min_positive_run: int,
) -> dict[str, Any]:
    """Compute binary classification metrics for a fixed threshold."""
    if smoothing_mode == "moving_average" and smoothing_window > 1:
        y_prob = np.convolve(y_prob, np.ones(smoothing_window) / smoothing_window, mode="same")
    y_pred = (y_prob >= threshold).astype(int)
    if smoothing_mode == "consecutive" and min_positive_run > 1:
        y_pred = _consecutive_smooth(y_pred, min_positive_run)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        auroc = float("nan")
    return {"f1": f1, "precision": precision, "recall": recall, "auroc": auroc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": threshold}


def _consecutive_smooth(y_pred: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only runs of ≥ min_run consecutive positives."""
    result = np.zeros_like(y_pred)
    start = None
    for i, v in enumerate(y_pred):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            if i - start >= min_run:
                result[start:i] = 1
            start = None
    if start is not None and len(y_pred) - start >= min_run:
        result[start:] = 1
    return result


def _load_data(args: argparse.Namespace, config_path: Path):
    """Load dataloaders; return (train, val, test, channels, samples, sfreq)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    channels = int(cfg.get("channels", {}).get("target_count", 16))
    sfreq = int(cfg.get("signal", {}).get("target_sfreq", 256))
    samples = int(cfg.get("windowing", {}).get("window_sec", 1.0) * sfreq)

    if args.data_path:
        train_dl, val_dl, test_dl = _load_from_tensors(
            Path(args.data_path), args.batch_size,
        )
    else:
        from src.data_loader.load_cache import get_dataloaders
        train_dl, val_dl, test_dl = get_dataloaders(
            batch_size=args.batch_size, num_workers=args.num_workers,
            augment_train=args.train_augment,
        )
    return train_dl, val_dl, test_dl, channels, samples, sfreq


def _load_from_tensors(data_path: Path, batch_size: int):
    """Load pre-processed .pt tensor splits — same format as benchmark models."""
    from torch.utils.data import DataLoader, TensorDataset
    pin = torch.cuda.is_available()

    def _split(name: str):
        d = torch.load(data_path / name / "data.pt", weights_only=True).float()
        l = torch.load(data_path / name / "labels.pt", weights_only=True).long().squeeze()
        return TensorDataset(d, l)

    return (
        DataLoader(_split("train"), batch_size=batch_size, shuffle=True, pin_memory=pin),
        DataLoader(_split("val"), batch_size=batch_size, shuffle=False, pin_memory=pin),
        DataLoader(_split("test"), batch_size=batch_size, shuffle=False, pin_memory=pin),
    )


def _build_model(args: argparse.Namespace, channels: int, samples: int, sfreq: int,
                 device: torch.device) -> nn.Module:
    """Instantiate model from factory and move to device."""
    return create_hf_model(
        args.model, in_channels=channels, num_classes=args.num_classes,
        n_times=samples, sfreq=sfreq, dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)


def _build_criterion(args: argparse.Namespace) -> nn.Module:
    """Build cross-entropy or focal loss from CLI args."""
    if args.loss == "focal":
        return _BinaryFocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    return nn.CrossEntropyLoss()


def _setup_dirs(args: argparse.Namespace, config_path: Path):
    """Create results / checkpoint / log directories."""
    src_dir = Path(__file__).parent.parent.parent
    run_name = args.model if not args.run_name else f"{args.model}__{args.run_name}"
    results_root = src_dir.parent / "results" / "model" / run_name
    ckpt_dir = results_root / "checkpoints"
    logs_dir = results_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return results_root, ckpt_dir, logs_dir


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_config(logs_dir: Path, args: argparse.Namespace, channels: int,
                 samples: int, sfreq: int, device: str, config_path: Path) -> None:
    _save_json(logs_dir / "config.json", vars(args) | {"channels": channels, "samples": samples,
                                                        "sfreq": sfreq, "device": device,
                                                        "config_path": str(config_path)})


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _save_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


class _BinaryFocalLoss(nn.Module):
    """Focal loss for 2-class cross-entropy logits."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args: logits (batch, 2), targets (batch,) long."""
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(targets == 1, torch.full_like(ce, self.alpha),
                              torch.full_like(ce, 1.0 - self.alpha))
        return (alpha_t * (1.0 - pt) ** self.gamma * ce).mean()


if __name__ == "__main__":
    main()
