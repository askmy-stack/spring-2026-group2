from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

THIS_DIR = Path(__file__).parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from data_loader.load_cache import get_dataloaders
from model.factory import create_model, list_models

try:
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG CNN models on cached dataloader tensors.")
    parser.add_argument("--model", default="baseline_cnn_1d", choices=list_models(), help="Model to train.")
    parser.add_argument("--run-name", default="", help="Optional suffix for the results directory, e.g. focal or thresh.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Model dropout override where supported.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on validation F1.")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--channels", type=int, default=16, help="Input EEG channels.")
    parser.add_argument("--samples", type=int, default=128, help="Input samples per EEG window.")
    parser.add_argument("--sfreq", type=int, default=128, help="Sampling frequency of EEG windows.")
    parser.add_argument("--config-path", default=str(SRC_DIR / "data_loader" / "config.yaml"), help="Path to dataloader config.")
    parser.add_argument("--num-classes", type=int, default=2, help="Output classes.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze the pretrained backbone when supported.")
    parser.add_argument("--train-augment", action="store_true", help="Enable train-time augmentation in cached loader.")
    parser.add_argument("--loss", default="ce", choices=("ce", "focal"), help="Training loss.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma.")
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.25,
        help="Positive-class alpha for focal loss. Ignored for cross-entropy.",
    )
    parser.add_argument(
        "--threshold-mode",
        default="fixed",
        choices=("fixed", "tune"),
        help="Use a fixed decision threshold or tune it on validation each epoch.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Fixed positive-class threshold when threshold-mode=fixed.",
    )
    parser.add_argument(
        "--smoothing-mode",
        default="none",
        choices=("none", "moving_average", "consecutive"),
        help="Optional temporal smoothing applied during validation/test metric computation.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Window size for moving-average temporal smoothing.",
    )
    parser.add_argument(
        "--min-positive-run",
        type=int,
        default=2,
        help="Minimum consecutive positive windows required when smoothing-mode=consecutive.",
    )
    parser.add_argument("--log-interval", type=int, default=500, help="Print batch progress every N steps during training.")
    parser.add_argument("--eval-log-interval", type=int, default=200, help="Print batch progress every N steps during validation/test.")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Optional limit on train batches per epoch. 0 means full epoch.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Optional limit on validation batches per epoch. 0 means full validation.")
    parser.add_argument("--max-test-batches", type=int, default=0, help="Optional limit on test batches. 0 means full test set.")
    parser.add_argument("--grad-clip-norm", type=float, default=0.0, help="Gradient clipping max norm. 0 disables clipping.")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0 or y_prob.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "auroc": float("nan"),
            "tp": 0.0,
            "tn": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    finite_mask = np.isfinite(y_prob)
    if not finite_mask.all():
        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]
        y_prob = y_prob[finite_mask]
        if y_true.size == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "specificity": 0.0,
                "f1": 0.0,
                "auroc": float("nan"),
                "tp": 0.0,
                "tn": 0.0,
                "fp": 0.0,
                "fn": 0.0,
            }

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = safe_div(tp + tn, len(y_true))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)

    if SKLEARN_AVAILABLE and len(np.unique(y_true)) > 1:
        auroc = float(roc_auc_score(y_true, y_prob))
    else:
        auroc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auroc": auroc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    metrics["threshold"] = float(threshold)
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
    best_metrics = compute_metrics_with_smoothing(
        y_true,
        y_prob,
        best_threshold,
        smoothing_mode=smoothing_mode,
        smoothing_window=smoothing_window,
        min_positive_run=min_positive_run,
    )

    for threshold in thresholds:
        metrics = compute_metrics_with_smoothing(
            y_true,
            y_prob,
            float(threshold),
            smoothing_mode=smoothing_mode,
            smoothing_window=smoothing_window,
            min_positive_run=min_positive_run,
        )
        if metrics["f1"] > best_metrics["f1"] or (
            metrics["f1"] == best_metrics["f1"] and metrics["precision"] > best_metrics["precision"]
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(ce, self.alpha),
            torch.full_like(ce, 1.0 - self.alpha),
        )
        loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
    tune_threshold: bool = False,
    smoothing_mode: str = "none",
    smoothing_window: int = 5,
    min_positive_run: int = 2,
    log_interval: int = 0,
    split_name: str = "train",
    max_batches: int = 0,
    grad_clip_norm: float = 0.0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_items = 0
    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []

    total_batches = len(dataloader)
    effective_total = min(total_batches, max_batches) if max_batches and max_batches > 0 else total_batches

    if log_interval > 0:
        print(f"  [{split_name}] running {effective_total} batch(es)", flush=True)

    for step, batch in enumerate(dataloader, start=1):
        if max_batches and step > max_batches:
            break
        x, y = to_device(batch, device)

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1]
        batch_size = x.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

        y_true_all.append(y.detach().cpu().numpy())
        y_prob_all.append(probs.detach().cpu().numpy())

        if log_interval > 0 and (step % log_interval == 0 or step == effective_total):
            avg_loss = safe_div(total_loss, total_items)
            print(
                f"  [{split_name}] step {step:>6}/{effective_total:<6} "
                f"avg_loss={avg_loss:.4f}",
                flush=True,
            )

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([], dtype=float)
    if total_items:
        if tune_threshold:
            best_threshold, metrics = find_best_threshold(
                y_true,
                y_prob,
                smoothing_mode=smoothing_mode,
                smoothing_window=smoothing_window,
                min_positive_run=min_positive_run,
            )
        else:
            best_threshold = threshold
            metrics = compute_metrics_with_smoothing(
                y_true,
                y_prob,
                threshold,
                smoothing_mode=smoothing_mode,
                smoothing_window=smoothing_window,
                min_positive_run=min_positive_run,
            )
    else:
        best_threshold = threshold
        metrics = {}
    metrics["loss"] = safe_div(total_loss, total_items)
    metrics["threshold"] = float(best_threshold)
    return metrics


def save_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_history_csv(path: Path, history: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return history


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config_path = Path(args.config_path).resolve()
    run_dir_name = args.model if not args.run_name else f"{args.model}__{args.run_name}"
    results_root = SRC_DIR.parent / "results" / "model" / run_dir_name
    ckpt_dir = results_root / "checkpoints"
    logs_dir = results_root / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cached dataloaders on {device} ...")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    channels = int(cfg.get("channels", {}).get("target_count", args.channels))
    sfreq = int(cfg.get("signal", {}).get("target_sfreq", args.sfreq))
    window_sec = float(cfg.get("windowing", {}).get("window_sec", 1.0))
    samples = int(window_sec * sfreq)
    train_dl, val_dl, test_dl = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=args.train_augment,
        config_path=config_path,
    )

    model_kwargs = {
        "in_channels": channels,
        "num_classes": args.num_classes,
        "n_times": samples,
        "sfreq": sfreq,
        "freeze_backbone": args.freeze_backbone,
    }
    if args.model in {"baseline_cnn_1d", "multiscale_cnn", "multiscale_attention_cnn"}:
        model_kwargs["dropout"] = args.dropout

    model = create_model(args.model, **model_kwargs).to(device)
    if args.loss == "focal":
        criterion = BinaryFocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    best_threshold = args.decision_threshold

    config_payload = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "device": str(device),
        "seed": args.seed,
        "patience": args.patience,
        "samples": samples,
        "sfreq": sfreq,
        "config_path": str(config_path),
        "freeze_backbone": args.freeze_backbone,
        "train_augment": args.train_augment,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "threshold_mode": args.threshold_mode,
        "decision_threshold": args.decision_threshold,
        "smoothing_mode": args.smoothing_mode,
        "smoothing_window": args.smoothing_window,
        "min_positive_run": args.min_positive_run,
        "run_name": args.run_name,
        "grad_clip_norm": args.grad_clip_norm,
    }
    history_path = logs_dir / "history.csv"
    save_json(logs_dir / "config.json", config_payload)

    print("=" * 88)
    print(f"Training model: {args.model}")
    print(f"Results dir   : {results_root}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"LR            : {args.lr}")
    print("=" * 88)

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch:03d}/{args.epochs:03d}] Training ...", flush=True)
        train_metrics = run_epoch(
            model,
            train_dl,
            criterion,
            device,
            optimizer=optimizer,
            threshold=best_threshold,
            smoothing_mode="none",
            log_interval=args.log_interval,
            split_name="train",
            max_batches=args.max_train_batches,
            grad_clip_norm=args.grad_clip_norm,
        )
        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] Validating ...", flush=True)
        val_metrics = run_epoch(
            model,
            val_dl,
            criterion,
            device,
            optimizer=None,
            threshold=best_threshold,
            tune_threshold=args.threshold_mode == "tune",
            smoothing_mode=args.smoothing_mode,
            smoothing_window=args.smoothing_window,
            min_positive_run=args.min_positive_run,
            log_interval=args.eval_log_interval,
            split_name="val",
            max_batches=args.max_val_batches,
        )
        current_threshold = float(val_metrics["threshold"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_auroc": train_metrics["auroc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
            "val_threshold": current_threshold,
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f} train_f1={row['train_f1']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_f1={row['val_f1']:.4f} "
            f"val_recall={row['val_recall']:.4f} val_threshold={current_threshold:.2f}"
        )

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": config_payload,
                },
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

        if row["val_f1"] > best_val_f1:
            best_val_f1 = row["val_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_threshold = current_threshold
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": config_payload,
                    "best_threshold": best_threshold,
                },
                ckpt_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        save_history_csv(history_path, history)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} (no validation F1 improvement for {args.patience} epochs).")
            break

    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    best_threshold = float(best_ckpt.get("best_threshold", best_threshold))
    print("\n[Final] Testing best checkpoint ...", flush=True)
    test_metrics = run_epoch(
        model,
        test_dl,
        criterion,
        device,
        optimizer=None,
        threshold=best_threshold,
        smoothing_mode=args.smoothing_mode,
        smoothing_window=args.smoothing_window,
        min_positive_run=args.min_positive_run,
        log_interval=args.eval_log_interval,
        split_name="test",
        max_batches=args.max_test_batches,
    )
    save_json(logs_dir / "test_metrics.json", test_metrics)

    summary = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_threshold": best_threshold,
        "test_metrics": test_metrics,
        "config": config_payload,
    }
    save_json(results_root / "summary.json", summary)

    print("=" * 88)
    print(f"Best epoch     : {best_epoch}")
    print(f"Best val F1    : {best_val_f1:.4f}")
    print(f"Best threshold : {best_threshold:.2f}")
    print(
        f"Test metrics   : loss={test_metrics['loss']:.4f} "
        f"acc={test_metrics['accuracy']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"auroc={test_metrics['auroc']:.4f} "
        f"threshold={test_metrics['threshold']:.2f}"
    )
    print(f"Artifacts saved: {results_root}")
    print("=" * 88)


if __name__ == "__main__":
    main()
