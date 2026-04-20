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

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parents[2]
sys.path.insert(0, str(SRC_DIR))

from data_loader.load_cache import get_dataloaders
from models.utils.artifacts import ensure_artifact_dirs
from models.utils.losses import BinaryFocalLoss
from models.utils.metrics import compute_metrics_with_smoothing, find_best_threshold, safe_div
from models.utils.registry import create_model, list_models


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device):
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def save_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def save_history_csv(path: Path, history: list[dict[str, Any]]):
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def build_train_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=list_models(), required=True)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sfreq", type=int, default=128)
    parser.add_argument("--config-path", default=str(SRC_DIR / "data_loader" / "config.yaml"))
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--train-augment", action="store_true")
    parser.add_argument("--loss", default="ce", choices=("ce", "focal"))
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--threshold-mode", default="fixed", choices=("fixed", "tune"))
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--smoothing-mode", default="none", choices=("none", "moving_average", "consecutive"))
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--min-positive-run", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--eval-log-interval", type=int, default=200)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    return parser


def build_eval_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=list_models(), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=("cpu", "cuda"))
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--sfreq", type=int, default=128)
    parser.add_argument("--config-path", default=str(SRC_DIR / "data_loader" / "config.yaml"))
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser


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
            print(f"  [{split_name}] step {step:>6}/{effective_total:<6} avg_loss={safe_div(total_loss, total_items):.4f}", flush=True)
    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([], dtype=float)
    if total_items:
        if tune_threshold:
            best_threshold, metrics = find_best_threshold(y_true, y_prob, smoothing_mode, smoothing_window, min_positive_run)
        else:
            best_threshold = threshold
            metrics = compute_metrics_with_smoothing(y_true, y_prob, threshold, smoothing_mode, smoothing_window, min_positive_run)
    else:
        best_threshold = threshold
        metrics = {}
    metrics["loss"] = safe_div(total_loss, total_items)
    metrics["threshold"] = float(best_threshold)
    return metrics


def train_model(model_name: str, args: argparse.Namespace) -> int:
    set_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config_path = Path(args.config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    channels = int(cfg.get("channels", {}).get("target_count", args.channels))
    sfreq = int(cfg.get("signal", {}).get("target_sfreq", args.sfreq))
    window_sec = float(cfg.get("windowing", {}).get("window_sec", 1.0))
    samples = int(window_sec * sfreq)
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, augment_train=args.train_augment, config_path=config_path)
    model_kwargs = {"in_channels": channels, "num_classes": args.num_classes, "n_times": samples, "sfreq": sfreq, "freeze_backbone": args.freeze_backbone}
    if model_name in {"baseline_cnn_1d", "multiscale_attention_cnn"}:
        model_kwargs["dropout"] = args.dropout
    model = create_model(model_name, **model_kwargs).to(device)
    criterion = BinaryFocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha) if args.loss == "focal" else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    artifact_model_name = model_name if not args.run_name else f"{model_name}__{args.run_name}"
    artifact_paths = ensure_artifact_dirs(model_name=artifact_model_name)
    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_threshold = args.decision_threshold
    epochs_without_improvement = 0
    config_payload = {
        "model": model_name,
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
    save_json(artifact_paths.config_path, config_payload)
    print("=" * 88)
    print(f"Training model: {model_name}")
    print(f"Checkpoint    : {artifact_paths.checkpoint_path}")
    print(f"Results dir   : {artifact_paths.results_dir}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"LR            : {args.lr}")
    print("=" * 88)
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch:03d}/{args.epochs:03d}] Training ...", flush=True)
        train_metrics = run_epoch(model, train_dl, criterion, device, optimizer=optimizer, threshold=best_threshold, log_interval=args.log_interval, split_name="train", max_batches=args.max_train_batches, grad_clip_norm=args.grad_clip_norm)
        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] Validating ...", flush=True)
        val_metrics = run_epoch(model, val_dl, criterion, device, optimizer=None, threshold=best_threshold, tune_threshold=args.threshold_mode == "tune", smoothing_mode=args.smoothing_mode, smoothing_window=args.smoothing_window, min_positive_run=args.min_positive_run, log_interval=args.eval_log_interval, split_name="val", max_batches=args.max_val_batches)
        current_threshold = float(val_metrics["threshold"])
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"], "train_precision": train_metrics["precision"], "train_recall": train_metrics["recall"], "train_f1": train_metrics["f1"], "train_auroc": train_metrics["auroc"], "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"], "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"], "val_f1": val_metrics["f1"], "val_auroc": val_metrics["auroc"], "val_threshold": current_threshold}
        history.append(row)
        print(f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} train_f1={row['train_f1']:.4f} | val_loss={row['val_loss']:.4f} val_f1={row['val_f1']:.4f} val_recall={row['val_recall']:.4f} val_threshold={current_threshold:.2f}")
        if row["val_f1"] > best_val_f1:
            best_val_f1 = row["val_f1"]
            best_epoch = epoch
            best_threshold = current_threshold
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "config": config_payload,
                    "best_threshold": best_threshold,
                    "artifact_model_name": artifact_model_name,
                },
                artifact_paths.checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        save_history_csv(artifact_paths.history_path, history)
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} (no validation F1 improvement for {args.patience} epochs).")
            break
    best_ckpt = torch.load(artifact_paths.checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    best_threshold = float(best_ckpt.get("best_threshold", best_threshold))
    print("\n[Final] Testing best checkpoint ...", flush=True)
    test_metrics = run_epoch(model, test_dl, criterion, device, optimizer=None, threshold=best_threshold, smoothing_mode=args.smoothing_mode, smoothing_window=args.smoothing_window, min_positive_run=args.min_positive_run, log_interval=args.eval_log_interval, split_name="test", max_batches=args.max_test_batches)
    save_json(artifact_paths.test_metrics_path, test_metrics)
    save_json(
        artifact_paths.summary_path,
        {
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "best_threshold": best_threshold,
            "test_metrics": test_metrics,
            "config": config_payload,
        },
    )
    print("=" * 88)
    print(f"Best epoch     : {best_epoch}")
    print(f"Best val F1    : {best_val_f1:.4f}")
    print(f"Best threshold : {best_threshold:.2f}")
    print(f"Test metrics   : loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f} precision={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} f1={test_metrics['f1']:.4f} auroc={test_metrics['auroc']:.4f} threshold={test_metrics['threshold']:.2f}")
    print(f"Artifacts saved: {artifact_paths.results_dir}")
    print("=" * 88)
    return 0


def eval_model(model_name: str, args: argparse.Namespace) -> int:
    from torch.utils.data import DataLoader
    from data_loader.dataset.factory import create_loader
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config_path = Path(args.config_path).resolve()
    datasets = {
        "train": create_loader("cached", config_path=str(config_path), mode="train", cache_memory_mb=2048, augment_data=False),
        "val": create_loader("cached", config_path=str(config_path), mode="val", cache_memory_mb=512, augment_data=False),
        "test": create_loader("cached", config_path=str(config_path), mode="test", cache_memory_mb=512, augment_data=False),
    }
    batch_x, batch_y = next(iter(DataLoader(datasets[args.split], batch_size=args.batch_size, shuffle=(args.split == "train"), num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False, persistent_workers=args.num_workers > 0, prefetch_factor=2 if args.num_workers > 0 else None)))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    channels = int(cfg.get("channels", {}).get("target_count", args.channels))
    sfreq = int(cfg.get("signal", {}).get("target_sfreq", args.sfreq))
    window_sec = float(cfg.get("windowing", {}).get("window_sec", 1.0))
    samples = int(window_sec * sfreq)
    model = create_model(model_name, in_channels=channels, num_classes=args.num_classes, n_times=samples, sfreq=sfreq, freeze_backbone=args.freeze_backbone).to(device)
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch_x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
    print("=" * 72)
    print(f"Model            : {model_name}")
    print(f"Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Input batch      : {tuple(batch_x.shape)}")
    print(f"Labels           : {tuple(batch_y.shape)}")
    print(f"Logits           : {tuple(logits.shape)}")
    print(f"Predictions      : {tuple(preds.shape)}")
    print(f"Pred class count : {(preds == 1).sum().item()} seizure / {(preds == 0).sum().item()} background")
    print(f"Prob range       : min={probs.min().item():.4f} max={probs.max().item():.4f}")
    print("=" * 72)
    return 0
