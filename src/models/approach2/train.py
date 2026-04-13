#!/usr/bin/env python3
"""
Approach 2: Advanced Training Script
------------------------------------
Train 7 diverse models with optional pre-trained encoders.

Usage:
    python train.py --model m3_criss_cross --epochs 50
    python train.py --model all --epochs 50 --pretrained
    python train.py --model m4_cnn_lstm --pretrained --encoder cbramod
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from architectures import get_model, list_models, MODEL_REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions (from Approach 1)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()


def apply_label_smoothing(y: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
    """Apply label smoothing."""
    if epsilon == 0.0:
        return y
    return y * (1.0 - epsilon) + 0.5 * epsilon


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find optimal threshold using Youden's J statistic."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_j = -1
    best_thresh = 0.5

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(float)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thresh = thresh

    return best_thresh


# ─────────────────────────────────────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool = True,
    label_smoothing: float = 0.05,
    fn_multiplier: float = 2.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).float().unsqueeze(1)

        # Label smoothing
        y_smooth = apply_label_smoothing(y_batch, label_smoothing)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            # Handle models that return multiple outputs (e.g., VQ-Transformer)
            output = model(x_batch)
            if isinstance(output, tuple):
                logits = output[0]
                # Add VQ loss if present
                if len(output) > 2:
                    vq_loss = output[2]
                else:
                    vq_loss = 0.0
            else:
                logits = output
                vq_loss = 0.0

            loss = criterion(logits, y_smooth)

            # Asymmetric FN weighting
            if fn_multiplier != 1.0:
                with torch.no_grad():
                    fn_mask = (y_batch == 1) & (torch.sigmoid(logits) < 0.5)
                    fn_weight = torch.ones_like(y_batch)
                    fn_weight[fn_mask] = fn_multiplier
                loss = (loss * fn_weight.squeeze()).mean() if loss.dim() > 0 else loss

            # Add VQ loss
            if isinstance(vq_loss, torch.Tensor):
                loss = loss + 0.1 * vq_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: Optional[float] = None,
) -> Dict:
    """Evaluate model."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)

            output = model(x_batch)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(y_batch.numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = find_optimal_threshold(all_labels, all_probs)

    all_preds = (all_probs >= threshold).astype(float)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "auc_roc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        "threshold": threshold,
    }

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_path: str, test_size: float = 0.2) -> Tuple:
    """Load EEG data."""
    data_path = Path(data_path)

    # Try to load preprocessed data
    x_path = data_path / "X_windows.npy"
    y_path = data_path / "y_windows.npy"

    if x_path.exists() and y_path.exists():
        print(f"Loading data from {data_path}")
        X = np.load(x_path)
        y = np.load(y_path)
    else:
        # Generate synthetic data for testing
        print("Generating synthetic data for testing...")
        np.random.seed(42)
        n_samples = 5000
        n_channels = 16
        time_steps = 256

        X = np.random.randn(n_samples, n_channels, time_steps).astype(np.float32)
        y = np.random.randint(0, 2, n_samples).astype(np.float32)

        # Add some signal for seizure class
        seizure_idx = y == 1
        X[seizure_idx] += np.sin(np.linspace(0, 10 * np.pi, time_steps)) * 0.5

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Main Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model_name: str,
    data_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    use_pretrained: bool = False,
    pretrained_encoder: str = "cbramod",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
    fn_multiplier: float = 2.0,
    device: Optional[str] = None,
) -> Dict:
    """Train a single model."""
    # Setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"Device: {device}")
    print(f"Pre-trained: {use_pretrained} ({pretrained_encoder})")
    print(f"{'='*60}")

    # Load data
    X_train, X_test, y_train, y_test = load_data(data_path)
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"Class balance: {y_train.mean():.2%} positive")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).float(),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Create model
    n_channels = X_train.shape[1]
    time_steps = X_train.shape[2]

    model = get_model(
        model_name,
        n_channels=n_channels,
        time_steps=time_steps,
        use_pretrained=use_pretrained,
        pretrained_encoder=pretrained_encoder if use_pretrained else None,
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Loss function
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    criterion = FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    use_amp = device.type == "cuda"

    # Training loop
    best_f1 = 0.0
    best_metrics = {}
    history = {"train_loss": [], "val_f1": [], "val_auc": []}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            use_amp=use_amp,
            label_smoothing=label_smoothing,
            fn_multiplier=fn_multiplier,
        )

        # Evaluate
        metrics = evaluate(model, test_loader, device)

        # Update scheduler
        scheduler.step()

        # Log
        history["train_loss"].append(train_loss)
        history["val_f1"].append(metrics["f1"])
        history["val_auc"].append(metrics["auc_roc"])

        # Save best
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_metrics = metrics.copy()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                },
                output_path / f"{model_name}_best.pt",
            )

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"AUC: {metrics['auc_roc']:.4f} | "
                f"Sens: {metrics['sensitivity']:.4f} | "
                f"Spec: {metrics['specificity']:.4f}"
            )

    # Final results
    print(f"\nBest F1: {best_f1:.4f}")
    print(f"Best metrics: {best_metrics}")

    # Save results
    results = {
        "model": model_name,
        "best_metrics": best_metrics,
        "history": history,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "use_pretrained": use_pretrained,
            "pretrained_encoder": pretrained_encoder,
            "focal_gamma": focal_gamma,
            "label_smoothing": label_smoothing,
            "fn_multiplier": fn_multiplier,
        },
    }

    with open(output_path / f"{model_name}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Approach 2 models")
    parser.add_argument(
        "--model",
        type=str,
        default="m3_criss_cross",
        help=f"Model name or 'all'. Available: {list_models()}",
    )
    parser.add_argument("--data_path", type=str, default="../../data/processed")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained encoder")
    parser.add_argument("--encoder", type=str, default="cbramod", 
                       help="Pretrained encoder: cbramod, eegpt, biot, labram")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--fn_multiplier", type=float, default=2.0)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    # Train
    if args.model.lower() == "all":
        models_to_train = list_models()
    else:
        models_to_train = [args.model]

    all_results = {}
    for model_name in models_to_train:
        results = train_model(
            model_name=model_name,
            data_path=args.data_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            use_pretrained=args.pretrained,
            pretrained_encoder=args.encoder,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
            fn_multiplier=args.fn_multiplier,
            device=args.device,
        )
        all_results[model_name] = results

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for name, res in all_results.items():
        print(f"{name}: F1={res['best_metrics']['f1']:.4f}, AUC={res['best_metrics']['auc_roc']:.4f}")


if __name__ == "__main__":
    main()
