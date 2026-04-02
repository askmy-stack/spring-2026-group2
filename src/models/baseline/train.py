"""
train.py - Unified Training Script
====================================
Trains any of the five LSTM architectures on pre-saved EEG tensors.

Usage:
    python train.py --model vanilla_lstm --data_dir ./tensors/chb01 --epochs 30
    python train.py --model attention_bilstm --data_dir ./tensors/chb01 --epochs 50
    python train.py --model feature_bilstm --data_dir ./features/chb01 --epochs 30
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)
from __init__ import MODEL_REGISTRY


# ======================== Config ========================

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM seizure classifier")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture to train")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing pre-saved tensor .pt files")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    # Feature-based model specific
    parser.add_argument("--n_features", type=int, default=226,
                        help="Number of features (for feature_bilstm only)")
    return parser.parse_args()


# ======================== Data Loading ========================

def load_tensor_data(data_dir):
    """
    Load pre-saved EEG tensors from directory.
    Expects files like: windows.pt (or X.pt) and labels.pt (or y.pt)
    Adapt this function to match your actual tensor file naming.
    """
    # Try common naming conventions
    x_names = ["windows.pt", "X.pt", "data.pt", "eeg_windows.pt"]
    y_names = ["labels.pt", "y.pt", "targets.pt"]

    X, y = None, None

    for name in x_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            X = torch.load(path, weights_only=False)
            print(f"  Loaded features: {path} -> shape {X.shape}")
            break

    for name in y_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            y = torch.load(path, weights_only=False)
            print(f"  Loaded labels:   {path} -> shape {y.shape}")
            break

    if X is None or y is None:
        # Fallback: load individual .pt files and stack them
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])
        raise FileNotFoundError(
            f"Could not find data tensors in {data_dir}. "
            f"Expected (windows.pt + labels.pt) or (X.pt + y.pt). "
            f"Found files: {files}"
        )

    return X, y


def create_dataloaders(X, y, val_split, batch_size, seed):
    """Split data and create balanced DataLoaders."""
    # Train/val split with stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    print(f"\n  Train: {len(X_train)} samples "
          f"(seizure: {y_train.sum().item()}, "
          f"non-seizure: {len(y_train) - y_train.sum().item()})")
    print(f"  Val:   {len(X_val)} samples "
          f"(seizure: {y_val.sum().item()}, "
          f"non-seizure: {len(y_val) - y_val.sum().item()})")

    # Weighted sampler to handle class imbalance
    class_counts = torch.bincount(y_train.long())
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[y_train.long()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    return train_loader, val_loader


# ======================== Training Loop ========================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.to(device, dtype=torch.float32).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model, return loss and all predictions/labels."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_probs = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.to(device, dtype=torch.float32).unsqueeze(1)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        probs = torch.sigmoid(logits).cpu()
        preds = (probs >= 0.5).float()

        total_loss += loss.item()
        n_batches += 1
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(y_batch.cpu())

    all_probs = torch.cat(all_probs).numpy().flatten()
    all_preds = torch.cat(all_preds).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()

    avg_loss = total_loss / n_batches

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = avg_loss

    return metrics


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),  # seizure recall
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    return metrics


# ======================== Main ========================

def main():
    args = parse_args()

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n{'='*60}")
    print(f"  Training: {args.model}")
    print(f"  Device:   {device}")
    print(f"{'='*60}")

    # Load data
    print("\nLoading data...")
    X, y = load_tensor_data(args.data_dir)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X, y, args.val_split, args.batch_size, args.seed
    )

    # Determine input dimensions
    if args.model == "feature_bilstm":
        n_channels = args.n_features  # not used internally, but for compat
        seq_len = 1
        model_kwargs = {
            "n_features": args.n_features,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        }
    else:
        # Raw tensor models: X shape is (samples, channels, time_steps)
        n_channels = X.shape[1]
        seq_len = X.shape[2]
        model_kwargs = {
            "n_channels": n_channels,
            "seq_len": seq_len,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        }

    # Build model
    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(**model_kwargs).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model params: {trainable_params:,} trainable / {total_params:,} total")

    # Loss with class-weight for imbalanced data
    n_pos = y.sum().item()
    n_neg = len(y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  Pos weight:   {pos_weight.item():.2f} (neg/pos ratio)")

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Training loop with early stopping
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_metrics = None
    patience_counter = 0
    history = []

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | "
          f"{'Acc':>6} | {'Sens':>6} | {'Spec':>6} | {'F1':>6} | {'AUC':>6}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - t0

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_metrics['loss']:>10.4f} | "
              f"{val_metrics['accuracy']:>6.3f} | {val_metrics['sensitivity']:>6.3f} | "
              f"{val_metrics['specificity']:>6.3f} | {val_metrics['f1']:>6.3f} | "
              f"{val_metrics['auc_roc']:>6.3f}  ({elapsed:.1f}s)")

        # Track history
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            **{k: v for k, v in val_metrics.items() if k != "loss"},
        })

        # Early stopping check
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics.copy()
            patience_counter = 0

            # Save best model
            ckpt_path = os.path.join(args.save_dir, f"{args.model}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Final summary
    print(f"\n{'='*60}")
    print(f"  Best Validation Results ({args.model})")
    print(f"{'='*60}")
    print(f"  Loss:        {best_metrics['loss']:.4f}")
    print(f"  Accuracy:    {best_metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {best_metrics['sensitivity']:.4f}  (seizure recall)")
    print(f"  Specificity: {best_metrics['specificity']:.4f}")
    print(f"  Precision:   {best_metrics['precision']:.4f}")
    print(f"  F1 Score:    {best_metrics['f1']:.4f}")
    print(f"  AUC-ROC:     {best_metrics['auc_roc']:.4f}")
    print(f"  Confusion:   TP={best_metrics['tp']} FP={best_metrics['fp']} "
          f"TN={best_metrics['tn']} FN={best_metrics['fn']}")
    print(f"\n  Checkpoint:  {ckpt_path}")

    # Save training history
    hist_path = os.path.join(args.save_dir, f"{args.model}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History:     {hist_path}\n")


if __name__ == "__main__":
    main()
