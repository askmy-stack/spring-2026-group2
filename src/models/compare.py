"""
compare.py - Model Comparison Framework
==========================================
Trains all five LSTM architectures and generates a comparison report.

Usage:
    python compare.py --data_dir ./tensors/chb01 --epochs 30
    python compare.py --data_dir ./tensors/chb01 --feature_dir ./features/chb01 --epochs 50
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
)
from models import MODEL_REGISTRY


def parse_args():
    parser = argparse.ArgumentParser(description="Compare all LSTM architectures")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with raw EEG tensors (windows.pt + labels.pt)")
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="Directory with feature tensors for feature_bilstm. "
                             "If not provided, feature_bilstm is skipped.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./comparison_results")
    parser.add_argument("--n_features", type=int, default=226)
    return parser.parse_args()


def load_tensors(data_dir):
    """Load X and y tensors from directory."""
    x_names = ["windows.pt", "X.pt", "data.pt"]
    y_names = ["labels.pt", "y.pt", "targets.pt"]

    X, y = None, None
    for name in x_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            X = torch.load(path, weights_only=False)
            break
    for name in y_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            y = torch.load(path, weights_only=False)
            break

    if X is None or y is None:
        raise FileNotFoundError(f"Cannot find tensor files in {data_dir}")
    return X, y


def make_loaders(X, y, val_split, batch_size, seed):
    """Create train/val DataLoaders with balanced sampling."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    class_counts = torch.bincount(y_tr.long())
    weights = 1.0 / class_counts.float()
    sample_w = weights[y_tr.long()]
    sampler = WeightedRandomSampler(sample_w, len(sample_w))

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size,
        sampler=sampler, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size,
        shuffle=False, pin_memory=True,
    )

    pos_count = y.sum().item()
    neg_count = len(y) - pos_count
    pos_weight = neg_count / max(pos_count, 1)

    return train_loader, val_loader, pos_weight


def train_and_evaluate(model, train_loader, val_loader, pos_weight,
                       epochs, patience, lr, device):
    """Full training loop, returns best validation metrics and training time."""
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_loss = float("inf")
    best_metrics = None
    wait = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for X_b, y_b in train_loader:
            X_b = X_b.to(device, dtype=torch.float32)
            y_b = y_b.to(device, dtype=torch.float32).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        val_loss_sum, n = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device, dtype=torch.float32)
                y_b = y_b.to(device, dtype=torch.float32).unsqueeze(1)
                logits = model(X_b)
                val_loss_sum += criterion(logits, y_b).item()
                n += 1
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(y_b.cpu())

        val_loss = val_loss_sum / n
        scheduler.step(val_loss)

        probs = torch.cat(all_probs).numpy().flatten()
        labels = torch.cat(all_labels).numpy().flatten()
        preds = (probs >= 0.5).astype(float)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "sensitivity": recall_score(labels, preds, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "precision": precision_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "auc_roc": roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
            "val_loss": val_loss,
            "epoch": epoch,
        }

        if val_loss < best_loss:
            best_loss = val_loss
            best_metrics = metrics.copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    train_time = time.time() - start_time
    best_metrics["train_time_sec"] = round(train_time, 1)
    best_metrics["total_params"] = sum(p.numel() for p in model.parameters())
    return best_metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # Load raw EEG data
    print("\nLoading raw EEG tensors...")
    X_raw, y_raw = load_tensors(args.data_dir)
    n_channels = X_raw.shape[1]
    seq_len = X_raw.shape[2]
    print(f"  Shape: {X_raw.shape}, Labels: {y_raw.shape}")
    print(f"  Seizure: {y_raw.sum().item()}, Non-seizure: {len(y_raw) - y_raw.sum().item()}")

    raw_train, raw_val, pos_w = make_loaders(
        X_raw, y_raw, args.val_split, args.batch_size, args.seed
    )

    # Load feature data if provided
    feat_train, feat_val, feat_pos_w = None, None, None
    if args.feature_dir and os.path.isdir(args.feature_dir):
        print("\nLoading feature tensors...")
        X_feat, y_feat = load_tensors(args.feature_dir)
        print(f"  Shape: {X_feat.shape}, Labels: {y_feat.shape}")
        feat_train, feat_val, feat_pos_w = make_loaders(
            X_feat, y_feat, args.val_split, args.batch_size, args.seed
        )

    # Models to compare
    raw_models = ["vanilla_lstm", "bilstm", "attention_bilstm", "cnn_lstm"]
    all_results = {}

    # Train raw-input models
    for model_name in raw_models:
        print(f"\n{'='*60}")
        print(f"  Training: {model_name}")
        print(f"{'='*60}")

        ModelClass = MODEL_REGISTRY[model_name]
        model = ModelClass(
            n_channels=n_channels, seq_len=seq_len,
            hidden_size=args.hidden_size, num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")

        metrics = train_and_evaluate(
            model, raw_train, raw_val, pos_w,
            args.epochs, args.patience, args.lr, device,
        )
        all_results[model_name] = metrics

        print(f"  -> Acc: {metrics['accuracy']:.3f} | "
              f"Sens: {metrics['sensitivity']:.3f} | "
              f"F1: {metrics['f1']:.3f} | "
              f"AUC: {metrics['auc_roc']:.3f} | "
              f"Time: {metrics['train_time_sec']}s")

    # Train feature-based model
    if feat_train is not None:
        model_name = "feature_bilstm"
        print(f"\n{'='*60}")
        print(f"  Training: {model_name}")
        print(f"{'='*60}")

        ModelClass = MODEL_REGISTRY[model_name]
        model = ModelClass(
            n_features=args.n_features,
            hidden_size=args.hidden_size, num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")

        metrics = train_and_evaluate(
            model, feat_train, feat_val, feat_pos_w,
            args.epochs, args.patience, args.lr, device,
        )
        all_results[model_name] = metrics

        print(f"  -> Acc: {metrics['accuracy']:.3f} | "
              f"Sens: {metrics['sensitivity']:.3f} | "
              f"F1: {metrics['f1']:.3f} | "
              f"AUC: {metrics['auc_roc']:.3f} | "
              f"Time: {metrics['train_time_sec']}s")
    else:
        print(f"\n  Skipping feature_bilstm (no --feature_dir provided)")

    # ==================== Comparison Report ====================
    print(f"\n\n{'='*80}")
    print(f"  COMPARISON REPORT")
    print(f"{'='*80}\n")

    header = (f"{'Model':<22} | {'Acc':>6} | {'Sens':>6} | {'Spec':>6} | "
              f"{'F1':>6} | {'AUC':>6} | {'Params':>10} | {'Time':>7}")
    print(header)
    print("-" * len(header))

    # Sort by F1 score descending
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]["f1"], reverse=True)

    for name, m in sorted_models:
        print(f"{name:<22} | {m['accuracy']:>6.3f} | {m['sensitivity']:>6.3f} | "
              f"{m['specificity']:>6.3f} | {m['f1']:>6.3f} | {m['auc_roc']:>6.3f} | "
              f"{m['total_params']:>10,} | {m['train_time_sec']:>6.1f}s")

    best_name = sorted_models[0][0]
    print(f"\n  >>> Best model by F1: {best_name} "
          f"(F1={sorted_models[0][1]['f1']:.4f}, "
          f"AUC={sorted_models[0][1]['auc_roc']:.4f})")

    # Save results
    results_path = os.path.join(args.save_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
