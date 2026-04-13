"""
compare.py - Model Comparison Framework
==========================================
Trains all five LSTM architectures and generates a comparison report.
Uses state-of-the-art improvements: Focal Loss, label smoothing, asymmetric FN weighting,
learning rate warmup, AdamW + CosineAnnealing, Youden's J threshold.

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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, roc_curve,
)
from models import MODEL_REGISTRY


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Reduces loss for easy examples and focuses on hard examples.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Compare all LSTM architectures")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory with raw EEG tensors (windows.pt + labels.pt)")
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="Directory with feature tensors for feature_bilstm. "
                             "If not provided, feature_bilstm is skipped.")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Directory with test set tensors for final evaluation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter (0=disabled, use standard BCE)")
    parser.add_argument("--fn_multiplier", type=float, default=2.0,
                        help="Multiply loss on positive samples to penalize FN more")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                        help="Label smoothing: 0->label_smoothing, 1->(1-label_smoothing)")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear warmup epochs before cosine annealing")
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


def find_optimal_threshold(y_true, y_prob):
    """Youden's J: threshold maximizing sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return float(thresholds[int(np.argmax(j_scores))])


def get_lr_with_warmup(epoch, base_lr, warmup_epochs):
    """Linear warmup for first warmup_epochs, then 1.0 (for cosine scheduler to handle)."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


def compute_metrics(y_true, y_pred, y_prob):
    """Accuracy, sensitivity, specificity, precision, F1, AUC-ROC."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc":     float(roc_auc_score(y_true, y_prob))
                       if len(np.unique(y_true)) > 1 else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def make_loaders(X, y, val_split, batch_size, seed):
    """Create train/val DataLoaders. Class imbalance handled via pos_weight in loss."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size,
        shuffle=True, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size,
        shuffle=False, pin_memory=True,
    )

    pos_count = y.sum().item()
    neg_count = len(y) - pos_count
    pos_weight = neg_count / max(pos_count, 1)

    return train_loader, val_loader, pos_weight


def train_and_evaluate(model, train_loader, val_loader, pos_weight, epochs,
                       patience, lr, device, focal_gamma=2.0, fn_multiplier=2.0,
                       label_smoothing=0.05, warmup_epochs=5):
    """Full training loop with modern improvements, returns best validation metrics."""
    # Loss function: Focal Loss for handling class imbalance
    if focal_gamma > 0:
        criterion = FocalLoss(gamma=focal_gamma,
                             pos_weight=torch.tensor([pos_weight]).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs - warmup_epochs), T_mult=1, eta_min=1e-6
    )

    best_f1 = 0.0
    best_metrics = None
    wait = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Learning rate warmup
        if epoch <= warmup_epochs:
            lr_scale = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr * lr_scale

        # Train
        model.train()
        for X_b, y_b in train_loader:
            X_b = X_b.to(device, dtype=torch.float32)
            y_b = y_b.to(device, dtype=torch.float32).unsqueeze(1)

            # Apply label smoothing
            y_smooth = y_b * (1.0 - label_smoothing) + 0.5 * label_smoothing

            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_smooth)

            # Asymmetric FN weighting: penalize missed seizures more
            seizure_mask = (y_b == 1).float()
            loss = loss * (1.0 + (fn_multiplier - 1.0) * seizure_mask.squeeze(1))
            loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Cosine scheduler step (after warmup)
        if epoch > warmup_epochs:
            scheduler.step()

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device, dtype=torch.float32)
                y_b = y_b.to(device, dtype=torch.float32).unsqueeze(1)
                logits = model(X_b)
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(y_b.cpu())

        probs = torch.cat(all_probs).numpy().flatten()
        labels = torch.cat(all_labels).numpy().flatten()

        # Find optimal threshold using Youden's J
        threshold = find_optimal_threshold(labels, probs)
        preds = (probs >= threshold).astype(float)

        metrics = compute_metrics(labels, preds, probs)
        metrics["epoch"] = epoch
        metrics["threshold"] = threshold

        # Early stopping on F1
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
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
            focal_gamma=args.focal_gamma,
            fn_multiplier=args.fn_multiplier,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
        )
        all_results[model_name] = metrics

        print(f"  -> Acc: {metrics['accuracy']:.3f} | "
              f"Sens: {metrics['sensitivity']:.3f} | "
              f"F1: {metrics['f1']:.3f} | "
              f"AUC: {metrics['auc_roc']:.3f} | "
              f"Threshold: {metrics['threshold']:.4f} | "
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
            focal_gamma=args.focal_gamma,
            fn_multiplier=args.fn_multiplier,
            label_smoothing=args.label_smoothing,
            warmup_epochs=args.warmup_epochs,
        )
        all_results[model_name] = metrics

        print(f"  -> Acc: {metrics['accuracy']:.3f} | "
              f"Sens: {metrics['sensitivity']:.3f} | "
              f"F1: {metrics['f1']:.3f} | "
              f"AUC: {metrics['auc_roc']:.3f} | "
              f"Threshold: {metrics['threshold']:.4f} | "
              f"Time: {metrics['train_time_sec']}s")
    else:
        print(f"\n  Skipping feature_bilstm (no --feature_dir provided)")

    # ==================== Comparison Report ====================
    print(f"\n\n{'='*80}")
    print(f"  COMPARISON REPORT")
    print(f"{'='*80}\n")

    header = (f"{'Model':<22} | {'F1':>6} | {'Sens':>6} | {'Spec':>6} | "
              f"{'AUC':>6} | {'Threshold':>9} | {'Params':>10}")
    print(header)
    print("-" * len(header))

    # Sort by F1 score descending
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]["f1"], reverse=True)

    for name, m in sorted_models:
        print(f"{name:<22} | {m['f1']:>6.3f} | {m['sensitivity']:>6.3f} | "
              f"{m['specificity']:>6.3f} | {m['auc_roc']:>6.3f} | "
              f"{m['threshold']:>9.4f} | {m['total_params']:>10,}")

    best_name = sorted_models[0][0]
    best_model_metrics = sorted_models[0][1]
    print(f"\n  >>> Best model by F1: {best_name} "
          f"(F1={best_model_metrics['f1']:.4f}, AUC={best_model_metrics['auc_roc']:.4f})")
    print(f"      Threshold: {best_model_metrics['threshold']:.4f}")
    print(f"      Sensitivity: {best_model_metrics['sensitivity']:.4f}, "
          f"Specificity: {best_model_metrics['specificity']:.4f}")

    # Save results
    results_path = os.path.join(args.save_dir, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
