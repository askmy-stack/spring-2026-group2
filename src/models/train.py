"""
train.py - Unified Training Script
====================================
Trains any of the five LSTM architectures on pre-saved EEG tensors.
Evaluates on both validation AND held-out test split for real generalisation scores.

Usage:
    python train.py --model vanilla_lstm   --data_dir ../results/tensors/chbmit/train --test_dir ../results/tensors/chbmit/test --epochs 50
    python train.py --model cnn_lstm       --data_dir ../results/tensors/chbmit/train --test_dir ../results/tensors/chbmit/test --epochs 50
    python train.py --model feature_bilstm --data_dir ../results/tensors/chbmit/train --test_dir ../results/tensors/chbmit/test --epochs 50
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
from architectures import MODEL_REGISTRY


# ======================== Focal Loss ========================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    FL(p) = -(1-p)^gamma * log(p)

    gamma=0  -> standard BCE
    gamma=2  -> down-weights easy examples, focuses on hard seizure windows
    Combined with pos_weight for class imbalance handling.
    """
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)                    # (batch, 1)
        probs = torch.sigmoid(logits)
        # p_t: probability of the true class
        p_t = torch.where(targets == 1, probs, 1.0 - probs)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()


# ======================== Config ========================

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM seizure classifier")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture to train")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing pre-saved tensor .pt files (train split)")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Directory containing held-out test tensors. "
                             "If provided, final model is evaluated on test set too.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (tracks val F1)")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if not set)")
    parser.add_argument("--n_features", type=int, default=226,
                        help="Number of features (for feature_bilstm only)")
    # Loss improvements
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma. 0 = standard BCE. Default 2.0.")
    parser.add_argument("--fn_multiplier", type=float, default=2.0,
                        help="Extra penalty multiplier for false negatives (missed seizures). "
                             "1.0 = no asymmetry. Default 2.0.")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                        help="Label smoothing epsilon. 0 = hard labels. Default 0.05.")
    # LR schedule
    parser.add_argument("--warmup_epochs", type=int, default=5,
                        help="Linear LR warmup epochs before cosine annealing. Default 5.")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision training (AMP)")
    return parser.parse_args()


# ======================== Data Loading ========================

def load_tensor_data(data_dir: str):
    """Load pre-saved EEG tensors from directory."""
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
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])
        raise FileNotFoundError(
            f"Could not find data tensors in {data_dir}. "
            f"Expected (windows.pt + labels.pt) or (X.pt + y.pt). "
            f"Found files: {files}"
        )
    return X, y


def create_dataloaders(X, y, val_split: float, batch_size: int, seed: int):
    """
    Split data into train/val and create DataLoaders.
    Class imbalance handled via pos_weight in loss — no WeightedRandomSampler
    (avoids double-weighting the minority class).
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )

    print(f"\n  Train: {len(X_train)} samples "
          f"(seizure={y_train.sum().item()}, "
          f"non-seizure={len(y_train) - y_train.sum().item()})")
    print(f"  Val:   {len(X_val)} samples "
          f"(seizure={y_val.sum().item()}, "
          f"non-seizure={len(y_val) - y_val.sum().item()})")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader


def make_test_loader(X_test, y_test, batch_size: int):
    return DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )


# ======================== Loss helpers ========================

def apply_label_smoothing(y_batch: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Soft labels: 0 -> eps/2, 1 -> 1 - eps/2."""
    if epsilon == 0.0:
        return y_batch
    return y_batch * (1.0 - epsilon) + 0.5 * epsilon


def apply_fn_multiplier(loss_per_sample: torch.Tensor,
                        y_batch: torch.Tensor,
                        fn_multiplier: float) -> torch.Tensor:
    """
    Boost loss on positive (seizure) samples by fn_multiplier.
    Makes false negatives more costly than false positives.
    fn_multiplier=1.0 -> no effect.
    """
    if fn_multiplier == 1.0:
        return loss_per_sample
    seizure_mask = (y_batch >= 0.5).float()
    weight = 1.0 + (fn_multiplier - 1.0) * seizure_mask
    return loss_per_sample * weight


# ======================== LR Warmup ========================

def get_lr_with_warmup(epoch: int, warmup_epochs: int, base_lr: float) -> float:
    """Linear warmup: epochs 1..warmup_epochs ramp from base_lr/warmup to base_lr."""
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return None  # Let the cosine scheduler handle it
    return base_lr * (epoch + 1) / warmup_epochs


# ======================== Training Loop ========================

def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, use_amp, fn_multiplier, label_smoothing):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.to(device, dtype=torch.float32).unsqueeze(1)

        # Label smoothing
        y_smooth = apply_label_smoothing(y_batch, label_smoothing)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                logits = model(X_batch)
                loss = criterion(logits, y_smooth)
        else:
            logits = model(X_batch)
            loss = criterion(logits, y_smooth)

        # Asymmetric FN weighting — recalculate per-sample then mean
        # Note: FocalLoss already returns scalar mean; we re-weight per sample
        # by computing raw per-sample BCE then applying FN weight
        if fn_multiplier != 1.0:
            with torch.no_grad():
                bce_per = nn.functional.binary_cross_entropy_with_logits(
                    logits, y_smooth, reduction="none"
                )
                probs = torch.sigmoid(logits.detach())
                p_t = torch.where(y_smooth >= 0.5, probs, 1.0 - probs)
                gamma = getattr(criterion, "gamma", 0.0)
                fl_weight = (1.0 - p_t) ** gamma
                per_sample = fl_weight * bce_per
                per_sample = apply_fn_multiplier(per_sample, y_batch, fn_multiplier)
                loss = per_sample.mean()

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    """Evaluate model — returns loss + full metrics dict."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, dtype=torch.float32)
        y_batch = y_batch.to(device, dtype=torch.float32).unsqueeze(1)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        probs = torch.sigmoid(logits).cpu()
        total_loss += loss.item()
        n_batches += 1
        all_probs.append(probs)
        all_labels.append(y_batch.cpu())

    all_probs = torch.cat(all_probs).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()
    all_preds = (all_probs >= threshold).astype(float)

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / n_batches
    return metrics


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


def find_optimal_threshold(y_true, y_prob) -> float:
    """Youden's J: threshold maximising sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    return float(thresholds[int(np.argmax(j_scores))])


@torch.no_grad()
def _collect_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, dtype=torch.float32)
        logits = model(X_batch)
        all_probs.append(torch.sigmoid(logits).cpu().numpy().flatten())
        all_labels.append(y_batch.numpy().flatten())
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ======================== Main ========================

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    use_amp = (not args.no_amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler() if use_amp else None

    print(f"\n{'='*65}")
    print(f"  Training: {args.model}")
    print(f"  Device:   {device}  |  AMP: {use_amp}")
    print(f"  Focal γ:  {args.focal_gamma}  |  FN multiplier: {args.fn_multiplier}")
    print(f"  Label smoothing: {args.label_smoothing}  |  LR warmup: {args.warmup_epochs} epochs")
    print(f"{'='*65}")

    # ── Load train data ──
    print("\nLoading train data...")
    X, y = load_tensor_data(args.data_dir)
    train_loader, val_loader = create_dataloaders(
        X, y, args.val_split, args.batch_size, args.seed
    )

    # ── Load test data (optional held-out) ──
    test_loader = None
    if args.test_dir and os.path.isdir(args.test_dir):
        print("\nLoading held-out test data...")
        X_test, y_test = load_tensor_data(args.test_dir)
        test_loader = make_test_loader(X_test, y_test, args.batch_size)
        print(f"  Test: {len(X_test)} samples "
              f"(seizure={y_test.sum().item()}, "
              f"non-seizure={len(y_test) - y_test.sum().item()})")

    # ── Build model ──
    if args.model == "feature_bilstm":
        model_kwargs = {
            "n_features":  args.n_features,
            "hidden_size": args.hidden_size,
            "num_layers":  args.num_layers,
            "dropout":     args.dropout,
        }
    else:
        n_channels = X.shape[1]
        seq_len    = X.shape[2]
        model_kwargs = {
            "n_channels":  n_channels,
            "seq_len":     seq_len,
            "hidden_size": args.hidden_size,
            "num_layers":  args.num_layers,
            "dropout":     args.dropout,
        }

    ModelClass = MODEL_REGISTRY[args.model]
    model = ModelClass(**model_kwargs).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model params: {trainable_params:,} trainable / {total_params:,} total")

    # ── Loss: Focal + pos_weight ──
    n_pos = y.sum().item()
    n_neg = len(y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    criterion = FocalLoss(gamma=args.focal_gamma, pos_weight=pos_weight)
    print(f"  Pos weight:   {pos_weight.item():.2f}  |  Focal γ: {args.focal_gamma}")

    # ── Optimizer + Cosine scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_f1   = -1.0
    best_metrics  = None
    best_threshold = 0.5
    patience_counter = 0
    history  = []
    ckpt_path = os.path.join(args.save_dir, f"{args.model}_best.pt")

    print(f"\n{'Epoch':>6} | {'TrainLoss':>10} | {'ValLoss':>9} | "
          f"{'Acc':>6} | {'Sens':>6} | {'Spec':>6} | {'F1':>6} | {'AUC':>6} | {'Thresh':>7}")
    print("-" * 97)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── LR warmup override ──
        warmup_lr = get_lr_with_warmup(epoch - 1, args.warmup_epochs, args.lr)
        if warmup_lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_amp, args.fn_multiplier, args.label_smoothing,
        )

        # Only step cosine scheduler after warmup
        if epoch > args.warmup_epochs:
            scheduler.step(epoch - args.warmup_epochs)

        val_metrics = evaluate(model, val_loader, criterion, device, threshold=0.5)
        elapsed = time.time() - t0

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_metrics['loss']:>9.4f} | "
              f"{val_metrics['accuracy']:>6.3f} | {val_metrics['sensitivity']:>6.3f} | "
              f"{val_metrics['specificity']:>6.3f} | {val_metrics['f1']:>6.3f} | "
              f"{val_metrics['auc_roc']:>6.3f} | {'0.500':>7}  ({elapsed:.1f}s)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{k: v for k, v in val_metrics.items()},
        })

        # ── Early stopping on val F1 ──
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0

            # Optimal threshold via Youden's J on val
            all_probs, all_labels = _collect_probs(model, val_loader, device)
            optimal_threshold = find_optimal_threshold(all_labels, all_probs)
            best_metrics = evaluate(
                model, val_loader, criterion, device, threshold=optimal_threshold
            )
            best_threshold = optimal_threshold

            torch.save({
                "epoch":             epoch,
                "model_state_dict":  model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics":       best_metrics,
                "optimal_threshold": optimal_threshold,
                "args":              vars(args),
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no F1 improvement for {args.patience} epochs)")
                break

    # ── Validation summary ──
    print(f"\n{'='*65}")
    print(f"  VALIDATION RESULTS — {args.model}")
    print(f"  Optimal threshold: {best_threshold:.4f}  (Youden's J)")
    print(f"{'='*65}")
    _print_metrics(best_metrics)
    print(f"\n  Checkpoint: {ckpt_path}")

    # ── Test set evaluation (held-out, never seen during training) ──
    if test_loader is not None:
        print(f"\n{'='*65}")
        print(f"  TEST SET RESULTS — {args.model}  (held-out, real generalisation)")
        print(f"  Using threshold: {best_threshold:.4f}  (from val Youden's J)")
        print(f"{'='*65}")
        test_metrics = evaluate(model, test_loader, criterion, device,
                                threshold=best_threshold)
        _print_metrics(test_metrics)

        # Save test results alongside checkpoint
        test_path = os.path.join(args.save_dir, f"{args.model}_test_results.json")
        with open(test_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\n  Test results saved: {test_path}")

    # ── Save training history ──
    hist_path = os.path.join(args.save_dir, f"{args.model}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History:    {hist_path}\n")


def _print_metrics(m: dict):
    print(f"  Accuracy:    {m['accuracy']:.4f}")
    print(f"  Sensitivity: {m['sensitivity']:.4f}  (seizure recall — catch rate)")
    print(f"  Specificity: {m['specificity']:.4f}")
    print(f"  Precision:   {m['precision']:.4f}")
    print(f"  F1 Score:    {m['f1']:.4f}")
    print(f"  AUC-ROC:     {m['auc_roc']:.4f}")
    print(f"  Confusion:   TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")


if __name__ == "__main__":
    main()
