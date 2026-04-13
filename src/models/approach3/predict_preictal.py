#!/usr/bin/env python3
"""
Pre-ictal Prediction: 30-60 Minute Seizure Forecasting
------------------------------------------------------
Predict seizure probability in the upcoming 30-60 minutes.

Usage:
    python predict_preictal.py --horizon 30 --context 60 --epochs 50
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from architectures import HierarchicalLSTM, PreIctalPredictor


def create_preictal_dataset(
    X: np.ndarray,
    y: np.ndarray,
    context_windows: int = 60,
    horizon_windows: int = 30,
) -> tuple:
    """
    Create pre-ictal prediction dataset.

    For each seizure, create a sample using context_windows before
    the seizure onset, with label = 1 (seizure coming in horizon_windows).

    Args:
        X: EEG windows (n_samples, n_channels, time_steps)
        y: Labels (n_samples,)
        context_windows: Number of context windows
        horizon_windows: Prediction horizon in windows

    Returns:
        X_context: Context windows (n_samples, context_windows, n_channels, time_steps)
        y_preictal: Pre-ictal labels
    """
    n_samples, n_channels, time_steps = X.shape

    X_context = []
    y_preictal = []

    # Find seizure onset indices
    seizure_onsets = []
    for i in range(1, len(y)):
        if y[i] == 1 and y[i - 1] == 0:
            seizure_onsets.append(i)

    # Create pre-ictal samples
    for onset in seizure_onsets:
        # Pre-ictal: horizon_windows before seizure
        start = onset - horizon_windows - context_windows
        end = onset - horizon_windows

        if start >= 0:
            context = X[start:end]
            if len(context) == context_windows:
                X_context.append(context)
                y_preictal.append(1)  # Seizure coming

    # Create interictal (normal) samples
    n_normal = len(X_context) * 2  # 2:1 ratio
    normal_indices = np.where(y == 0)[0]

    for _ in range(min(n_normal, len(normal_indices) - context_windows)):
        start = np.random.choice(normal_indices[:-context_windows])
        # Ensure no seizure in next horizon_windows
        if (y[start:start + context_windows + horizon_windows] == 0).all():
            context = X[start:start + context_windows]
            if len(context) == context_windows:
                X_context.append(context)
                y_preictal.append(0)

    X_context = np.array(X_context)
    y_preictal = np.array(y_preictal)

    # Shuffle
    indices = np.random.permutation(len(y_preictal))
    X_context = X_context[indices]
    y_preictal = y_preictal[indices]

    return X_context, y_preictal


def train_preictal(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training pre-ictal predictor on {device}")
    print(f"Prediction horizon: {args.horizon} windows")
    print(f"Context length: {args.context} windows")

    # Load data
    data_path = Path(args.data_path)
    if (data_path / "X_windows.npy").exists():
        X = np.load(data_path / "X_windows.npy")
        y = np.load(data_path / "y_windows.npy")
        print(f"Loaded {len(X)} samples")
    else:
        print("Generating synthetic data...")
        np.random.seed(42)
        n_samples = 10000
        X = np.random.randn(n_samples, 16, 256).astype(np.float32)
        y = np.zeros(n_samples, dtype=np.float32)
        # Add seizure events
        for i in range(0, n_samples, 500):
            if i + 50 < n_samples:
                y[i:i + 50] = 1  # 50-window seizure every 500 windows

    # Create pre-ictal dataset
    print("Creating pre-ictal dataset...")
    X_context, y_preictal = create_preictal_dataset(
        X, y,
        context_windows=args.context,
        horizon_windows=args.horizon,
    )
    print(f"Pre-ictal samples: {len(y_preictal)}")
    print(f"Positive rate: {y_preictal.mean():.2%}")

    if len(y_preictal) < 10:
        print("Not enough samples for training. Using synthetic data...")
        np.random.seed(42)
        X_context = np.random.randn(200, args.context, 16, 256).astype(np.float32)
        y_preictal = np.random.randint(0, 2, 200).astype(np.float32)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_context, y_preictal, test_size=0.2, random_state=42, stratify=y_preictal
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size
    )

    # Model
    model = HierarchicalLSTM(
        n_channels=16,
        time_steps=256,
        n_windows=args.context,
        hidden_size=128,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for x, y_batch in train_loader:
            x = x.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y_batch in test_loader:
                x = x.to(device)
                probs = torch.sigmoid(model(x)).cpu().numpy()
                all_probs.extend(probs.flatten())
                all_labels.extend(y_batch.numpy().flatten())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs >= 0.5).astype(float)

        f1 = f1_score(all_labels, preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), output_path / f"preictal_{args.horizon}min_best.pt")

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    print(f"\nTraining complete!")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Model can predict seizures {args.horizon} windows ahead with F1={best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../data/processed")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--horizon", type=int, default=30, help="Prediction horizon in windows")
    parser.add_argument("--context", type=int, default=60, help="Context length in windows")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train_preictal(args)


if __name__ == "__main__":
    main()
