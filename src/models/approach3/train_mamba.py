#!/usr/bin/env python3
"""
Approach 3: Mamba Training Script
---------------------------------
Train Mamba-based models for seizure detection.

Usage:
    python train_mamba.py --model eeg_mamba --epochs 50
    python train_mamba.py --model eeg_mamba_moe --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from architectures import get_model, list_models


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        return ((1 - pt) ** self.gamma * bce_loss).mean()


def load_data(data_path: str) -> Tuple:
    data_path = Path(data_path)
    x_path = data_path / "X_windows.npy"
    y_path = data_path / "y_windows.npy"

    if x_path.exists() and y_path.exists():
        X = np.load(x_path)
        y = np.load(y_path)
    else:
        print("Generating synthetic data...")
        np.random.seed(42)
        X = np.random.randn(5000, 16, 256).astype(np.float32)
        y = np.random.randint(0, 2, 5000).astype(np.float32)
        X[y == 1] += np.sin(np.linspace(0, 10 * np.pi, 256)) * 0.5

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training {args.model} on {device}")

    # Load data
    X_train, X_test, y_train, y_test = load_data(args.data_path)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size
    )

    # Create model
    model = get_model(args.model, n_channels=16, time_steps=256).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    pos_weight = torch.tensor([y_train.sum() / (len(y_train) - y_train.sum())]).to(device)
    criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_f1 = 0.0
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_aux_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()

            # Handle MoE models
            if "moe" in args.model:
                logits, aux_loss = model(x, return_aux_loss=True)
                loss = criterion(logits, y) + 0.01 * aux_loss
                total_aux_loss += aux_loss.item()
            else:
                logits = model(x)
                loss = criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                all_probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
                all_labels.extend(y.numpy().flatten())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs >= 0.5).astype(float)

        f1 = f1_score(all_labels, preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), output_path / f"{args.model}_best.pt")

        if (epoch + 1) % 5 == 0:
            aux_str = f" | Aux: {total_aux_loss/len(train_loader):.4f}" if "moe" in args.model else ""
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f}{aux_str} | F1: {f1:.4f} | AUC: {auc:.4f}")

    print(f"\nBest F1: {best_f1:.4f}")
    return {"model": args.model, "best_f1": best_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="eeg_mamba", help=f"Model: {list_models()}")
    parser.add_argument("--data_path", type=str, default="../../data/processed")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
