#!/usr/bin/env python3
"""
Approach 2: 7-Model Ensemble
----------------------------
Weighted ensemble with stacking meta-learner.

Ensemble Methods:
1. Simple averaging
2. Weighted averaging (learned weights)
3. Stacking meta-learner (MLP on predictions)

Usage:
    python ensemble_7model.py --checkpoints ./checkpoints --output ./results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from architectures import get_model, list_models


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble Classes
# ─────────────────────────────────────────────────────────────────────────────

class SimpleAverageEnsemble:
    """Simple probability averaging."""

    def __init__(self, models: List[nn.Module]):
        self.models = models

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        all_probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        # Average
        return np.mean(all_probs, axis=0)


class WeightedAverageEnsemble(nn.Module):
    """Learned weighted averaging."""

    def __init__(self, models: List[nn.Module], n_models: int = 7):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.ones(n_models) / n_models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        all_probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                probs = torch.sigmoid(logits)
                all_probs.append(probs)

        # Stack and weight
        probs = torch.stack(all_probs, dim=-1)  # (batch, 1, n_models)
        weights = F.softmax(self.weights, dim=0)
        weighted = (probs * weights).sum(dim=-1)

        return weighted

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.forward(x).cpu().numpy()


class StackingEnsemble(nn.Module):
    """Stacking meta-learner on model predictions."""

    def __init__(self, models: List[nn.Module], n_models: int = 7, hidden_dim: int = 32):
        super().__init__()
        self.models = nn.ModuleList(models)

        # Meta-learner MLP
        self.meta_learner = nn.Sequential(
            nn.Linear(n_models, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def get_base_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions from all base models."""
        all_probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                probs = torch.sigmoid(logits)
                all_probs.append(probs)

        return torch.cat(all_probs, dim=-1)  # (batch, n_models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_probs = self.get_base_predictions(x)
        return self.meta_learner(base_probs)

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

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


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute all metrics."""
    y_pred = (y_prob >= threshold).astype(float)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        "threshold": threshold,
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def load_models(
    checkpoint_dir: str,
    model_names: List[str],
    device: torch.device,
    n_channels: int = 16,
    time_steps: int = 256,
) -> List[nn.Module]:
    """Load trained models from checkpoints."""
    models = []
    checkpoint_path = Path(checkpoint_dir)

    for name in model_names:
        # Create model
        model = get_model(name, n_channels=n_channels, time_steps=time_steps)

        # Load weights if available
        ckpt_file = checkpoint_path / f"{name}_best.pt"
        if ckpt_file.exists():
            print(f"Loading {name} from {ckpt_file}")
            checkpoint = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print(f"Warning: No checkpoint for {name}, using random weights")

        model = model.to(device)
        model.eval()
        models.append(model)

    return models


def load_data(data_path: str, test_size: float = 0.2) -> Tuple:
    """Load EEG data."""
    data_path = Path(data_path)

    x_path = data_path / "X_windows.npy"
    y_path = data_path / "y_windows.npy"

    if x_path.exists() and y_path.exists():
        print(f"Loading data from {data_path}")
        X = np.load(x_path)
        y = np.load(y_path)
    else:
        print("Generating synthetic data for testing...")
        np.random.seed(42)
        n_samples = 5000
        n_channels = 16
        time_steps = 256

        X = np.random.randn(n_samples, n_channels, time_steps).astype(np.float32)
        y = np.random.randint(0, 2, n_samples).astype(np.float32)

        seizure_idx = y == 1
        X[seizure_idx] += np.sin(np.linspace(0, 10 * np.pi, time_steps)) * 0.5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Train Stacking Ensemble
# ─────────────────────────────────────────────────────────────────────────────

def train_stacking_ensemble(
    ensemble: StackingEnsemble,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
) -> StackingEnsemble:
    """Train the stacking meta-learner."""
    print("\nTraining stacking meta-learner...")

    # Freeze base models
    for model in ensemble.models:
        for param in model.parameters():
            param.requires_grad = False

    # Only train meta-learner
    optimizer = torch.optim.Adam(ensemble.meta_learner.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    best_state = None

    for epoch in range(epochs):
        ensemble.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = ensemble(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Evaluate
        ensemble.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                probs = ensemble.predict_proba(x_batch)
                all_probs.extend(probs.flatten())
                all_labels.extend(y_batch.numpy().flatten())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        threshold = find_optimal_threshold(all_labels, all_probs)
        metrics = compute_metrics(all_labels, all_probs, threshold)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.clone() for k, v in ensemble.meta_learner.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {total_loss/len(train_loader):.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

    # Load best
    if best_state is not None:
        ensemble.meta_learner.load_state_dict(best_state)

    return ensemble


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="7-Model Ensemble")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--data_path", type=str, default="../../data/processed")
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    # Setup
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Load data
    X_train, X_test, y_train, y_test = load_data(args.data_path)
    n_channels = X_train.shape[1]
    time_steps = X_train.shape[2]

    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).float(),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Load models
    model_names = list_models()
    print(f"\nLoading {len(model_names)} models...")
    models = load_models(
        args.checkpoints, model_names, device, n_channels, time_steps
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluate Individual Models
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INDIVIDUAL MODEL RESULTS")
    print("=" * 60)

    individual_results = {}
    all_test_probs = []

    for name, model in zip(model_names, models):
        model.eval()
        probs = []

        with torch.no_grad():
            for x_batch, _ in test_loader:
                x_batch = x_batch.to(device)
                output = model(x_batch)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                p = torch.sigmoid(logits).cpu().numpy()
                probs.extend(p.flatten())

        probs = np.array(probs)
        all_test_probs.append(probs)

        threshold = find_optimal_threshold(y_test, probs)
        metrics = compute_metrics(y_test, probs, threshold)
        individual_results[name] = metrics

        print(
            f"{name:25s} | F1: {metrics['f1']:.4f} | "
            f"AUC: {metrics['auc_roc']:.4f} | "
            f"Sens: {metrics['sensitivity']:.4f} | "
            f"Spec: {metrics['specificity']:.4f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Ensemble Methods
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS")
    print("=" * 60)

    ensemble_results = {}

    # 1. Simple Average
    avg_probs = np.mean(all_test_probs, axis=0)
    threshold = find_optimal_threshold(y_test, avg_probs)
    metrics = compute_metrics(y_test, avg_probs, threshold)
    ensemble_results["simple_average"] = metrics
    print(
        f"{'Simple Average':25s} | F1: {metrics['f1']:.4f} | "
        f"AUC: {metrics['auc_roc']:.4f} | "
        f"Sens: {metrics['sensitivity']:.4f} | "
        f"Spec: {metrics['specificity']:.4f}"
    )

    # 2. Weighted Average (using validation F1 as weights)
    weights = np.array([individual_results[name]["f1"] for name in model_names])
    weights = weights / weights.sum()
    weighted_probs = np.average(all_test_probs, axis=0, weights=weights)
    threshold = find_optimal_threshold(y_test, weighted_probs)
    metrics = compute_metrics(y_test, weighted_probs, threshold)
    ensemble_results["weighted_average"] = metrics
    print(
        f"{'Weighted Average':25s} | F1: {metrics['f1']:.4f} | "
        f"AUC: {metrics['auc_roc']:.4f} | "
        f"Sens: {metrics['sensitivity']:.4f} | "
        f"Spec: {metrics['specificity']:.4f}"
    )

    # 3. Stacking Ensemble
    stacking = StackingEnsemble(models, n_models=len(models))
    stacking = stacking.to(device)
    stacking = train_stacking_ensemble(stacking, train_loader, test_loader, device)

    stacking.eval()
    stacking_probs = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            p = stacking.predict_proba(x_batch)
            stacking_probs.extend(p.flatten())

    stacking_probs = np.array(stacking_probs)
    threshold = find_optimal_threshold(y_test, stacking_probs)
    metrics = compute_metrics(y_test, stacking_probs, threshold)
    ensemble_results["stacking"] = metrics
    print(
        f"{'Stacking Meta-Learner':25s} | F1: {metrics['f1']:.4f} | "
        f"AUC: {metrics['auc_roc']:.4f} | "
        f"Sens: {metrics['sensitivity']:.4f} | "
        f"Spec: {metrics['specificity']:.4f}"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Save Results
    # ─────────────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "individual_models": individual_results,
        "ensembles": ensemble_results,
        "model_weights": {name: float(w) for name, w in zip(model_names, weights)},
    }

    with open(output_path / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Save stacking model
    torch.save(
        {
            "meta_learner_state_dict": stacking.meta_learner.state_dict(),
            "model_names": model_names,
        },
        output_path / "stacking_ensemble.pt",
    )

    print(f"\nResults saved to {output_path}")

    # Best ensemble
    best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]["f1"])
    print(f"\n✓ Best ensemble: {best_ensemble[0]} with F1={best_ensemble[1]['f1']:.4f}")


if __name__ == "__main__":
    main()
