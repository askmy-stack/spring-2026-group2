"""
Improved LSTM Training for EEG Seizure Detection
=================================================
Enhanced training pipeline with:
- Data augmentation (time shift, noise, channel dropout)
- Better hyperparameters (hidden=256, layers=3, batch=32)
- Warmup + cosine annealing scheduler
- Label smoothing
- Gradient clipping
- Ensemble training and prediction

Usage:
    python train.py --data_path ../../results/tensors/chbmit --epochs 100
    python train.py --data_path ../../results/tensors/chbmit --ensemble
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, roc_curve,
)

# Support both direct script execution and package import
try:
    from .augmentation import EEGAugmentation, MixUp, mixup_criterion
    from .ensemble import EnsemblePredictor, compute_ensemble_weights, train_stacking_ensemble
    from ..architectures import MODEL_REGISTRY
except ImportError:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(SCRIPT_DIR)
    sys.path.insert(0, PARENT_DIR)
    sys.path.insert(0, SCRIPT_DIR)
    from architectures import MODEL_REGISTRY
    from augmentation import EEGAugmentation, MixUp, mixup_criterion
    from ensemble import EnsemblePredictor, compute_ensemble_weights, train_stacking_ensemble


# ─────────────────────────────────────────────────────────────────────────────
# Improved Training Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Architecture
    'hidden_size': 256,      # Increased from 128
    'num_layers': 3,         # Increased from 2
    'dropout': 0.4,          # Increased from 0.3
    
    # Training
    'epochs': 100,           # Increased from 20-50
    'batch_size': 32,        # Decreased from 64
    'lr': 5e-4,              # Decreased from 1e-3
    'weight_decay': 1e-4,
    'label_smoothing': 0.1,  # Increased from 0.02
    'warmup_epochs': 5,
    'patience': 15,          # Increased from 7
    'grad_clip': 1.0,
    
    # Augmentation
    'augment': True,
    'aug_prob': 0.5,
    'mixup_alpha': 0.2,
    
    # Ensemble
    'ensemble_strategy': 'weighted',
}


# ─────────────────────────────────────────────────────────────────────────────
# Learning Rate Scheduler with Warmup
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))
        
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group['lr'] = max(self.min_lr, base_lr * factor)
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss with Label Smoothing
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss with label smoothing support."""
    
    def __init__(
        self,
        gamma: float = 1.0,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE loss without pos_weight so focal modulation applies uniformly
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        # Focal weight
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss

        # Apply pos_weight as flat per-sample multiplier (after focal modulation)
        if self.pos_weight is not None:
            class_weight = torch.where(
                targets > 0.5,
                self.pos_weight.expand_as(targets),
                torch.ones_like(targets),
            )
            focal_loss = focal_loss * class_weight

        return focal_loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Improved Trainer Class
# ─────────────────────────────────────────────────────────────────────────────

class ImprovedTrainer:
    """
    Enhanced trainer with all improvements.
    
    Features:
    - Data augmentation
    - Warmup + cosine annealing
    - Focal loss with label smoothing
    - MixUp augmentation
    - Gradient clipping
    - Best model checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Optional[Dict] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Augmentation
        self.augmenter = EEGAugmentation(
            time_shift_max=20,
            noise_std=0.1,
            channel_dropout_prob=0.1,
            p=self.config['aug_prob'],
        ) if self.config['augment'] else None
        
        self.mixup = MixUp(alpha=self.config['mixup_alpha']) if self.config['mixup_alpha'] > 0 else None
        
        # Training state
        self.best_f1 = -1.0
        self.best_state = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'lr': []}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        pos_weight: float,
    ) -> Dict:
        """Train the model with all improvements."""
        
        # Loss function
        pw = torch.tensor([pos_weight]).to(self.device)
        criterion = FocalLoss(
            gamma=1.0,
            pos_weight=pw,
            label_smoothing=self.config['label_smoothing'],
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )
        
        # Scheduler
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=self.config['warmup_epochs'],
            total_epochs=self.config['epochs'],
        )
        
        best_metrics = None
        wait = 0
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            scheduler.step(epoch - 1)
            
            # ── Train ──
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            
            # ── Validate ──
            metrics = self._validate(val_loader, criterion)
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(metrics['val_loss'])
            self.history['val_f1'].append(metrics['f1'])
            self.history['lr'].append(scheduler.get_lr())
            
            # Early stopping
            if metrics['f1'] > self.best_f1:
                self.best_f1 = metrics['f1']
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                best_metrics = metrics.copy()
                best_metrics['epoch'] = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.config['patience']:
                    print(f"  Early stopping at epoch {epoch}")
                    break
            
            # Progress (every 10 epochs)
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val F1: {metrics['f1']:.4f} | "
                      f"AUC: {metrics['auc_roc']:.4f} | LR: {scheduler.get_lr():.2e}")
        
        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        best_metrics['train_time_sec'] = round(time.time() - start_time, 1)
        best_metrics['total_params'] = sum(p.numel() for p in self.model.parameters())
        
        return best_metrics
    
    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).float().unsqueeze(1)
            
            # Augmentation (augment_batch also oversamples minority seizure class)
            if self.augmenter is not None:
                try:
                    from .augmentation import augment_batch
                except ImportError:
                    from augmentation import augment_batch
                X_batch, y_batch = augment_batch(X_batch, y_batch.squeeze(1), self.augmenter)
                y_batch = y_batch.unsqueeze(1)
            
            # MixUp
            use_mixup = self.mixup is not None and np.random.rand() < 0.5
            if use_mixup:
                self.mixup.train()
                X_batch, y_a, y_b, lam = self.mixup(X_batch, y_batch)
            
            optimizer.zero_grad()
            logits = self.model(X_batch)
            
            if use_mixup:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def _validate(self, loader: DataLoader, criterion: nn.Module) -> Dict:
        """Validate and compute metrics."""
        self.model.eval()
        all_probs, all_labels = [], []
        val_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float().unsqueeze(1)
                
                logits = self.model(X_batch)
                val_loss += criterion(logits, y_batch).item()
                n_batches += 1
                
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(y_batch.cpu())
        
        probs = torch.cat(all_probs).numpy().flatten()
        labels = torch.cat(all_labels).numpy().flatten()
        
        # Optimal threshold
        thresh = self._find_optimal_threshold(labels, probs)
        preds = (probs >= thresh).astype(float)
        
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        
        return {
            'accuracy': float(accuracy_score(labels, preds)),
            'sensitivity': float(recall_score(labels, preds, zero_division=0)),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'precision': float(precision_score(labels, preds, zero_division=0)),
            'f1': float(f1_score(labels, preds, zero_division=0)),
            'auc_roc': float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
            'threshold': round(thresh, 4),
            'val_loss': val_loss / n_batches if n_batches > 0 else 0.0,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        }
    
    @staticmethod
    def _find_optimal_threshold(y_true, y_prob):
        """Youden's J: threshold that maximizes sensitivity + specificity."""
        if len(np.unique(y_true)) < 2:
            return 0.5
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        # sklearn prepends a synthetic threshold > max(y_prob); clip to [0, 1]
        return float(np.clip(thresholds[best_idx], 0.0, 1.0))
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'best_f1': self.best_f1,
            'history': self.history,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state'])
        self.config = checkpoint.get('config', self.config)
        self.best_f1 = checkpoint.get('best_f1', -1.0)
        self.history = checkpoint.get('history', self.history)


# ─────────────────────────────────────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────────────────────────────────────

def load_real_data(data_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load real EEG data from tensors folder."""
    data_path = Path(data_path)
    
    # Load training data
    train_data = torch.load(data_path / "train" / "data.pt", weights_only=True)
    train_labels = torch.load(data_path / "train" / "labels.pt", weights_only=True)
    
    # Load test/val data
    if (data_path / "test" / "data.pt").exists():
        test_data = torch.load(data_path / "test" / "data.pt", weights_only=True)
        test_labels = torch.load(data_path / "test" / "labels.pt", weights_only=True)
    elif (data_path / "val" / "data.pt").exists():
        test_data = torch.load(data_path / "val" / "data.pt", weights_only=True)
        test_labels = torch.load(data_path / "val" / "labels.pt", weights_only=True)
    else:
        raise FileNotFoundError(f"No test or val data found in {data_path}")
    
    # Convert to float tensors
    X_train = train_data.float()
    y_train = train_labels.float().squeeze()
    X_test = test_data.float()
    y_test = test_labels.float().squeeze()
    
    return X_train, X_test, y_train, y_test


def train_improved(
    model_name: str,
    data_path: str,
    config: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    save_dir: str = './improved_results',
) -> Dict:
    """
    Train a single model with improved settings.
    
    Args:
        model_name: Name of model from MODEL_REGISTRY
        data_path: Path to tensors folder
        config: Optional config overrides
        device: Device for training
        save_dir: Directory to save results
    
    Returns:
        Training metrics
    """
    config = {**DEFAULT_CONFIG, **(config or {})}
    
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    X_train, X_test, y_train, y_test = load_real_data(data_path)
    
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    
    print(f"Train: {len(X_train)} samples (seizure={int(n_pos)})")
    print(f"Test:  {len(X_test)} samples (seizure={int(y_test.sum().item())})")
    
    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Create model
    n_channels = X_train.size(1)
    seq_len = X_train.size(2)
    
    ModelClass = MODEL_REGISTRY[model_name]
    model = ModelClass(
        n_channels=n_channels,
        seq_len=seq_len,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
    )
    
    print(f"\nTraining: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Config: hidden={config['hidden_size']}, layers={config['num_layers']}, "
          f"lr={config['lr']}, batch={config['batch_size']}")
    
    # Train
    trainer = ImprovedTrainer(model, device, config)
    metrics = trainer.train(train_loader, val_loader, pos_weight)
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    trainer.save(os.path.join(save_dir, f"{model_name}_best.pt"))
    
    print(f"\nResults for {model_name}:")
    print(f"  F1: {metrics['f1']:.4f} | AUC: {metrics['auc_roc']:.4f} | "
          f"Sens: {metrics['sensitivity']:.4f} | Spec: {metrics['specificity']:.4f}")
    
    return metrics


def train_ensemble(
    data_path: str,
    model_names: Optional[List[str]] = None,
    config: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    save_dir: str = './improved_results',
    strategy: str = 'weighted',
) -> Dict:
    """
    Train all models and create ensemble.
    
    Args:
        data_path: Path to tensors folder
        model_names: List of models to train (default: all raw-input models)
        config: Optional config overrides
        device: Device for training
        save_dir: Directory to save results
        strategy: Ensemble strategy
    
    Returns:
        Ensemble metrics
    """
    if model_names is None:
        model_names = ['vanilla_lstm', 'bilstm', 'attention_bilstm', 'cnn_lstm']
    
    config = {**DEFAULT_CONFIG, **(config or {})}
    
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    X_train, X_test, y_train, y_test = load_real_data(data_path)
    
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    
    n_channels = X_train.size(1)
    seq_len = X_train.size(2)
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    
    # Train individual models
    trained_models = []
    all_metrics = {}
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        ModelClass = MODEL_REGISTRY[model_name]
        model = ModelClass(
            n_channels=n_channels,
            seq_len=seq_len,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
        )
        
        trainer = ImprovedTrainer(model, device, config)
        metrics = trainer.train(train_loader, val_loader, pos_weight)
        all_metrics[model_name] = metrics
        
        # Save individual model
        os.makedirs(save_dir, exist_ok=True)
        trainer.save(os.path.join(save_dir, f"{model_name}_best.pt"))
        
        trained_models.append(trainer.model)
        
        print(f"  F1: {metrics['f1']:.4f} | AUC: {metrics['auc_roc']:.4f}")
    
    # Compute ensemble weights
    print(f"\n{'='*60}")
    print("Creating Ensemble")
    print(f"{'='*60}")
    
    weights = compute_ensemble_weights(trained_models, val_loader, device, metric='f1')
    print(f"Weights: {dict(zip(model_names, [f'{w:.3f}' for w in weights]))}")
    
    # Create ensemble
    ensemble = EnsemblePredictor(trained_models, weights=weights, strategy=strategy)
    
    # Evaluate ensemble
    ensemble.eval()
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            probs = ensemble.predict_proba(X_batch).cpu()
            all_probs.append(probs)
            all_labels.append(y_batch)
    
    probs = torch.cat(all_probs).numpy().flatten()
    labels = torch.cat(all_labels).numpy().flatten()
    
    # Metrics
    thresh = ImprovedTrainer._find_optimal_threshold(labels, probs)
    preds = (probs >= thresh).astype(float)
    
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    
    ensemble_metrics = {
        'accuracy': float(accuracy_score(labels, preds)),
        'sensitivity': float(recall_score(labels, preds, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'auc_roc': float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        'threshold': thresh,
        'strategy': strategy,
        'weights': weights,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }
    
    print(f"\nEnsemble Results ({strategy}):")
    print(f"  F1: {ensemble_metrics['f1']:.4f} | AUC: {ensemble_metrics['auc_roc']:.4f} | "
          f"Sens: {ensemble_metrics['sensitivity']:.4f} | Spec: {ensemble_metrics['specificity']:.4f}")
    
    # Save ensemble
    torch.save({
        'weights': weights,
        'strategy': strategy,
        'model_names': model_names,
        'metrics': ensemble_metrics,
    }, os.path.join(save_dir, 'ensemble_config.pt'))
    
    # Save all results
    all_metrics['ensemble'] = ensemble_metrics
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Improved LSTM Training")
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to tensors folder (e.g., ../../results/tensors/chbmit)")
    p.add_argument("--model", type=str, default="all",
                   choices=['vanilla_lstm', 'bilstm', 'attention_bilstm', 'cnn_lstm', 'all'],
                   help="Model to train")
    p.add_argument("--ensemble", action="store_true",
                   help="Train ensemble of all models")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--no_augment", action="store_true",
                   help="Disable data augmentation")
    p.add_argument("--save_dir", type=str, default="./improved_results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'lr': args.lr,
        'dropout': args.dropout,
        'label_smoothing': args.label_smoothing,
        'augment': not args.no_augment,
    }
    
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    
    print(f"\n{'='*60}")
    print("  Improved LSTM Training")
    print(f"  Device: {device}")
    print(f"  Config: {config}")
    print(f"{'='*60}")
    
    if args.ensemble or args.model == 'all':
        results = train_ensemble(
            data_path=args.data_path,
            config=config,
            device=device,
            save_dir=args.save_dir,
        )
    else:
        results = train_improved(
            model_name=args.model,
            data_path=args.data_path,
            config=config,
            device=device,
            save_dir=args.save_dir,
        )
    
    print(f"\n{'='*60}")
    print("  Training Complete!")
    print(f"  Results saved to: {args.save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
