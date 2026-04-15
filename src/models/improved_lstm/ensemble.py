"""
Ensemble Prediction for EEG Seizure Detection
==============================================
Combines predictions from multiple LSTM models for improved accuracy.

Methods:
- Average: Simple probability averaging
- Weighted: Learned or validation-based weights
- Stacking: Meta-learner on base model outputs
- Voting: Majority voting with threshold
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class EnsemblePredictor(nn.Module):
    """
    Ensemble predictor that combines multiple LSTM models.
    
    Supports multiple ensemble strategies:
    - 'average': Simple probability averaging
    - 'weighted': Weighted average based on validation F1
    - 'voting': Majority voting
    - 'max': Maximum probability (most confident)
    - 'stacking': Meta-learner on base predictions
    
    Args:
        models: List of trained models
        weights: Optional weights for weighted averaging
        strategy: Ensemble strategy ('average', 'weighted', 'voting', 'max', 'stacking')
        threshold: Classification threshold (default: 0.5)
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        strategy: str = 'weighted',
        threshold: float = 0.5,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.strategy = strategy
        self.threshold = threshold

        # Set base models to inference mode once at construction (not inside forward)
        for m in self.models:
            m.eval()
        
        # Normalize weights
        if weights is not None:
            weights = np.array(weights)
            weights = weights / weights.sum()
            self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        else:
            self.register_buffer('weights', torch.ones(len(models)) / len(models))
        
        # Stacking meta-learner
        if strategy == 'stacking':
            self.meta_learner = nn.Sequential(
                nn.Linear(len(models), 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get ensemble predictions.
        
        Args:
            x: Input tensor (batch, channels, time)
        
        Returns:
            Ensemble logits (batch, 1)
        """
        # Get predictions from all models (base models must already be in eval mode)
        all_logits = []
        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                all_logits.append(logits)
        
        all_logits = torch.stack(all_logits, dim=-1)  # (batch, 1, n_models)
        all_probs = torch.sigmoid(all_logits)
        
        if self.strategy == 'average':
            ensemble_prob = all_probs.mean(dim=-1)
        
        elif self.strategy == 'weighted':
            weights = self.weights.view(1, 1, -1)
            ensemble_prob = (all_probs * weights).sum(dim=-1)
        
        elif self.strategy == 'voting':
            votes = (all_probs > self.threshold).float()
            ensemble_prob = votes.mean(dim=-1)
        
        elif self.strategy == 'max':
            ensemble_prob = all_probs.max(dim=-1).values
        
        elif self.strategy == 'stacking':
            # Use meta-learner
            stacked = all_probs.squeeze(1)  # (batch, n_models)
            ensemble_logits = self.meta_learner(stacked)
            return ensemble_logits
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Convert probability back to logits for BCEWithLogitsLoss compatibility
        ensemble_prob = torch.clamp(ensemble_prob, 1e-7, 1 - 1e-7)
        ensemble_logits = torch.log(ensemble_prob / (1 - ensemble_prob))
        
        return ensemble_logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get ensemble probabilities."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get ensemble binary predictions."""
        probs = self.predict_proba(x)
        return (probs >= self.threshold).long()
    
    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: List[str],
        model_classes: List[type],
        model_kwargs: List[Dict],
        weights: Optional[List[float]] = None,
        strategy: str = 'weighted',
        device: str = 'cpu',
    ) -> 'EnsemblePredictor':
        """
        Load ensemble from saved checkpoints.
        
        Args:
            checkpoint_paths: Paths to model checkpoints
            model_classes: Model classes for each checkpoint
            model_kwargs: Kwargs for each model class
            weights: Optional ensemble weights
            strategy: Ensemble strategy
            device: Device to load models to
        
        Returns:
            EnsemblePredictor instance
        """
        models = []
        for path, model_cls, kwargs in zip(checkpoint_paths, model_classes, model_kwargs):
            model = model_cls(**kwargs)
            state_dict = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models.append(model)
        
        return cls(models, weights=weights, strategy=strategy)


def compute_ensemble_weights(
    models: List[nn.Module],
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    metric: str = 'f1',
) -> List[float]:
    """
    Compute ensemble weights based on validation performance.
    
    Args:
        models: List of trained models
        val_loader: Validation data loader
        device: Device for computation
        metric: Metric to use for weighting ('f1', 'auc', 'accuracy')
    
    Returns:
        List of weights (one per model)
    """
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
    
    weights = []
    
    for model in models:
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                preds = (probs >= 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy().flatten())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        if metric == 'f1':
            score = f1_score(all_labels, all_preds, zero_division=0)
        elif metric == 'auc':
            try:
                score = roc_auc_score(all_labels, all_probs)
            except ValueError:
                score = 0.5
        elif metric == 'accuracy':
            score = accuracy_score(all_labels, all_preds)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        weights.append(score)
    
    # Normalize weights
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(models)) / len(models)
    
    return weights.tolist()


def train_stacking_ensemble(
    ensemble: EnsemblePredictor,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
) -> EnsemblePredictor:
    """
    Train the meta-learner for stacking ensemble.
    
    Args:
        ensemble: EnsemblePredictor with strategy='stacking'
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device for training
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained ensemble
    """
    assert ensemble.strategy == 'stacking', "Ensemble must use stacking strategy"
    
    # Freeze base models
    for model in ensemble.models:
        for param in model.parameters():
            param.requires_grad = False
    
    ensemble = ensemble.to(device)
    optimizer = torch.optim.Adam(ensemble.meta_learner.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        ensemble.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Get base model predictions
            all_probs = []
            for model in ensemble.models:
                model.eval()
                with torch.no_grad():
                    logits = model(X_batch)
                    probs = torch.sigmoid(logits)
                    all_probs.append(probs)
            
            stacked = torch.cat(all_probs, dim=1)  # (batch, n_models)
            
            # Train meta-learner
            ensemble.meta_learner.train()
            meta_logits = ensemble.meta_learner(stacked)
            loss = criterion(meta_logits, y_batch)
            
            loss.backward()
            optimizer.step()
        
        # Validation
        ensemble.eval()
        val_loss = 0
        n_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).float().unsqueeze(1)
                
                logits = ensemble(X_batch)
                val_loss += criterion(logits, y_batch).item()
                n_batches += 1
        
        if n_batches > 0:
            val_loss /= n_batches
        if val_loss < best_loss:
            best_loss = val_loss
    
    return ensemble
