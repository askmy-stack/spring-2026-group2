"""
Ensemble Prediction for Improved LSTM Models.

Combines multiple model predictions via averaging, weighted averaging,
voting, max probability, or stacking.

Import from here — never define ensemble logic inline in a training script.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EnsemblePredictor(nn.Module):
    """
    Ensemble predictor combining multiple trained LSTM models.

    Args:
        models: List of trained nn.Module instances.
        weights: Optional per-model weights for 'weighted' strategy.
        strategy: One of 'average', 'weighted', 'voting', 'max', 'stacking'.
        threshold: Classification threshold for 'voting' strategy (default: 0.5).

    Example:
        >>> ens = EnsemblePredictor([model_a, model_b], strategy='average')
        >>> logits = ens(torch.randn(4, 16, 256))
        >>> assert logits.shape == (4, 1)
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        strategy: str = "weighted",
        threshold: float = 0.5,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.strategy = strategy
        self.threshold = threshold
        for model in self.models:
            model.eval()
        self.register_buffer("weights", _normalise_weights(weights, len(models)))
        if strategy == "stacking":
            self.meta_learner = _build_meta_learner(len(models))

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Compute ensemble prediction.

        Args:
            eeg_input: EEG batch, shape (batch, n_channels, time_steps).

        Returns:
            Ensemble logits, shape (batch, 1).
        """
        all_probs = self._collect_probs(eeg_input)
        return self._combine(all_probs)

    def predict_proba(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Return ensemble probabilities.

        Args:
            eeg_input: EEG batch, shape (batch, n_channels, time_steps).

        Returns:
            Probabilities in [0, 1], shape (batch,).
        """
        logits = self.forward(eeg_input)
        return torch.sigmoid(logits).squeeze(-1)

    def _collect_probs(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """Run all base models and stack their sigmoid probabilities."""
        all_probs = []
        for model in self.models:
            with torch.no_grad():
                logits = model(eeg_input)
            all_probs.append(torch.sigmoid(logits))
        return torch.stack(all_probs, dim=-1)

    def _combine(self, all_probs: torch.Tensor) -> torch.Tensor:
        """Combine per-model probabilities using the selected strategy."""
        if self.strategy == "average":
            ensemble_prob = all_probs.mean(dim=-1)
        elif self.strategy == "weighted":
            w = self.weights.to(all_probs.device).view(1, 1, -1)
            ensemble_prob = (all_probs * w).sum(dim=-1)
        elif self.strategy == "voting":
            ensemble_prob = (all_probs > self.threshold).float().mean(dim=-1)
        elif self.strategy == "max":
            ensemble_prob = all_probs.max(dim=-1).values
        elif self.strategy == "stacking":
            stacked = all_probs.squeeze(1)
            return self.meta_learner(stacked)
        else:
            raise ValueError(f"Unknown ensemble strategy: '{self.strategy}'")
        ensemble_prob = torch.clamp(ensemble_prob, 1e-7, 1.0 - 1e-7)
        return torch.log(ensemble_prob / (1.0 - ensemble_prob))


def load_ensemble_from_checkpoints(
    model_builders: List[callable],
    checkpoint_paths: List[str],
    device: torch.device,
    strategy: str = "weighted",
    weights: Optional[List[float]] = None,
) -> EnsemblePredictor:
    """
    Load checkpoint files and build an EnsemblePredictor.

    Args:
        model_builders: List of callables that return an nn.Module each.
        checkpoint_paths: Corresponding list of .pt checkpoint file paths.
        device: Device to map checkpoint tensors to.
        strategy: Ensemble combination strategy.
        weights: Optional per-model weights.

    Returns:
        EnsemblePredictor with all checkpoints loaded.

    Raises:
        FileNotFoundError: If any checkpoint path does not exist.
    """
    models = []
    for builder, ckpt_path in zip(model_builders, checkpoint_paths):
        model = builder()
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device)
        models.append(model)
    logger.info("Loaded %d model checkpoints for ensemble.", len(models))
    return EnsemblePredictor(models, weights=weights, strategy=strategy)


def _normalise_weights(weights: Optional[List[float]], n_models: int) -> torch.Tensor:
    """Return a normalised weight tensor (uniform if weights is None)."""
    if weights is None:
        return torch.ones(n_models) / n_models
    w = np.array(weights, dtype=np.float32)
    return torch.tensor(w / w.sum(), dtype=torch.float32)


def _build_meta_learner(n_models: int) -> nn.Sequential:
    """Build a small FC meta-learner for stacking ensemble."""
    return nn.Sequential(
        nn.Linear(n_models, 16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, 1),
    )
