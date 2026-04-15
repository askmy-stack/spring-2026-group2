"""
Improved LSTM Models for EEG Seizure Detection
===============================================
Uses existing LSTM architectures with enhanced training:
- Data augmentation (time shift, noise, channel dropout)
- Better hyperparameters (hidden=256, layers=3)
- Improved training (warmup, cosine annealing, label smoothing)
- Ensemble prediction
"""

# Import existing architectures
from ..architectures import (
    VanillaLSTM,
    BiLSTM,
    AttentionBiLSTM,
    CNNLSTM,
    FeatureBiLSTM,
    MODEL_REGISTRY,
)

from .augmentation import EEGAugmentation
from .ensemble import EnsemblePredictor
from .train import train_improved, ImprovedTrainer

__all__ = [
    # Existing models (re-exported)
    "VanillaLSTM",
    "BiLSTM",
    "AttentionBiLSTM",
    "CNNLSTM",
    "FeatureBiLSTM",
    "MODEL_REGISTRY",
    # New components
    "EEGAugmentation",
    "EnsemblePredictor",
    "train_improved",
    "ImprovedTrainer",
]
