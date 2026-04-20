"""
Approach 3 Techniques for EEG Seizure Detection
==============================================================

This package provides advanced/experimental techniques:

1. **Mamba + MoE**: O(n) state-space models with expert routing
2. **Diffusion Augmentation**: Generate synthetic seizure EEG
3. **Pre-ictal Prediction**: 30-60 minute seizure forecasting
4. **Multi-Teacher Distillation**: Compress ensemble to tiny model
5. **Uncertainty Quantification**: MC Dropout for clinical safety

Quick Start:
    # Train Mamba model
    python train_mamba.py --model eeg_mamba --epochs 50

    # Generate synthetic seizures
    python pretrain_diffusion.py --epochs 100 --generate 1000

    # Pre-ictal prediction
    python predict_preictal.py --horizon 30 --epochs 50

    # Distill to tiny model
    python distill.py --teachers ../approach2_advanced/checkpoints --student tiny

    # Inference with uncertainty
    python inference_uncertain.py --model eeg_mamba --mc_samples 30
"""

from .architectures import (
    get_model,
    list_models,
    EEGMamba,
    EEGMambaMoE,
    HierarchicalLSTM,
    PreIctalPredictor,
    TinySeizureNet,
    MicroSeizureNet,
)

from .modules import (
    MambaBlock,
    BidirectionalMamba,
    MixtureOfExperts,
    EEGDiffusion,
    MCDropout,
    mc_dropout_inference,
    EvidentialClassifier,
)

__version__ = "3.0.0"

__all__ = [
    # Model factory
    "get_model",
    "list_models",
    # Architectures
    "EEGMamba",
    "EEGMambaMoE",
    "HierarchicalLSTM",
    "PreIctalPredictor",
    "TinySeizureNet",
    "MicroSeizureNet",
    # Modules
    "MambaBlock",
    "BidirectionalMamba",
    "MixtureOfExperts",
    "EEGDiffusion",
    "MCDropout",
    "mc_dropout_inference",
    "EvidentialClassifier",
]
