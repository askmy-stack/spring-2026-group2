"""
Approach 2: Advanced Ensemble with Pre-trained Models
======================================================

This package provides 7 diverse model architectures for EEG seizure detection
with support for pre-trained transformer encoders and ensemble methods.

Models:
    M1: VanillaLSTM + ChannelAttn
    M2: BiLSTM + ChannelAttn
    M3: AttentionBiLSTM + CrissCross (CBraMod-inspired)
    M4: CNN-LSTM (multi-scale branches)
    M5: FeatureBiLSTM + TemporalAttn
    M6: GraphBiLSTM (GMAEEG-inspired)
    M7: VQ-Transformer (EEGFormer-inspired)

Pre-trained Encoders:
    - CBraMod (ICLR 2025)
    - EEGPT (NeurIPS 2024)
    - BIOT
    - LaBraM (ICLR 2024)

Usage:
    # Train single model
    python train.py --model m3_criss_cross --epochs 50

    # Train with pretrained encoder
    python train.py --model m4_cnn_lstm --pretrained --encoder cbramod

    # Train all models
    python train.py --model all --epochs 50

    # Run 7-model ensemble
    python ensemble_7model.py --checkpoints ./checkpoints
"""

from .architectures import (
    get_model,
    list_models,
    MODEL_REGISTRY,
    M1_VanillaLSTM,
    M2_BiLSTM,
    M3_CrissCrossBiLSTM,
    M4_CNNLSTM,
    M5_FeatureBiLSTM,
    M6_GraphBiLSTM,
    M7_VQTransformer,
)

from .modules import (
    ChannelAttention,
    CrissCrossAttention,
    GraphAttention,
    VectorQuantizer,
    load_pretrained_encoder,
)

__version__ = "2.0.0"

__all__ = [
    # Model factory
    "get_model",
    "list_models",
    "MODEL_REGISTRY",
    # Models
    "M1_VanillaLSTM",
    "M2_BiLSTM",
    "M3_CrissCrossBiLSTM",
    "M4_CNNLSTM",
    "M5_FeatureBiLSTM",
    "M6_GraphBiLSTM",
    "M7_VQTransformer",
    # Modules
    "ChannelAttention",
    "CrissCrossAttention",
    "GraphAttention",
    "VectorQuantizer",
    "load_pretrained_encoder",
]
