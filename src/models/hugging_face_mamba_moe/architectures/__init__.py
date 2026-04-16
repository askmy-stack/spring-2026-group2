"""Mamba + MoE + Hugging Face CNN architectures."""
from .eeg_mamba import EEGMamba, EEGMambaMoE
from .hf_cnn_models import (
    BaselineCNN1D, EnhancedCNN1D, EEGNetLocal, EEGNetHFPretrained,
    DeepConvNet, MultiScaleCNN, MultiScaleAttentionCNN, STEEGFormerPretrained,
)
from .hf_factory import create_hf_model, list_hf_models, HF_MODEL_REGISTRY
from .pretrained import (
    BENDRPretrainedModel, BIOTPretrainedModel,
    EEGPTPretrainedModel, HFSTEEGFormerPretrainedModel,
)

__all__ = [
    "EEGMamba", "EEGMambaMoE",
    "BaselineCNN1D", "EnhancedCNN1D", "EEGNetLocal", "EEGNetHFPretrained",
    "DeepConvNet", "MultiScaleCNN", "MultiScaleAttentionCNN", "STEEGFormerPretrained",
    "BENDRPretrainedModel", "BIOTPretrainedModel", "EEGPTPretrainedModel", "HFSTEEGFormerPretrainedModel",
    "create_hf_model", "list_hf_models", "HF_MODEL_REGISTRY",
]
