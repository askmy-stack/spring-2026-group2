"""
Hugging Face model factory — creates any registered EEG model by name.

Usage:
    from src.component.models.hugging_face_mamba_moe.architectures.hf_factory import create_hf_model, list_hf_models
    model = create_hf_model("baseline_cnn_1d", in_channels=16, num_classes=2)
"""
from __future__ import annotations

import inspect
import logging
from typing import Type

from torch import nn

from .hf_cnn_models import (
    BaselineCNN1D,
    DeepConvNet,
    EEGNetHFPretrained,
    EEGNetLocal,
    EnhancedCNN1D,
    MultiScaleAttentionCNN,
    MultiScaleCNN,
    STEEGFormerPretrained,
)
from .pretrained import (
    BENDRPretrainedModel,
    BIOTPretrainedModel,
    EEGPTPretrainedModel,
    HFSTEEGFormerPretrainedModel,
)

logger = logging.getLogger(__name__)

HF_MODEL_REGISTRY: dict[str, Type[nn.Module]] = {
    "baseline_cnn_1d":          BaselineCNN1D,
    "enhanced_cnn_1d":          EnhancedCNN1D,
    "eegnet_local":             EEGNetLocal,
    "eegnet":                   EEGNetHFPretrained,
    "deepconvnet":              DeepConvNet,
    "multiscale_cnn":           MultiScaleCNN,
    "multiscale_attention_cnn": MultiScaleAttentionCNN,
    "st_eegformer":             STEEGFormerPretrained,
    "bendr_pretrained":         BENDRPretrainedModel,
    "biot_pretrained":          BIOTPretrainedModel,
    "eegpt_pretrained":         EEGPTPretrainedModel,
    "hf_st_eegformer":          HFSTEEGFormerPretrainedModel,
}


def create_hf_model(name: str, **kwargs) -> nn.Module:
    """
    Instantiate a registered HF EEG model by name, filtering unknown kwargs.

    Args:
        name: Registry key (see list_hf_models()).
        **kwargs: Constructor keyword arguments; unknown keys are silently dropped.

    Returns:
        Instantiated nn.Module, or None if incompatible (channel mismatch).

    Raises:
        ValueError: If name is not in HF_MODEL_REGISTRY.

    Example:
        >>> model = create_hf_model("baseline_cnn_1d", in_channels=16, num_classes=2)
        >>> out = model(torch.randn(4, 16, 256))
    """
    key = name.lower()
    if key not in HF_MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list_hf_models()}")
    cls = HF_MODEL_REGISTRY[key]
    valid = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    logger.debug("Creating %s with kwargs: %s", key, filtered)
    try:
        return cls(**filtered)
    except ValueError as e:
        if "channels" in str(e).lower():
            logger.warning("Skipping %s: %s", key, e)
            return None
        raise


def list_hf_models() -> list[str]:
    """Return sorted list of registered HF model names."""
    return sorted(HF_MODEL_REGISTRY)
