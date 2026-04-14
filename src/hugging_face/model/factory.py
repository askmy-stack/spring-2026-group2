from __future__ import annotations

import inspect
from typing import Callable

from torch import nn

from .architectures import (
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


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "baseline_cnn_1d": BaselineCNN1D,
    "bendr_pretrained": BENDRPretrainedModel,
    "biot_pretrained": BIOTPretrainedModel,
    "eegpt_pretrained": EEGPTPretrainedModel,
    "enhanced_cnn_1d": EnhancedCNN1D,
    "eegnet": EEGNetHFPretrained,
    "eegnet_local": EEGNetLocal,
    "deepconvnet": DeepConvNet,
    "multiscale_cnn": MultiScaleCNN,
    "multiscale_attention_cnn": MultiScaleAttentionCNN,
    "st_eegformer": HFSTEEGFormerPretrainedModel,
    "hf_st_eegformer": HFSTEEGFormerPretrainedModel,
}


def create_model(name: str, **kwargs) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list_models()}")
    model_cls = MODEL_REGISTRY[key]
    valid_kwargs = inspect.signature(model_cls.__init__).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_kwargs}
    return model_cls(**filtered_kwargs)


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY)
