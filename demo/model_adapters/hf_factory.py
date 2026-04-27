from __future__ import annotations

import inspect

from torch import nn

from .st_eegformer import HFSTEEGFormerPretrainedModel


HF_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "st_eegformer": HFSTEEGFormerPretrainedModel,
    "hf_st_eegformer": HFSTEEGFormerPretrainedModel,
}


def create_hf_model(name: str, **kwargs) -> nn.Module:
    key = name.lower()
    if key not in HF_MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(HF_MODEL_REGISTRY)}")
    cls = HF_MODEL_REGISTRY[key]
    valid = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in valid}
    return cls(**filtered)
