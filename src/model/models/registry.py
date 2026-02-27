from __future__ import annotations

from typing import Any, Dict, Callable

from eeg_pipeline.models.eegnet import EEGNet

_MODEL_BUILDERS: Dict[str, Callable[..., object]] = {
    "eegnet": EEGNet,
    # add later:
    # "tcn": TCN,
    # "resnet1d": ResNet1D,
}


def get_model(name: str, **kwargs: Any):
    """
    Create a model by name, e.g. get_model("eegnet", n_ch=16, n_samples=1024, ...)
    """
    key = name.strip().lower()
    if key not in _MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_MODEL_BUILDERS.keys())}")
    return _MODEL_BUILDERS[key](**kwargs)