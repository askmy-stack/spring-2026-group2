from __future__ import annotations

from typing import Dict, Type

from .cnn_benchmark import CNNBenchmark1D
from .cnn_improved import CNNImproved1D
from .cnn_mixture import CNNMixture1D
from .cnn_multiscale import CNNMultiScale1D
from .eegnet import EEGNet
from .eegnet_improved import EEGNetImproved

MODEL_REGISTRY: Dict[str, Type] = {
    "cnn_benchmark": CNNBenchmark1D,
    "cnn_improved": CNNImproved1D,
    "cnn_mixture": CNNMixture1D,
    "cnn_multiscale": CNNMultiScale1D,
    "eegnet": EEGNet,
    "eegnet_improved": EEGNetImproved,
}


def get_model(name: str, **kwargs):
    name = name.lower().strip()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
