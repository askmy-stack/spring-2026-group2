from .cnn_benchmark import CNNBenchmark1D
from .cnn_improved import CNNImproved1D
from .cnn_mixture import CNNMixture1D
from .cnn_multiscale import CNNMultiScale1D
from .eegnet import EEGNet
from .eegnet_improved import EEGNetImproved
from .registry import get_model, MODEL_REGISTRY

__all__ = [
    "CNNBenchmark1D",
    "CNNImproved1D",
    "CNNMixture1D",
    "CNNMultiScale1D",
    "EEGNet",
    "EEGNetImproved",
    "get_model",
    "MODEL_REGISTRY",
]
