"""
LSTM Benchmark Model Architectures — m1 through m6.

Input contract for all models: (batch, n_channels, time_steps) -> logits (batch, 1).
"""
import inspect
from typing import Dict, Type

import torch.nn as nn

from .m1_vanilla_lstm import M1_VanillaLSTM
from .m2_bilstm import M2_BiLSTM
from .m3_criss_cross import M3_CrissCrossBiLSTM
from .m4_cnn_lstm import M4_CNNLSTM
from .m5_feature_bilstm import M5_FeatureBiLSTM
from .m6_graph_bilstm import M6_GraphBiLSTM

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "m1_vanilla_lstm": M1_VanillaLSTM,
    "m2_bilstm": M2_BiLSTM,
    "m3_criss_cross": M3_CrissCrossBiLSTM,
    "m4_cnn_lstm": M4_CNNLSTM,
    "m5_feature_bilstm": M5_FeatureBiLSTM,
    "m6_graph_bilstm": M6_GraphBiLSTM,
}

__all__ = [
    "M1_VanillaLSTM", "M2_BiLSTM", "M3_CrissCrossBiLSTM",
    "M4_CNNLSTM", "M5_FeatureBiLSTM", "M6_GraphBiLSTM",
    "MODEL_REGISTRY", "get_benchmark_model",
]


def get_benchmark_model(model_name: str, **kwargs: object) -> nn.Module:
    """
    Instantiate a benchmark LSTM model by name.

    Args:
        model_name: Key in MODEL_REGISTRY (e.g. 'm1_vanilla_lstm').
        **kwargs: Forwarded to the model constructor.

    Returns:
        Instantiated nn.Module.

    Raises:
        ValueError: If model_name is not in MODEL_REGISTRY.

    Example:
        >>> model = get_benchmark_model("m2_bilstm", hidden_size=256)
    """
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY.keys())}"
        )
    model_cls = MODEL_REGISTRY[key]
    sig = inspect.signature(model_cls.__init__)
    valid_params = {k for k in sig.parameters if k != "self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return model_cls(**filtered)
