"""
Improved Benchmark Architectures — im1..im7.

Each ``im_i`` subclasses its ``m_i`` counterpart from
``lstm_benchmark_models.architectures`` and bakes in stronger defaults
(wider backbones, more heads, heavier dropout) plus a ``stochastic_depth``
constructor kwarg that subsequent commits wire into the residual paths.

Input / output contract is identical to the originals:
    ``(batch, n_channels, time_steps) -> (batch, 1)``
so the existing ensemble's shape-aware ``_predict`` handles them with no
changes.
"""
from typing import Dict, Type

import torch.nn as nn

from .im1_vanilla_lstm import IM1_VanillaLSTM
from .im2_bilstm import IM2_BiLSTM
from .im3_criss_cross import IM3_CrissCrossBiLSTM
from .im4_cnn_lstm import IM4_CNNLSTM
from .im5_feature_bilstm import IM5_FeatureBiLSTM
from .im6_graph_bilstm import IM6_GraphBiLSTM
from .im7_attention_lstm import IM7_AttentionLSTM

IMPROVED_REGISTRY: Dict[str, Type[nn.Module]] = {
    "im1_vanilla_lstm": IM1_VanillaLSTM,
    "im2_bilstm": IM2_BiLSTM,
    "im3_criss_cross": IM3_CrissCrossBiLSTM,
    "im4_cnn_lstm": IM4_CNNLSTM,
    "im5_feature_bilstm": IM5_FeatureBiLSTM,
    "im6_graph_bilstm": IM6_GraphBiLSTM,
    "im7_attention_lstm": IM7_AttentionLSTM,
}

__all__ = [
    "IM1_VanillaLSTM", "IM2_BiLSTM", "IM3_CrissCrossBiLSTM",
    "IM4_CNNLSTM", "IM5_FeatureBiLSTM", "IM6_GraphBiLSTM", "IM7_AttentionLSTM",
    "IMPROVED_REGISTRY", "get_improved_model",
]


def get_improved_model(model_name: str, **kwargs: object) -> nn.Module:
    """Instantiate an improved benchmark model by registry key."""
    key = model_name.lower()
    if key not in IMPROVED_REGISTRY:
        raise ValueError(
            f"Unknown improved model '{model_name}'. "
            f"Available: {sorted(IMPROVED_REGISTRY.keys())}"
        )
    cls = IMPROVED_REGISTRY[key]
    return cls(**kwargs)
