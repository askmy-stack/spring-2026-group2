"""
Approach 2 Model Architectures
------------------------------
7 diverse models for ensemble seizure detection.

M1: VanillaLSTM + ChannelAttn
M2: BiLSTM + ChannelAttn  
M3: AttentionBiLSTM + CrissCross
M4: CNN-LSTM (multi-scale)
M5: FeatureBiLSTM + TemporalAttn
M6: GraphBiLSTM
M7: VQ-Transformer
"""

from .m1_vanilla_lstm import M1_VanillaLSTM
from .m2_bilstm import M2_BiLSTM
from .m3_criss_cross import M3_CrissCrossBiLSTM
from .m4_cnn_lstm import M4_CNNLSTM
from .m5_feature_bilstm import M5_FeatureBiLSTM
from .m6_graph_bilstm import M6_GraphBiLSTM
from .m7_vq_transformer import M7_VQTransformer

MODEL_REGISTRY = {
    "m1_vanilla_lstm": M1_VanillaLSTM,
    "m2_bilstm": M2_BiLSTM,
    "m3_criss_cross": M3_CrissCrossBiLSTM,
    "m4_cnn_lstm": M4_CNNLSTM,
    "m5_feature_bilstm": M5_FeatureBiLSTM,
    "m6_graph_bilstm": M6_GraphBiLSTM,
    "m7_vq_transformer": M7_VQTransformer,
}


def get_model(model_name: str, **kwargs):
    """Get model by name."""
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)


def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    "M1_VanillaLSTM",
    "M2_BiLSTM",
    "M3_CrissCrossBiLSTM",
    "M4_CNNLSTM",
    "M5_FeatureBiLSTM",
    "M6_GraphBiLSTM",
    "M7_VQTransformer",
    "MODEL_REGISTRY",
    "get_model",
    "list_models",
]
