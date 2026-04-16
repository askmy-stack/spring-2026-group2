"""LSTM Benchmark Models — Directory 1 of 4. Six LSTM variants (m1-m6)."""
from .architectures import (
    M1_VanillaLSTM,
    M2_BiLSTM,
    M3_CrissCrossBiLSTM,
    M4_CNNLSTM,
    M5_FeatureBiLSTM,
    M6_GraphBiLSTM,
    MODEL_REGISTRY,
    get_benchmark_model,
)

__all__ = [
    "M1_VanillaLSTM", "M2_BiLSTM", "M3_CrissCrossBiLSTM",
    "M4_CNNLSTM", "M5_FeatureBiLSTM", "M6_GraphBiLSTM",
    "MODEL_REGISTRY", "get_benchmark_model",
]
