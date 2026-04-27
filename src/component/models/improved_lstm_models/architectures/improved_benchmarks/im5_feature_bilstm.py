"""IM5: Improved FeatureBiLSTM — subclass of M5_FeatureBiLSTM.

M5 accepts both raw EEG ``(B, C, T)`` and pre-extracted features
``(B, seq_len, n_features)`` — IM5 inherits that dual path and just
bumps the backbone defaults.
"""
from src.models.lstm_benchmark_models.architectures.m5_feature_bilstm import (
    M5_FeatureBiLSTM,
)
from src.models.improved_lstm_models.modules.regularization import wrap_with_droppath


class IM5_FeatureBiLSTM(M5_FeatureBiLSTM):
    """Improved M5 with wider defaults + 8 attention heads."""

    def __init__(
        self,
        n_features: int = 226,
        seq_len: int = 10,
        hidden_size: int = 192,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.4,
        n_channels: int = 16,
        time_steps: int = 256,
        stochastic_depth: float = 0.1,
    ):
        super().__init__(
            n_features=n_features,
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            n_channels=n_channels,
            time_steps=time_steps,
        )
        self.stochastic_depth = stochastic_depth
        wrap_with_droppath(self, attr_name="dropout_layer", p=stochastic_depth)
