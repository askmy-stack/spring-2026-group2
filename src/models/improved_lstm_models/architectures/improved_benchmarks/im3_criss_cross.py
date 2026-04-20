"""IM3: Improved CrissCross BiLSTM — subclass of M3_CrissCrossBiLSTM.

M3 collapsed on the last run (AUC 0.35). IM3 uses narrower hidden size,
lower default dropout (so gradient signal isn't smothered on top of
criss-cross attention), and leaves heavy regularisation to the trainer
(augmentation + MixUp + SWA). Commit 2 also swaps the zero-init
positional embedding for a truncated-normal init which is already the
case in the parent, so no change needed here.
"""
from src.models.lstm_benchmark_models.architectures.m3_criss_cross import (
    M3_CrissCrossBiLSTM,
)


class IM3_CrissCrossBiLSTM(M3_CrissCrossBiLSTM):
    """Improved M3 — safer defaults to prevent collapse."""

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 160,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.3,
        stochastic_depth: float = 0.05,
    ):
        super().__init__(
            n_channels=n_channels,
            time_steps=time_steps,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.stochastic_depth = stochastic_depth
