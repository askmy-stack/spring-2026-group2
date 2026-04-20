"""IM1: Improved VanillaLSTM — subclass of M1_VanillaLSTM.

Commit 1 skeleton: bumps backbone width defaults and accepts a
``stochastic_depth`` kwarg. Commit 2 wires DropPath + SE into the
residual paths.
"""
from src.models.lstm_benchmark_models.architectures.m1_vanilla_lstm import (
    M1_VanillaLSTM,
)


class IM1_VanillaLSTM(M1_VanillaLSTM):
    """Improved M1: wider hidden size + heavier dropout by default."""

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 192,
        num_layers: int = 2,
        dropout: float = 0.4,
        stochastic_depth: float = 0.1,
    ):
        super().__init__(
            n_channels=n_channels,
            time_steps=time_steps,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.stochastic_depth = stochastic_depth
