"""IM7: Improved Attention LSTM — subclass of M7_AttentionLSTM.

Deeper stack (4 self-attention blocks instead of 2) and 8 heads instead
of 4. Commit 2 wires DropPath into the residual paths.
"""
from src.models.lstm_benchmark_models.architectures.m7_attention_lstm import (
    M7_AttentionLSTM,
)


class IM7_AttentionLSTM(M7_AttentionLSTM):
    """Improved M7 — deeper attention stack + 8 heads."""

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 192,
        num_layers: int = 2,
        num_heads: int = 8,
        num_attn_blocks: int = 4,
        dropout: float = 0.4,
        stochastic_depth: float = 0.1,
    ):
        super().__init__(
            n_channels=n_channels,
            time_steps=time_steps,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_attn_blocks=num_attn_blocks,
            dropout=dropout,
        )
        self.stochastic_depth = stochastic_depth
