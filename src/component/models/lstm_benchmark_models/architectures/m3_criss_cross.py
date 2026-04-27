"""M3: CrissCross BiLSTM — Benchmark Model. Expected F1: 0.75-0.82."""
import logging
from typing import Optional

import torch
import torch.nn as nn

from ..modules.channel_attention import ChannelAttention
from ..modules.criss_cross_attention import CrissCrossBlock
from .m1_vanilla_lstm import _build_input_proj, _build_lstm, _build_classifier

logger = logging.getLogger(__name__)


class M3_CrissCrossBiLSTM(nn.Module):
    """
    BiLSTM with Criss-Cross Attention (spatial + temporal).

    Architecture:
        Input (batch, n_channels, time_steps)
        -> ChannelAttention -> LayerNorm -> Projection
        -> BiLSTM -> Positional Encoding
        -> CrissCrossAttention x2 -> AvgPool + MaxPool -> FC Head -> Logits

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Timesteps per window (default: 256).
        hidden_size: Hidden dimension (default: 128).
        num_layers: BiLSTM layers (default: 2).
        num_heads: Attention heads (default: 4).
        dropout: Dropout rate (default: 0.3).

    Example:
        >>> model = M3_CrissCrossBiLSTM(n_channels=16)
        >>> logits = model(torch.randn(4, 16, 256))
        >>> assert logits.shape == (4, 1)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        bilstm_out = hidden_size * 2
        self.channel_attn = ChannelAttention(n_channels, reduction=4)
        self.input_norm = nn.LayerNorm(n_channels)
        self.input_proj = _build_input_proj(n_channels, hidden_size, dropout)
        self.lstm = _build_lstm(hidden_size, num_layers, dropout, bidirectional=True)
        self.pos_embedding = _build_pos_embedding(time_steps, bilstm_out)
        self.criss_cross = nn.Sequential(
            CrissCrossBlock(bilstm_out, num_heads, dropout),
            CrissCrossBlock(bilstm_out, num_heads, dropout),
        )
        self.pool_norm = nn.LayerNorm(bilstm_out * 2)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = _build_classifier(hidden_size, bilstm_out * 2, dropout)

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            eeg_input: EEG windows, shape (batch, n_channels, time_steps)

        Returns:
            Logits, shape (batch, 1).
        """
        attended = self.channel_attn(eeg_input)
        projected = self.input_proj(self.input_norm(attended.permute(0, 2, 1)))
        lstm_out, _ = self.lstm(projected)
        positioned = self._add_positional(lstm_out)
        criss_out = self.criss_cross(positioned)
        pooled = self._pool_and_norm(criss_out)
        return self.classifier(pooled)

    def _add_positional(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Add learnable positional encoding to LSTM output."""
        seq_len = lstm_out.size(1)
        return lstm_out + self.pos_embedding[:, :seq_len, :]

    def _pool_and_norm(self, attended: torch.Tensor) -> torch.Tensor:
        """Avg + max pool, normalise, dropout."""
        avg_pooled = attended.mean(dim=1)
        max_pooled = attended.max(dim=1).values
        pooled = torch.cat([avg_pooled, max_pooled], dim=-1)
        return self.dropout_layer(self.pool_norm(pooled))


def _build_pos_embedding(time_steps: int, embed_dim: int) -> nn.Parameter:
    """Build learnable positional embedding parameter."""
    param = nn.Parameter(torch.zeros(1, time_steps, embed_dim))
    nn.init.trunc_normal_(param, std=0.02)
    return param


def _build_classifier(hidden_size: int, input_size: int, dropout: float) -> nn.Sequential:
    """Build two-layer FC classification head."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 1),
    )
