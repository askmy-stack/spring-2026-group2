"""M2: BiLSTM + Spatial-Channel Attention — Benchmark Model. Expected F1: 0.65-0.72."""
import logging

import torch
import torch.nn as nn

from ..modules.channel_attention import SpatialChannelAttention
from .m1_vanilla_lstm import _build_input_proj, _build_lstm, _build_classifier

logger = logging.getLogger(__name__)


class M2_BiLSTM(nn.Module):
    """
    Bidirectional LSTM with Spatial-Channel Attention.

    Architecture:
        Input (batch, n_channels, time_steps)
        -> SpatialChannelAttention -> LayerNorm -> LinearProjection
        -> BiLSTM (num_layers) -> AvgPool + MaxPool -> FC Head -> Logits

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Timesteps per window (default: 256).
        hidden_size: LSTM hidden dimension (default: 128).
        num_layers: Number of stacked BiLSTM layers (default: 2).
        dropout: Dropout rate (default: 0.3).

    Example:
        >>> model = M2_BiLSTM(n_channels=16, hidden_size=128)
        >>> logits = model(torch.randn(8, 16, 256))
        >>> assert logits.shape == (8, 1)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        bilstm_out = hidden_size * 2
        self.channel_attn = SpatialChannelAttention(n_channels, reduction=4)
        self.input_norm = nn.LayerNorm(n_channels)
        self.input_proj = _build_input_proj(n_channels, hidden_size, dropout)
        self.lstm = _build_lstm(hidden_size, num_layers, dropout, bidirectional=True)
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
        pooled = self._pool_bilstm(lstm_out)
        return self.classifier(pooled)

    def _pool_bilstm(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Avg + max pool over time for BiLSTM output; apply norm and dropout."""
        avg_pooled = lstm_out.mean(dim=1)
        max_pooled = lstm_out.max(dim=1).values
        pooled = torch.cat([avg_pooled, max_pooled], dim=-1)
        return self.dropout_layer(self.pool_norm(pooled))
