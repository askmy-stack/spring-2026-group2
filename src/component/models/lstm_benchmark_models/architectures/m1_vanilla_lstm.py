"""M1: VanillaLSTM + Channel Attention — Benchmark Model. Expected F1: 0.62-0.70."""
import logging

import torch
import torch.nn as nn

from ..modules.channel_attention import ChannelAttention

logger = logging.getLogger(__name__)


class M1_VanillaLSTM(nn.Module):
    """
    VanillaLSTM with Channel Attention for EEG seizure detection.

    Architecture:
        Input (batch, n_channels, time_steps)
        -> ChannelAttention -> LayerNorm -> LinearProjection
        -> LSTM (num_layers) -> AvgPool + MaxPool -> FC Head -> Logits

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Timesteps per window (default: 256).
        hidden_size: LSTM hidden dimension (default: 128).
        num_layers: Number of stacked LSTM layers (default: 2).
        dropout: Dropout rate (default: 0.3).

    Example:
        >>> model = M1_VanillaLSTM(n_channels=16, hidden_size=128)
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
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.channel_attn = ChannelAttention(n_channels, reduction=4)
        self.input_norm = nn.LayerNorm(n_channels)
        self.input_proj = _build_input_proj(n_channels, hidden_size, dropout)
        self.lstm = _build_lstm(hidden_size, num_layers, dropout, bidirectional=False)
        self.pool_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = _build_classifier(hidden_size, hidden_size * 2, dropout)

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            eeg_input: EEG windows, shape (batch, n_channels, time_steps)

        Returns:
            Logits, shape (batch, 1).
        """
        if eeg_input.ndim != 3:
            raise ValueError(f"M1_VanillaLSTM expects (batch, channels, time), got shape {tuple(eeg_input.shape)}")
        attended = self.channel_attn(eeg_input)
        projected = self._project(attended)
        lstm_out, _ = self.lstm(projected)
        pooled = self._pool_and_norm(lstm_out)
        return self.classifier(pooled)

    def _project(self, attended: torch.Tensor) -> torch.Tensor:
        """Normalize and project channel-attended input for LSTM."""
        return self.input_proj(self.input_norm(attended.permute(0, 2, 1)))

    def _pool_and_norm(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Combine avg and max pool over time; apply norm and dropout."""
        avg_pooled = lstm_out.mean(dim=1)
        max_pooled = lstm_out.max(dim=1).values
        pooled = torch.cat([avg_pooled, max_pooled], dim=-1)
        return self.dropout_layer(self.pool_norm(pooled))


def _build_input_proj(n_channels: int, hidden_size: int, dropout: float) -> nn.Sequential:
    """Build linear input projection block."""
    return nn.Sequential(
        nn.Linear(n_channels, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout * 0.5),
    )


def _build_lstm(
    hidden_size: int, num_layers: int, dropout: float, bidirectional: bool = False
) -> nn.LSTM:
    """Build stacked (optionally bidirectional) LSTM module."""
    return nn.LSTM(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True,
        bidirectional=bidirectional,
        dropout=dropout if num_layers > 1 else 0.0,
    )


def _build_classifier(hidden_size: int, input_size: int, dropout: float) -> nn.Sequential:
    """Build two-layer FC classification head."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 1),
    )
