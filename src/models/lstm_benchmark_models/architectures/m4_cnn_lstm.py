"""M4: CNN-LSTM Multi-Scale — Benchmark Model. Expected F1: 0.80-0.88."""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_CNN_OUT_FEATURES = 32 + 64 + 64


class ConvBlock(nn.Module):
    """
    1D convolutional block with BatchNorm, GELU, and residual connection.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Convolution kernel width.
        stride: Convolution stride.
        dropout: Dropout probability.

    Example:
        >>> block = ConvBlock(16, 32, kernel_size=3, stride=2)
        >>> out = block(torch.randn(4, 16, 256))
        >>> assert out.shape == (4, 32, 128)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_tensor: shape (batch, in_channels, time_steps)

        Returns:
            Convolved features with residual, same or downsampled time dim.
        """
        skip = self.residual(eeg_tensor)
        cnn_features = self.dropout(self.activation(self.bn(self.conv(eeg_tensor))))
        return cnn_features + skip


class M4_CNNLSTM(nn.Module):
    """
    Multi-scale CNN branches + BiLSTM + self-attention.

    Architecture:
        Input (batch, n_channels, time_steps)
        -> 3 parallel CNN branches (kernels 3, 15, 31)
        -> Concatenate -> BiLSTM -> MultiheadAttention
        -> AvgPool + MaxPool -> FC Head -> Logits

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Timesteps per window (default: 256).
        hidden_size: LSTM hidden dimension (default: 128).
        num_layers: BiLSTM layers (default: 2).
        num_heads: Attention heads (default: 4).
        dropout: Dropout rate (default: 0.3).

    Example:
        >>> model = M4_CNNLSTM(n_channels=16)
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
        self.branch_a = _build_cnn_branch(n_channels, 32, kernel_size=3, stride=2, dropout=dropout)
        self.branch_b = _build_cnn_branch(n_channels, 64, kernel_size=15, stride=2, dropout=dropout)
        self.branch_c = _build_cnn_branch(n_channels, 64, kernel_size=31, stride=1, dropout=dropout)
        self.lstm = _build_lstm(_CNN_OUT_FEATURES, hidden_size, num_layers, dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(bilstm_out, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(bilstm_out)
        self.pool_norm = nn.LayerNorm(bilstm_out * 2)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = _build_classifier(hidden_size, bilstm_out * 2, dropout)

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_input: shape (batch, n_channels, time_steps)

        Returns:
            Logits, shape (batch, 1).
        """
        cnn_features = self._extract_cnn_features(eeg_input)
        lstm_out, _ = self.lstm(cnn_features.permute(0, 2, 1))
        attended = self._apply_self_attention(lstm_out)
        pooled = self._pool_and_norm(attended)
        return self.classifier(pooled)

    def _extract_cnn_features(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """Run three parallel CNN branches and concatenate aligned features."""
        branch_a = self.branch_a(eeg_input)
        branch_b = self.branch_b(eeg_input)
        branch_c = self.branch_c(eeg_input)
        target_len = min(branch_a.size(2), branch_b.size(2))
        aligned_a = F.adaptive_avg_pool1d(branch_a, target_len)
        aligned_b = F.adaptive_avg_pool1d(branch_b, target_len)
        aligned_c = F.adaptive_avg_pool1d(branch_c, target_len)
        return torch.cat([aligned_a, aligned_b, aligned_c], dim=1)

    def _apply_self_attention(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Apply residual multi-head self-attention."""
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.attn_norm(attended + lstm_out)

    def _pool_and_norm(self, attended: torch.Tensor) -> torch.Tensor:
        """Avg + max pool, normalise, dropout."""
        avg_pooled = attended.mean(dim=1)
        max_pooled = attended.max(dim=1).values
        pooled = torch.cat([avg_pooled, max_pooled], dim=-1)
        return self.dropout_layer(self.pool_norm(pooled))


def _build_cnn_branch(
    n_channels: int, out_channels: int, kernel_size: int, stride: int, dropout: float
) -> nn.Sequential:
    """Build a two-block CNN feature extraction branch."""
    return nn.Sequential(
        ConvBlock(n_channels, out_channels, kernel_size=kernel_size, stride=stride, dropout=dropout * 0.5),
        ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, dropout=dropout * 0.5),
    )


def _build_lstm(input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool) -> nn.LSTM:
    """Build stacked (optionally bidirectional) LSTM module."""
    return nn.LSTM(
        input_size=input_size,
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
