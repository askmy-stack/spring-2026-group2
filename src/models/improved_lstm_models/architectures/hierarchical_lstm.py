"""
Hierarchical LSTM for long-context EEG seizure prediction.

Two-level hierarchy:
    Level 1 — encode individual 1-second windows via CNN.
    Level 2 — LSTM over a sequence of windows (30-60 min context).

References:
    Seizure prediction with long-term EEG (hierarchical attention)
"""
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WindowEncoder(nn.Module):
    """
    Encodes individual EEG windows (1 second) via CNN.

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Time steps per window (default: 256).
        hidden_size: Output embedding dimension (default: 128).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> enc = WindowEncoder(n_channels=16, hidden_size=128)
        >>> out = enc(torch.randn(4, 16, 256))
        >>> assert out.shape == (4, 128)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = _build_window_cnn(n_channels)
        self.proj = nn.Linear(128, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_input: Single EEG window, shape (batch, n_channels, time_steps)

        Returns:
            Window embedding, shape (batch, hidden_size).
        """
        cnn_features = self.cnn(eeg_input).squeeze(-1)
        return self.norm(self.proj(cnn_features))


class HierarchicalLSTM(nn.Module):
    """
    Hierarchical LSTM for long-context seizure prediction.

    Architecture:
        Input (batch, n_windows, n_channels, time_steps)
        -> Level 1: WindowEncoder per window -> (batch, n_windows, hidden)
        -> Level 2: BiLSTM over windows -> Attention pooling -> Classifier

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Time steps per window (default: 256).
        n_windows: Context length in windows (default: 60).
        hidden_size: Hidden dimension (default: 128).
        num_layers: BiLSTM layer count (default: 2).
        dropout: Dropout probability (default: 0.2).

    Example:
        >>> model = HierarchicalLSTM(n_channels=16, n_windows=10)
        >>> out = model(torch.randn(2, 10, 16, 256))
        >>> assert out.shape == (2, 1)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        n_windows: int = 60,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_windows = n_windows
        bilstm_out = hidden_size * 2
        self.window_encoder = WindowEncoder(n_channels, time_steps, hidden_size, dropout)
        self.context_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = _build_attention(bilstm_out)
        self.pool_norm = nn.LayerNorm(bilstm_out)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = _build_classifier(hidden_size, bilstm_out, dropout)

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_input: shape (batch, n_windows, n_channels, time_steps)

        Returns:
            Logits, shape (batch, 1).
        """
        if eeg_input.ndim != 4:
            raise ValueError(
                f"HierarchicalLSTM expects (batch, n_windows, channels, time), "
                f"got shape {tuple(eeg_input.shape)}. Unsqueeze dim=1 for single windows."
            )
        batch, n_windows, n_channels, time_steps = eeg_input.shape
        window_embeddings = self._encode_windows(eeg_input, batch, n_windows)
        context_out, _ = self.context_lstm(window_embeddings)
        pooled = self._attention_pool(context_out)
        return self.classifier(self.dropout_layer(self.pool_norm(pooled)))

    def _encode_windows(
        self, eeg_input: torch.Tensor, batch: int, n_windows: int
    ) -> torch.Tensor:
        """Encode each window independently via WindowEncoder."""
        flat = eeg_input.view(batch * n_windows, *eeg_input.shape[2:])
        encoded = self.window_encoder(flat)
        return encoded.view(batch, n_windows, -1)

    def _attention_pool(self, context_out: torch.Tensor) -> torch.Tensor:
        """Soft attention pooling over window sequence."""
        attn_weights = F.softmax(self.attention(context_out), dim=1)
        return (context_out * attn_weights).sum(dim=1)


def _build_window_cnn(n_channels: int) -> nn.Sequential:
    """Build CNN feature extractor for a single EEG window."""
    return nn.Sequential(
        nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
        nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        nn.Conv1d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm1d(128), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
    )


def _build_attention(hidden_dim: int) -> nn.Sequential:
    """Build scalar attention weight generator."""
    return nn.Sequential(nn.Linear(hidden_dim, 1))


def _build_classifier(hidden_size: int, input_size: int, dropout: float) -> nn.Sequential:
    """Build two-layer FC classification head."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 1),
    )
