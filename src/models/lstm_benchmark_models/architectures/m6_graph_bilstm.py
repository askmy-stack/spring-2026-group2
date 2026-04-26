"""M6: Graph BiLSTM (GMAEEG-style) — Benchmark Model. Expected F1: 0.78-0.85."""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.graph_attention import GraphAttention

logger = logging.getLogger(__name__)


class M6_GraphBiLSTM(nn.Module):
    """
    Graph Attention BiLSTM for EEG seizure detection.

    Models EEG electrodes as graph nodes, applies GAT per time step,
    then runs BiLSTM for temporal modeling.

    Architecture:
        Input (batch, n_channels, time_steps)
        -> Per-channel temporal CNN -> Graph Attention (per time step)
        -> BiLSTM -> MultiheadAttention -> AvgPool + MaxPool -> FC -> Logits

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Timesteps per window (default: 256).
        hidden_size: Hidden dimension (default: 128).
        num_layers: BiLSTM layers (default: 2).
        num_heads: Graph + LSTM attention heads (default: 4).
        dropout: Dropout rate (default: 0.3).

    Example:
        >>> model = M6_GraphBiLSTM(n_channels=16)
        >>> logits = model(torch.randn(2, 16, 256))
        >>> assert logits.shape == (2, 1)
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
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        bilstm_out = hidden_size * 2
        self.temporal_conv = _build_temporal_conv(dropout)
        self.graph_layers = nn.ModuleList([
            GraphAttention(64, hidden_size // num_heads, n_channels, num_heads, dropout),
            GraphAttention(hidden_size, hidden_size // num_heads, n_channels, num_heads, dropout),
        ])
        self.graph_norms = nn.ModuleList([nn.LayerNorm(hidden_size), nn.LayerNorm(hidden_size)])
        self.lstm = nn.LSTM(hidden_size * n_channels, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
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
        batch, n_channels, _ = eeg_input.shape
        channel_features = self._extract_channel_features_vectorised(eeg_input, batch, n_channels)
        graph_out = self._apply_graph_over_time_vectorised(channel_features, batch, n_channels)
        lstm_out, _ = self.lstm(graph_out)
        attended = self._apply_self_attention(lstm_out)
        pooled = self._pool_and_norm(attended)
        return self.classifier(pooled)

    def _extract_channel_features_vectorised(
        self, eeg_input: torch.Tensor, batch: int, n_channels: int
    ) -> torch.Tensor:
        """Apply per-channel temporal CNN in one pass by folding channels into the batch dim.

        Returns ``(batch, n_channels, 64, time_conv)`` — same layout as the old
        loop-based implementation but with a single conv forward instead of
        ``n_channels`` Python dispatches.
        """
        time_steps = eeg_input.size(-1)
        cnn_in = eeg_input.reshape(batch * n_channels, 1, time_steps)
        cnn_out = self.temporal_conv(cnn_in)              # (B*C, 64, T')
        time_conv = cnn_out.size(-1)
        return cnn_out.reshape(batch, n_channels, 64, time_conv)

    def _apply_graph_over_time_vectorised(
        self, channel_features: torch.Tensor, batch: int, n_channels: int
    ) -> torch.Tensor:
        """Run each GAT layer once over the folded ``(batch*time, n_channels, feat)`` tensor.

        Returns ``(batch, time_conv, n_channels * hidden_size)`` for the LSTM.
        """
        time_conv = channel_features.size(-1)
        # (B, C, 64, T') -> (B, T', C, 64) -> (B*T', C, 64)
        node_features = (
            channel_features.permute(0, 3, 1, 2).reshape(batch * time_conv, n_channels, 64)
        )
        for graph_layer, graph_norm in zip(self.graph_layers, self.graph_norms):
            node_features = graph_norm(F.relu(graph_layer(node_features)))
        # (B*T', C, hidden) -> (B, T', C*hidden)
        return node_features.reshape(batch, time_conv, n_channels * self.hidden_size)

    def _apply_self_attention(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Residual multi-head self-attention on LSTM output."""
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.attn_norm(attended + lstm_out)

    def _pool_and_norm(self, attended: torch.Tensor) -> torch.Tensor:
        """Avg + max pool, normalise, dropout."""
        avg_pooled = attended.mean(dim=1)
        max_pooled = attended.max(dim=1).values
        pooled = torch.cat([avg_pooled, max_pooled], dim=-1)
        return self.dropout_layer(self.pool_norm(pooled))


def _build_temporal_conv(dropout: float) -> nn.Sequential:
    """Build per-channel temporal feature extractor."""
    return nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=7, padding=3),
        nn.BatchNorm1d(32),
        nn.GELU(),
        nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm1d(64),
        nn.GELU(),
    )


def _build_classifier(hidden_size: int, input_size: int, dropout: float) -> nn.Sequential:
    """Build two-layer FC classification head."""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 1),
    )
