"""
Attention-Based Bidirectional LSTM Classifier
================================================
BiLSTM + learned attention weights over time steps.

Improvements over original:
- Replaced single-head Bahdanau attention with 4-head MultiheadAttention
  (nn.MultiheadAttention) — head diversity captures different seizure patterns
- Added pre-attention residual connection: attended + lstm_out (stabilises gradients)
- Global avg-pool + max-pool over attended sequence (full context)
- LayerNorm after pooled output
- 2-layer FC head with ReLU + dropout
- Input LayerNorm + projection (matches VanillaLSTM / BiLSTM improvements)
- forward_with_attention() preserved for interpretability / visualisation
"""

import torch
import torch.nn as nn


class AttentionBiLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int = 16,
        seq_len: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
        num_heads: int = 4,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.num_classes = num_classes

        # Input normalisation + projection
        self.input_norm = nn.LayerNorm(n_channels)
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * 2  # bidirectional

        # Multi-head self-attention over BiLSTM outputs
        # embed_dim must be divisible by num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_size)

        # avg-pool + max-pool -> lstm_out_size * 2
        self.pool_norm = nn.LayerNorm(lstm_out_size * 2)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def _run(self, x):
        """Shared forward logic — returns logits and attention weight map."""
        x = x.permute(0, 2, 1)  # (batch, time_steps, channels)
        x = self.input_norm(x)
        x = self.input_proj(x)   # (batch, time_steps, hidden_size)

        lstm_out, _ = self.lstm(x)  # (batch, time_steps, hidden_size * 2)

        # Multi-head self-attention (query = key = value = lstm_out)
        attended, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        # attn_weights: (batch, time_steps, time_steps)

        # Residual connection: keep both attended and raw LSTM context
        attended = self.attn_norm(attended + lstm_out)

        # Global avg + max pooling over attended sequence
        avg_pool = attended.mean(dim=1)
        max_pool = attended.max(dim=1).values
        out = torch.cat([avg_pool, max_pool], dim=1)  # (batch, hidden_size * 4)

        out = self.pool_norm(out)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits, attn_weights

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_steps) raw EEG tensor
        Returns:
            logits: (batch, 1)
        """
        logits, _ = self._run(x)
        return logits

    def forward_with_attention(self, x):
        """Same as forward but also returns attention weights for visualisation.

        Returns:
            logits:       (batch, 1)
            attn_weights: (batch, time_steps, time_steps)
        """
        return self._run(x)
