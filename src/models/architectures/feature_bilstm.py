"""
Feature-Based BiLSTM Classifier
==================================
Operates on pre-extracted feature vectors (226 features) instead of raw EEG.

Improvements (v2):
- BatchNorm1d on raw 226 features
- Deep 2-layer projection MLP with residual shortcut
- Default seq_len 1 -> 10 (10-second pre-ictal context)
- BiLSTM over feature sequence
- Temporal MultiheadAttention over 10-window sequence (weights pre-ictal windows)
- Global avg+max pooling
- LayerNorm + 2-layer FC head
"""

import torch
import torch.nn as nn


class FeatureBiLSTM(nn.Module):
    def __init__(
        self,
        n_features: int = 226,
        seq_len: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
        num_heads: int = 4,
        # Kept for API compatibility with raw-signal models
        n_channels: int = 16,
    ):
        """
        Args:
            n_features: Number of extracted features per window (default 226).
            seq_len: Consecutive windows processed as one sequence.
                     Default 10 = 10 seconds of context at 1s windows.
            hidden_size: LSTM hidden dimension.
            num_layers: Number of stacked BiLSTM layers.
            dropout: Dropout probability.
            num_heads: Number of attention heads for temporal attention.
        """
        super().__init__()
        self.n_features  = n_features
        self.seq_len     = seq_len
        self.hidden_size = hidden_size

        # Normalise raw features (handles arbitrary feature scales)
        self.feature_norm = nn.BatchNorm1d(n_features)

        # Deep projection: 226 -> hidden_size*2 -> hidden_size with residual
        self.proj1 = nn.Sequential(
            nn.Linear(n_features, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        # 1-layer residual shortcut from input to proj2 output
        self.proj_shortcut = nn.Linear(n_features, hidden_size)
        self.proj_norm = nn.LayerNorm(hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * 2

        # Temporal attention: weights each of the 10 windows differently
        # Pre-ictal windows (just before seizure) should get higher weights
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=lstm_out_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_size)

        # BiLSTM output: hidden_size*2; avg+max pool -> hidden_size*4
        self.pool_norm = nn.LayerNorm(lstm_out_size * 2)
        self.dropout   = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, n_features) for single window, or
               (batch, seq_len, n_features) for sequential windows
        Returns:
            logits: (batch, 1)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, n_features)

        batch_size, actual_seq_len, n_feat = x.shape
        assert n_feat == self.n_features, (
            f"Expected {self.n_features} features, got {n_feat}"
        )

        # Apply BatchNorm across the feature dimension
        # Reshape to (batch*seq_len, n_features) for BN, then restore
        x_flat = x.reshape(batch_size * actual_seq_len, n_feat)
        x_flat = self.feature_norm(x_flat)
        x = x_flat.reshape(batch_size, actual_seq_len, n_feat)

        # Deep projection with residual: (batch, seq_len, hidden_size)
        shortcut = self.proj_shortcut(x)
        x = self.proj2(self.proj1(x))
        x = self.proj_norm(x + shortcut)

        # BiLSTM: (batch, seq_len, hidden_size * 2)
        lstm_out, _ = self.lstm(x)

        # Temporal MultiheadAttention: weights the 10 windows
        # Pre-ictal context (windows close to seizure onset) get higher weights
        attended, _ = self.temporal_attn(lstm_out, lstm_out, lstm_out)
        attended = self.attn_norm(attended + lstm_out)

        # Global avg + max pooling over sequence
        avg_pool = attended.mean(dim=1)
        max_pool = attended.max(dim=1).values
        out = torch.cat([avg_pool, max_pool], dim=1)  # (batch, hidden_size * 4)

        out = self.pool_norm(out)
        out = self.dropout(out)
        return self.fc(out)
