"""
Feature-Based BiLSTM Classifier
==================================
Operates on pre-extracted feature vectors (226 features) instead of raw EEG.
Integrates domain knowledge from clinical EEG analysis into the classification.

Improvements over original:
- BatchNorm1d on raw 226 features before processing (features may be unscaled)
- Deeper 2-layer projection MLP: 226 -> hidden_size*2 -> hidden_size
- Residual connection in projection block (stabilises gradient flow)
- LayerNorm between projection and LSTM
- Default seq_len changed 1 -> 10 (10-second context captures pre-ictal build-up)
- Global avg-pool + max-pool over LSTM outputs (vs single final hidden state)
- LayerNorm after pooling
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
        # Kept for API compatibility with raw-signal models
        n_channels: int = 16,
    ):
        """
        Args:
            n_features: Number of extracted features per window (default 226).
            seq_len: Consecutive windows processed as one sequence.
                     Default 10 = 10 seconds of context at 1s windows.
                     Use 1 for single-window classification.
            hidden_size: LSTM hidden dimension.
            num_layers:  Number of stacked BiLSTM layers.
            dropout:     Dropout probability.
        """
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
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

        # BiLSTM output: hidden_size*2; avg+max pool -> hidden_size*4
        self.pool_norm = nn.LayerNorm(hidden_size * 4)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
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

        # Global avg + max pooling over sequence
        avg_pool = lstm_out.mean(dim=1)
        max_pool = lstm_out.max(dim=1).values
        out = torch.cat([avg_pool, max_pool], dim=1)  # (batch, hidden_size * 4)

        out = self.pool_norm(out)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
