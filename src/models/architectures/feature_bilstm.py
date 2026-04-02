"""
Feature-Based BiLSTM Classifier
==================================
Operates on pre-extracted feature vectors (226 features) instead of raw EEG.
Integrates domain knowledge from clinical EEG analysis into the classification.
"""

import torch
import torch.nn as nn


class FeatureBiLSTM(nn.Module):
    def __init__(
        self,
        n_features: int = 226,
        seq_len: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
        # Ignored params for API compatibility with other models
        n_channels: int = 23,
    ):
        """
        Args:
            n_features: Number of extracted features per window (default 226).
            seq_len: Number of consecutive feature windows to process as a sequence.
                     Use 1 for single-window classification, or >1 to capture
                     temporal trends across consecutive windows.
            hidden_size: LSTM hidden dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        # Optional feature projection to reduce dimensionality before LSTM
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size),
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

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
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
        # Handle both single window and sequential inputs
        if x.dim() == 2:
            # Single window: (batch, n_features) -> (batch, 1, n_features)
            x = x.unsqueeze(1)

        batch_size, seq_len, n_feat = x.shape

        # Project features: (batch, seq_len, n_features) -> (batch, seq_len, hidden_size)
        x = self.feature_proj(x)

        # BiLSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Concatenate final forward and backward hidden states
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        out = torch.cat([h_forward, h_backward], dim=1)

        out = self.dropout(out)
        logits = self.fc(out)
        return logits
