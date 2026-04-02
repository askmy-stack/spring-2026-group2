"""
Bidirectional LSTM Classifier
===============================
Processes EEG in both forward and backward directions for richer temporal context.

Improvements over original:
- Fixed n_channels default (23 -> 16, matches pipeline)
- LayerNorm on raw input before processing
- Linear input projection (channel embedding)
- Global avg-pool + max-pool over all BiLSTM timestep outputs replaces
  single final-hidden-state readout (4x more context used)
- LayerNorm after pooled representation
- 2-layer FC head with ReLU + dropout
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int = 16,
        seq_len: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.num_classes = num_classes

        # Normalise raw channel inputs
        self.input_norm = nn.LayerNorm(n_channels)

        # Project channels to hidden_size
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

        # BiLSTM output is hidden_size*2; avg+max pool -> hidden_size*4
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
            x: (batch, channels, time_steps) raw EEG tensor
        Returns:
            logits: (batch, 1)
        """
        x = x.permute(0, 2, 1)  # (batch, time_steps, channels)

        x = self.input_norm(x)
        x = self.input_proj(x)  # (batch, time_steps, hidden_size)

        lstm_out, _ = self.lstm(x)  # (batch, time_steps, hidden_size * 2)

        # Global avg + max pooling over full sequence
        avg_pool = lstm_out.mean(dim=1)        # (batch, hidden_size * 2)
        max_pool = lstm_out.max(dim=1).values  # (batch, hidden_size * 2)
        out = torch.cat([avg_pool, max_pool], dim=1)  # (batch, hidden_size * 4)

        out = self.pool_norm(out)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
