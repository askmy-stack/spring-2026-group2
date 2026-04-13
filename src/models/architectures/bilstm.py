"""
Bidirectional LSTM Classifier
===============================
Processes EEG in both forward and backward directions.

Improvements (v2):
- ChannelAttention: learns per-channel scalar weights before LSTM
- Fixed n_channels default 23 -> 16
- LayerNorm on raw input
- Linear input projection
- Global avg+max pooling over all BiLSTM timestep outputs -> hidden*4
- LayerNorm after pooling
- 2-layer FC head
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Lightweight channel-wise attention over EEG electrodes.
    Learns which of the 16 channels are most discriminative for seizure detection.

    Input:  (batch, n_channels, time_steps)
    Output: (batch, n_channels, time_steps)  — channel-scaled
    """
    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(n_channels // reduction, 2)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels, mid),
            nn.ReLU(),
            nn.Linear(mid, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.gate(x).unsqueeze(-1)   # (batch, channels, 1)
        return x * weights


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
        self.n_channels  = n_channels
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout_p   = dropout
        self.num_classes = num_classes

        # Channel attention: focuses on seizure-relevant electrodes
        self.channel_attn = ChannelAttention(n_channels)

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

        # BiLSTM output: hidden_size*2; avg+max pool -> hidden_size*4
        self.pool_norm = nn.LayerNorm(hidden_size * 4)
        self.dropout   = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time_steps) raw EEG tensor
        Returns:
            logits: (batch, 1)
        """
        x = self.channel_attn(x)               # (batch, channels, time_steps)
        x = x.permute(0, 2, 1)                 # (batch, time_steps, channels)

        x = self.input_norm(x)
        x = self.input_proj(x)                 # (batch, time_steps, hidden_size)

        lstm_out, _ = self.lstm(x)             # (batch, time_steps, hidden_size * 2)

        avg_pool = lstm_out.mean(dim=1)
        max_pool = lstm_out.max(dim=1).values
        out = torch.cat([avg_pool, max_pool], dim=1)  # (batch, hidden_size * 4)

        out = self.pool_norm(out)
        out = self.dropout(out)
        return self.fc(out)
