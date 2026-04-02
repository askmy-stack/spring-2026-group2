"""
Bidirectional LSTM Classifier
===============================
Processes EEG in both forward and backward directions for richer temporal context.
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int = 23,
        seq_len: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        # BiLSTM concatenates forward + backward -> hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_steps) raw EEG tensor
        Returns:
            logits: (batch, 1)
        """
        x = x.permute(0, 2, 1)  # (batch, time_steps, channels)

        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, time_steps, hidden_size * 2)

        # h_n shape: (num_layers * 2, batch, hidden_size)
        # Grab last layer forward and backward hidden states
        h_forward = h_n[-2]  # (batch, hidden_size)
        h_backward = h_n[-1]  # (batch, hidden_size)
        out = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_size * 2)

        out = self.dropout(out)
        logits = self.fc(out)
        return logits
