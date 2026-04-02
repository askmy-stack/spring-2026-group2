"""
Vanilla LSTM Classifier
========================
Baseline unidirectional LSTM for seizure detection.
Processes raw EEG tensor (batch, channels, time_steps) -> seizure/non-seizure.
"""

import torch
import torch.nn as nn


class VanillaLSTM(nn.Module):
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
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM expects input shape: (batch, seq_len, input_size)
        # We treat time_steps as the sequence and channels as features
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_steps) raw EEG tensor
        Returns:
            logits: (batch, 1) seizure probability (pre-sigmoid)
        """
        # Reshape: (batch, channels, time_steps) -> (batch, time_steps, channels)
        x = x.permute(0, 2, 1)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state from the final layer
        out = h_n[-1]  # (batch, hidden_size)

        out = self.dropout(out)
        logits = self.fc(out)  # (batch, 1)
        return logits
