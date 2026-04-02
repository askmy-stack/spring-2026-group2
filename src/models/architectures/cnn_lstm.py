"""
CNN-LSTM Hybrid Classifier
============================
1D CNN extracts local spatial-temporal features from raw EEG,
then LSTM captures sequential dependencies across those features.
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int = 23,
        seq_len: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
        cnn_filters: int = 64,
        kernel_size: int = 7,
        pool_size: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # --- CNN Feature Extractor ---
        # Input: (batch, channels, time_steps) - treat EEG channels as Conv1d channels
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, cnn_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout * 0.5),

            # Block 2
            nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(pool_size),
            nn.Dropout(dropout * 0.5),
        )

        # After 2 pooling layers: seq_len -> seq_len // (pool_size^2)
        cnn_out_features = cnn_filters * 2

        # --- LSTM Sequence Modeler ---
        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_steps) raw EEG tensor
        Returns:
            logits: (batch, 1)
        """
        # CNN feature extraction
        # x is already (batch, channels, time_steps) which is what Conv1d expects
        cnn_out = self.cnn(x)  # (batch, cnn_filters*2, reduced_time)

        # Reshape for LSTM: (batch, reduced_time, cnn_features)
        cnn_out = cnn_out.permute(0, 2, 1)

        # LSTM over CNN feature sequence
        lstm_out, (h_n, c_n) = self.lstm(cnn_out)

        # Use final hidden states from both directions
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        out = torch.cat([h_forward, h_backward], dim=1)

        out = self.dropout(out)
        logits = self.fc(out)
        return logits
