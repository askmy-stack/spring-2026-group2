"""
Attention-Based Bidirectional LSTM Classifier
================================================
BiLSTM + learned attention weights over time steps.
Lets the model focus on the most seizure-relevant segments of the EEG window.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Additive (Bahdanau-style) attention over LSTM hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.context = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
        Returns:
            context_vector: (batch, hidden_size) - weighted sum of hidden states
            attn_weights:   (batch, seq_len) - attention distribution
        """
        # Score each time step
        energy = torch.tanh(self.attn(lstm_output))  # (batch, seq_len, hidden_size)
        scores = self.context(energy).squeeze(-1)     # (batch, seq_len)

        attn_weights = F.softmax(scores, dim=1)       # (batch, seq_len)

        # Weighted sum of LSTM outputs
        context_vector = torch.bmm(
            attn_weights.unsqueeze(1), lstm_output
        ).squeeze(1)  # (batch, hidden_size)

        return context_vector, attn_weights


class AttentionBiLSTM(nn.Module):
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

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention operates on BiLSTM output (hidden_size * 2)
        self.attention = Attention(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_steps) raw EEG tensor
        Returns:
            logits: (batch, 1)
        """
        x = x.permute(0, 2, 1)  # (batch, time_steps, channels)

        lstm_out, _ = self.lstm(x)  # (batch, time_steps, hidden_size * 2)

        # Attention-weighted context vector
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden_size * 2)

        out = self.dropout(context)
        logits = self.fc(out)
        return logits

    def forward_with_attention(self, x):
        """Same as forward but also returns attention weights for visualization."""
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context, attn_weights = self.attention(lstm_out)
        out = self.dropout(context)
        logits = self.fc(out)
        return logits, attn_weights
