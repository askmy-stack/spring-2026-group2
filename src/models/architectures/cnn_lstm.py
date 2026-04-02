"""
CNN-LSTM Hybrid Classifier
============================
1D CNN extracts local spatial-temporal features from raw EEG,
then BiLSTM captures sequential dependencies, followed by multi-head attention.

Improvements over original:
- 3rd CNN block (was 2) — deeper feature hierarchy
- Replaced MaxPool1d with strided Conv1d — preserves learned features vs discarding
- Residual skip connections in each CNN block (1x1 projection when dims change)
- SE (Squeeze-and-Excitation) channel attention after each block — learns which
  feature maps matter most, directly targeting seizure-discriminative frequencies
- Global avg-pool + max-pool over LSTM timesteps (vs single final hidden state)
- Multi-head self-attention (4 heads) over LSTM outputs before pooling
- LayerNorm + 2-layer FC head
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D feature maps."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),           # (batch, C, 1)
            nn.Flatten(),                       # (batch, C)
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, C, T)
        scale = self.se(x).unsqueeze(-1)  # (batch, C, 1)
        return x * scale


class ConvBlock(nn.Module):
    """Conv1d block with BN, ReLU, strided downsampling, SE attention, and residual."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.se = SEBlock(out_ch)

        # 1x1 projection for residual when dims change
        self.residual_proj = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual_proj(x)
        out = self.conv(x)
        out = self.se(out)
        # Align lengths in case of rounding differences
        min_len = min(out.size(-1), residual.size(-1))
        return out[..., :min_len] + residual[..., :min_len]


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int = 16,
        seq_len: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 1,
        cnn_filters: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.num_classes = num_classes

        # --- CNN Feature Extractor (3 blocks, strided convolutions) ---
        # Block 1: n_channels -> cnn_filters, stride=2 (256 -> ~128)
        # Block 2: cnn_filters -> cnn_filters*2, stride=2 (~128 -> ~64)
        # Block 3: cnn_filters*2 -> cnn_filters*2, stride=1 (refine, no downsample)
        self.cnn = nn.Sequential(
            ConvBlock(n_channels, cnn_filters, kernel_size=7, stride=2,
                      dropout=dropout * 0.5),
            ConvBlock(cnn_filters, cnn_filters * 2, kernel_size=5, stride=2,
                      dropout=dropout * 0.5),
            ConvBlock(cnn_filters * 2, cnn_filters * 2, kernel_size=3, stride=1,
                      dropout=dropout * 0.5),
        )

        cnn_out_features = cnn_filters * 2  # 128

        # --- BiLSTM Sequence Modeler ---
        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * 2

        # --- Multi-head self-attention over LSTM outputs ---
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

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time_steps) raw EEG tensor
        Returns:
            logits: (batch, 1)
        """
        # CNN: (batch, n_channels, time_steps) -> (batch, cnn_filters*2, reduced_time)
        cnn_out = self.cnn(x)

        # Reshape for LSTM: (batch, reduced_time, cnn_features)
        cnn_out = cnn_out.permute(0, 2, 1)

        # BiLSTM: (batch, reduced_time, hidden_size * 2)
        lstm_out, _ = self.lstm(cnn_out)

        # Multi-head self-attention with residual
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended = self.attn_norm(attended + lstm_out)

        # Global avg + max pooling over full sequence
        avg_pool = attended.mean(dim=1)
        max_pool = attended.max(dim=1).values
        out = torch.cat([avg_pool, max_pool], dim=1)

        out = self.pool_norm(out)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
