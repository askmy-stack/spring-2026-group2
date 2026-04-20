"""
M5: FeatureBiLSTM + Temporal Attention
--------------------------------------
BiLSTM for pre-extracted features with temporal attention
over sequential windows (10-window pre-ictal context).

Expected F1: 0.68-0.75
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


class M5_FeatureBiLSTM(nn.Module):
    """
    FeatureBiLSTM with Temporal Attention.

    Designed for pre-extracted features (e.g., statistical, spectral)
    with temporal context from multiple consecutive windows.

    Architecture:
        Input (batch, seq_len=10, n_features=226)
        → BatchNorm → Projection MLP
        → LayerNorm → BiLSTM
        → Temporal Multi-Head Attention
        → AvgPool + MaxPool
        → FC Head → Logits

    Args:
        n_features: Number of input features per window (default: 226)
        seq_len: Number of sequential windows (default: 10)
        hidden_size: Hidden size (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.3)
    """

    def __init__(
        self,
        n_features: int = 226,
        seq_len: int = 10,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_pretrained: bool = False,
        pretrained_encoder: Optional[str] = None,
        # For compatibility with raw EEG input
        n_channels: int = 16,
        time_steps: int = 256,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        # For raw EEG input compatibility
        self.n_channels = n_channels
        self.time_steps = time_steps
        self.use_raw_eeg = False

        # Raw EEG feature extractor (if needed)
        self.raw_feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(seq_len),  # Pool to seq_len windows
        )
        raw_feature_dim = 128

        # BatchNorm on input features
        self.input_bn = nn.BatchNorm1d(n_features)
        self.raw_input_bn = nn.BatchNorm1d(raw_feature_dim)

        # Projection MLP with residual
        proj_input = n_features
        self.proj = nn.Sequential(
            nn.Linear(proj_input, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.proj_shortcut = nn.Linear(proj_input, hidden_size)

        # For raw EEG
        self.raw_proj = nn.Sequential(
            nn.Linear(raw_feature_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.raw_proj_shortcut = nn.Linear(raw_feature_dim, hidden_size)

        # LayerNorm before LSTM
        self.pre_lstm_norm = nn.LayerNorm(hidden_size)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * 2

        # Temporal attention: weights each of the seq_len windows
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=lstm_out_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out_size)

        # Pooling normalization
        self.pool_norm = nn.LayerNorm(lstm_out_size * 2)
        self.dropout_layer = nn.Dropout(dropout)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Either:
               - Pre-extracted features (batch, seq_len, n_features)
               - Raw EEG (batch, n_channels, time_steps)

        Returns:
            logits: Classification logits (batch, 1)
        """
        # Detect input type
        if x.dim() == 3 and x.size(1) == self.n_channels and x.size(2) == self.time_steps:
            # Raw EEG input
            return self._forward_raw(x)
        else:
            # Pre-extracted features
            return self._forward_features(x)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for pre-extracted features."""
        batch, seq_len, n_features = x.shape

        # BatchNorm (transpose for batch norm)
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len)
        x = self.input_bn(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, n_features)

        # Projection with residual
        proj_out = self.proj(x)
        shortcut = self.proj_shortcut(x)
        x = proj_out + shortcut  # (batch, seq_len, hidden_size)

        # LayerNorm
        x = self.pre_lstm_norm(x)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * 2)

        # Temporal attention
        attended, _ = self.temporal_attn(lstm_out, lstm_out, lstm_out)
        attended = self.attn_norm(attended + lstm_out)  # Residual

        # Global pooling: avg + max
        avg_pool = attended.mean(dim=1)
        max_pool = attended.max(dim=1).values
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Normalize and dropout
        pooled = self.pool_norm(pooled)
        pooled = self.dropout_layer(pooled)

        # Classify
        logits = self.fc(pooled)

        return logits

    def _forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for raw EEG input."""
        # Extract features using CNN
        features = self.raw_feature_extractor(x)  # (batch, 128, seq_len)
        features = features.permute(0, 2, 1)  # (batch, seq_len, 128)

        batch, seq_len, feat_dim = features.shape

        # BatchNorm
        features = features.permute(0, 2, 1)
        features = self.raw_input_bn(features)
        features = features.permute(0, 2, 1)

        # Projection with residual
        proj_out = self.raw_proj(features)
        shortcut = self.raw_proj_shortcut(features)
        x = proj_out + shortcut

        # LayerNorm
        x = self.pre_lstm_norm(x)

        # BiLSTM
        lstm_out, _ = self.lstm(x)

        # Temporal attention
        attended, _ = self.temporal_attn(lstm_out, lstm_out, lstm_out)
        attended = self.attn_norm(attended + lstm_out)

        # Global pooling
        avg_pool = attended.mean(dim=1)
        max_pool = attended.max(dim=1).values
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Normalize and dropout
        pooled = self.pool_norm(pooled)
        pooled = self.dropout_layer(pooled)

        # Classify
        logits = self.fc(pooled)

        return logits


if __name__ == "__main__":
    # Test with pre-extracted features
    model = M5_FeatureBiLSTM(n_features=226, seq_len=10)
    x_feat = torch.randn(8, 10, 226)
    out = model(x_feat)
    print(f"Features input shape: {x_feat.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with raw EEG
    x_raw = torch.randn(8, 16, 256)
    out_raw = model(x_raw)
    print(f"Raw EEG input shape: {x_raw.shape}")
    print(f"Output shape: {out_raw.shape}")
