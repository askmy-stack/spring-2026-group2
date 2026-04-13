"""
M1: VanillaLSTM + Channel Attention
-----------------------------------
Baseline LSTM with channel attention for electrode weighting.

Expected F1: 0.62-0.70
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from modules.channel_attention import ChannelAttention, SpatialChannelAttention


class M1_VanillaLSTM(nn.Module):
    """
    VanillaLSTM with Channel Attention.

    Architecture:
        Input (batch, 16, 256)
        → ChannelAttention (weight electrodes)
        → LayerNorm → Linear Projection
        → LSTM (2 layers)
        → AvgPool + MaxPool
        → FC Head → Logits

    Args:
        n_channels: Number of EEG channels (default: 16)
        time_steps: Number of time steps (default: 256)
        hidden_size: LSTM hidden size (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout rate (default: 0.3)
        use_pretrained: Whether to use pretrained encoder (not applicable for M1)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_pretrained: bool = False,
        pretrained_encoder: Optional[str] = None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.time_steps = time_steps
        self.hidden_size = hidden_size

        # Channel attention: learn which electrodes matter
        self.channel_attn = ChannelAttention(n_channels, reduction=4)

        # Input normalization and projection
        self.input_norm = nn.LayerNorm(n_channels)
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Pooling normalization
        self.pool_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: EEG signal (batch, n_channels, time_steps)

        Returns:
            logits: Classification logits (batch, 1)
        """
        # Channel attention
        x = self.channel_attn(x)  # (batch, n_channels, time_steps)

        # Reshape: (batch, time_steps, n_channels)
        x = x.permute(0, 2, 1)

        # Normalize and project
        x = self.input_norm(x)
        x = self.input_proj(x)  # (batch, time_steps, hidden_size)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time_steps, hidden_size)

        # Global pooling: avg + max
        avg_pool = lstm_out.mean(dim=1)  # (batch, hidden_size)
        max_pool = lstm_out.max(dim=1).values  # (batch, hidden_size)
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch, hidden_size * 2)

        # Normalize and dropout
        pooled = self.pool_norm(pooled)
        pooled = self.dropout(pooled)

        # Classify
        logits = self.fc(pooled)

        return logits

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get channel attention weights for interpretability."""
        with torch.no_grad():
            weights = self.channel_attn.gate(x)
        return weights


if __name__ == "__main__":
    # Test
    model = M1_VanillaLSTM(n_channels=16, time_steps=256)
    x = torch.randn(8, 16, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
