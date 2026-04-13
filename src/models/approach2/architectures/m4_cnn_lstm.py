"""
M4: CNN-LSTM with Multi-Scale Branches
--------------------------------------
Three parallel CNN branches capturing different temporal scales,
followed by BiLSTM and attention.

Expected F1: 0.80-0.88 (with pre-training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from modules.channel_attention import SEBlock
from modules.pretrained_encoders import load_pretrained_encoder


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm, activation, and optional SE attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
        use_se: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # SE attention
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

        # Residual connection if dimensions match
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.se(x)
        x = self.dropout(x)
        return x + residual


class M4_CNNLSTM(nn.Module):
    """
    CNN-LSTM with Multi-Scale Branches.

    Architecture:
        Input (batch, 16, 256)
        → [Optional: Pretrained Encoder]
        → 3 Parallel CNN Branches:
            - Branch A: kernel=3 (fast, spike transients)
            - Branch B: kernel=15 (medium, spike-wave)
            - Branch C: kernel=31 (slow, ictal rhythm)
        → Concatenate → BiLSTM
        → Multi-Head Attention
        → AvgPool + MaxPool
        → FC Head → Logits

    Args:
        n_channels: Number of EEG channels (default: 16)
        time_steps: Number of time steps (default: 256)
        hidden_size: Hidden size (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.3)
        use_pretrained: Whether to use pretrained encoder
        pretrained_encoder: Name of pretrained encoder
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_pretrained: bool = False,
        pretrained_encoder: Optional[str] = "cbramod",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.time_steps = time_steps
        self.hidden_size = hidden_size
        self.use_pretrained = use_pretrained

        # Optional pretrained encoder
        if use_pretrained and pretrained_encoder:
            self.pretrained = load_pretrained_encoder(
                pretrained_encoder,
                n_channels=n_channels,
                time_steps=time_steps,
                hidden_dim=hidden_size,
                pretrained=True,
                freeze=True,
            )
        else:
            self.pretrained = None

        # Branch A: Fast (kernel=3) — captures spike transients
        self.branch_a = nn.Sequential(
            ConvBlock(n_channels, 32, kernel_size=3, stride=2, dropout=dropout * 0.5),
            ConvBlock(32, 32, kernel_size=3, stride=1, dropout=dropout * 0.5),
        )

        # Branch B: Medium (kernel=15) — captures spike-wave complexes (~250ms)
        self.branch_b = nn.Sequential(
            ConvBlock(n_channels, 64, kernel_size=15, stride=2, dropout=dropout * 0.5),
            ConvBlock(64, 64, kernel_size=3, stride=1, dropout=dropout * 0.5),
        )

        # Branch C: Slow (kernel=31) — captures ictal rhythm (~1s)
        self.branch_c = nn.Sequential(
            ConvBlock(n_channels, 64, kernel_size=31, stride=1, dropout=dropout * 0.5),
            ConvBlock(64, 64, kernel_size=3, stride=1, dropout=dropout * 0.5),
        )

        # Total features after concatenation: 32 + 64 + 64 = 160
        cnn_out_features = 32 + 64 + 64

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * 2

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
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
        fc_input_size = lstm_out_size * 2
        if use_pretrained:
            fc_input_size += hidden_size

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, hidden_size),
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
        # Get pretrained features if available
        pretrained_feat = None
        if self.pretrained is not None:
            pretrained_feat = self.pretrained(x)

        # Process three parallel branches
        a = self.branch_a(x)  # (batch, 32, T/2)
        b = self.branch_b(x)  # (batch, 64, T/2)
        c = self.branch_c(x)  # (batch, 64, T)

        # Align temporal dimensions
        target_len = min(a.size(2), b.size(2))
        a = F.adaptive_avg_pool1d(a, target_len)
        b = F.adaptive_avg_pool1d(b, target_len)
        c = F.adaptive_avg_pool1d(c, target_len)

        # Concatenate: (batch, 160, T')
        merged = torch.cat([a, b, c], dim=1)

        # Transpose for LSTM: (batch, T', 160)
        merged = merged.permute(0, 2, 1)

        # BiLSTM
        lstm_out, _ = self.lstm(merged)  # (batch, T', hidden_size * 2)

        # Self-attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended = self.attn_norm(attended + lstm_out)  # Residual

        # Global pooling: avg + max
        avg_pool = attended.mean(dim=1)
        max_pool = attended.max(dim=1).values
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # Combine with pretrained features
        if pretrained_feat is not None:
            pooled = torch.cat([pooled, pretrained_feat], dim=1)

        # Normalize and dropout
        pooled_norm = self.pool_norm(pooled[:, :self.hidden_size * 4])
        if pretrained_feat is not None:
            pooled = torch.cat([pooled_norm, pretrained_feat], dim=1)
        else:
            pooled = pooled_norm
        pooled = self.dropout_layer(pooled)

        # Classify
        logits = self.fc(pooled)

        return logits


if __name__ == "__main__":
    # Test
    model = M4_CNNLSTM(n_channels=16, time_steps=256, use_pretrained=False)
    x = torch.randn(8, 16, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
