"""
M3: AttentionBiLSTM + Criss-Cross Attention
-------------------------------------------
BiLSTM with criss-cross attention (separate spatial/temporal).
Inspired by CBraMod (ICLR 2025).

Expected F1: 0.75-0.82 (with pre-training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from modules.channel_attention import ChannelAttention
from modules.criss_cross_attention import CrissCrossBlock
from modules.pretrained_encoders import load_pretrained_encoder


class M3_CrissCrossBiLSTM(nn.Module):
    """
    BiLSTM with Criss-Cross Attention.

    Architecture:
        Input (batch, 16, 256)
        → [Optional: Pretrained Encoder]
        → ChannelAttention
        → LayerNorm → Linear Projection
        → BiLSTM
        → Learnable Positional Encoding
        → Criss-Cross Attention (spatial ⊥ temporal)
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
        pretrained_encoder: Name of pretrained encoder ('cbramod', 'eegpt', etc.)
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
            self.pretrained_proj = nn.Linear(hidden_size, hidden_size)
        else:
            self.pretrained = None

        # Channel attention
        self.channel_attn = ChannelAttention(n_channels, reduction=4)

        # Input normalization and projection
        self.input_norm = nn.LayerNorm(n_channels)
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

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

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, time_steps, lstm_out_size)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # Criss-Cross Attention blocks
        self.criss_cross = nn.Sequential(
            CrissCrossBlock(lstm_out_size, num_heads, dropout),
            CrissCrossBlock(lstm_out_size, num_heads, dropout),
        )

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
        batch_size = x.shape[0]

        # Get pretrained features if available
        pretrained_feat = None
        if self.pretrained is not None:
            pretrained_feat = self.pretrained(x)
            pretrained_feat = self.pretrained_proj(pretrained_feat)

        # Channel attention
        x = self.channel_attn(x)  # (batch, n_channels, time_steps)

        # Reshape: (batch, time_steps, n_channels)
        x = x.permute(0, 2, 1)

        # Normalize and project
        x = self.input_norm(x)
        x = self.input_proj(x)  # (batch, time_steps, hidden_size)

        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, time_steps, hidden_size * 2)

        # Add positional encoding
        seq_len = lstm_out.size(1)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        lstm_out = lstm_out + pos_emb

        # Criss-Cross Attention
        attended = self.criss_cross(lstm_out)  # (batch, time_steps, hidden_size * 2)

        # Global pooling: avg + max
        avg_pool = attended.mean(dim=1)  # (batch, hidden_size * 2)
        max_pool = attended.max(dim=1).values  # (batch, hidden_size * 2)
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch, hidden_size * 4)

        # Combine with pretrained features
        if pretrained_feat is not None:
            pooled = torch.cat([pooled, pretrained_feat], dim=1)

        # Normalize and dropout
        pooled = self.pool_norm(pooled[:, :self.hidden_size * 4])
        if pretrained_feat is not None:
            pooled = torch.cat([pooled, pretrained_feat], dim=1)
        pooled = self.dropout_layer(pooled)

        # Classify
        logits = self.fc(pooled)

        return logits

    def forward_with_attention(self, x: torch.Tensor):
        """Forward pass returning attention weights for visualization."""
        # Similar to forward but return intermediate attention
        batch_size = x.shape[0]

        x = self.channel_attn(x)
        channel_weights = self.channel_attn.gate(x.permute(0, 2, 1).mean(dim=1, keepdim=True).permute(0, 2, 1))

        x = x.permute(0, 2, 1)
        x = self.input_norm(x)
        x = self.input_proj(x)

        lstm_out, _ = self.lstm(x)

        seq_len = lstm_out.size(1)
        pos_emb = self.pos_embedding[:, :seq_len, :]
        lstm_out = lstm_out + pos_emb

        attended = self.criss_cross(lstm_out)

        avg_pool = attended.mean(dim=1)
        max_pool = attended.max(dim=1).values
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        pooled = self.pool_norm(pooled)
        pooled = self.dropout_layer(pooled)

        logits = self.fc(pooled)

        return logits, channel_weights


if __name__ == "__main__":
    # Test without pretrained
    model = M3_CrissCrossBiLSTM(n_channels=16, time_steps=256, use_pretrained=False)
    x = torch.randn(8, 16, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with pretrained
    model_pt = M3_CrissCrossBiLSTM(n_channels=16, time_steps=256, use_pretrained=True)
    out_pt = model_pt(x)
    print(f"With pretrained - Output shape: {out_pt.shape}")
    print(f"With pretrained - Parameters: {sum(p.numel() for p in model_pt.parameters()):,}")
