"""
Criss-Cross Attention Module
----------------------------
Separates spatial (channel) and temporal attention for EEG signals.
Inspired by CBraMod (ICLR 2025) - captures heterogeneous dependencies.

Key insight: EEG has different spatial vs temporal dynamics
- Spatial: seizure propagation across brain regions
- Temporal: ictal progression over time

References:
- CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding (ICLR 2025)
- Criss-Cross Attention for Semantic Segmentation (ICCV 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """
    Attention across EEG channels (electrodes) at each time step.
    Models how seizure activity spreads across brain regions.

    Input:  (batch, seq_len, n_channels, hidden_dim)
    Output: (batch, seq_len, n_channels, hidden_dim)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_channels, hidden_dim)
        batch, seq_len, n_channels, hidden_dim = x.shape
        residual = x

        # Reshape for attention across channels
        x = x.view(batch * seq_len, n_channels, hidden_dim)

        # Project to Q, K, V
        q = self.q_proj(x).view(batch * seq_len, n_channels, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch * seq_len, n_channels, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch * seq_len, n_channels, self.num_heads, self.head_dim)

        # Transpose for attention: (batch*seq, heads, channels, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch * seq_len, n_channels, hidden_dim)
        out = self.out_proj(out)
        out = out.view(batch, seq_len, n_channels, hidden_dim)

        # Residual + LayerNorm
        return self.norm(out + residual)


class TemporalAttention(nn.Module):
    """
    Attention across time steps for each channel independently.
    Models how seizure evolves over time within each electrode.

    Input:  (batch, seq_len, n_channels, hidden_dim)
    Output: (batch, seq_len, n_channels, hidden_dim)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_channels, hidden_dim)
        batch, seq_len, n_channels, hidden_dim = x.shape
        residual = x

        # Reshape for attention across time
        x = x.permute(0, 2, 1, 3).contiguous()  # (batch, n_channels, seq_len, hidden_dim)
        x = x.view(batch * n_channels, seq_len, hidden_dim)

        # Project to Q, K, V
        q = self.q_proj(x).view(batch * n_channels, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch * n_channels, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch * n_channels, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch * n_channels, seq_len, hidden_dim)
        out = self.out_proj(out)
        out = out.view(batch, n_channels, seq_len, hidden_dim)
        out = out.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, n_channels, hidden_dim)

        # Residual + LayerNorm
        return self.norm(out + residual)


class CrissCrossAttention(nn.Module):
    """
    Criss-Cross Attention: Parallel spatial and temporal attention.

    Processes spatial (cross-channel) and temporal dependencies separately,
    then combines them. This is more efficient and effective than full
    attention for EEG signals.

    Architecture:
        Input → [Spatial Attn] → [Temporal Attn] → Output
                      ↓                ↓
                 (channels)        (time steps)

    Input:  (batch, n_channels, time_steps) or (batch, seq_len, hidden_dim)
    Output: (batch, n_channels, time_steps) or (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_channels: int = 16,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_channels = n_channels

        # Input projection
        self.input_proj = nn.Linear(1, hidden_dim)

        # Stack of criss-cross layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    "spatial": SpatialAttention(hidden_dim, num_heads, dropout),
                    "temporal": TemporalAttention(hidden_dim, num_heads, dropout),
                    "ffn": nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 4),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim * 4, hidden_dim),
                        nn.Dropout(dropout),
                    ),
                    "ffn_norm": nn.LayerNorm(hidden_dim),
                })
            )

        # Learnable positional encodings
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, n_channels, hidden_dim) * 0.02)
        self.temporal_pos = None  # Will be created dynamically

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG signal (batch, n_channels, time_steps)

        Returns:
            Attended features (batch, n_channels, time_steps)
        """
        batch, n_channels, time_steps = x.shape

        # Reshape: (batch, n_channels, time_steps) → (batch, time_steps, n_channels, 1)
        x = x.permute(0, 2, 1).unsqueeze(-1)

        # Project to hidden dim: (batch, time_steps, n_channels, hidden_dim)
        x = self.input_proj(x)

        # Add positional encodings
        x = x + self.spatial_pos[:, :, :n_channels, :]

        # Create temporal positional encoding if needed
        if self.temporal_pos is None or self.temporal_pos.size(1) != time_steps:
            self.temporal_pos = nn.Parameter(
                torch.randn(1, time_steps, 1, self.hidden_dim, device=x.device) * 0.02
            )
        x = x + self.temporal_pos[:, :time_steps, :, :]

        # Apply criss-cross layers
        for layer in self.layers:
            # Spatial attention (across channels)
            x = layer["spatial"](x)

            # Temporal attention (across time)
            x = layer["temporal"](x)

            # Feed-forward network
            residual = x
            x = layer["ffn"](x)
            x = layer["ffn_norm"](x + residual)

        # Project back: (batch, time_steps, n_channels, 1)
        x = self.output_proj(x)

        # Reshape: (batch, n_channels, time_steps)
        x = x.squeeze(-1).permute(0, 2, 1)

        return x


class CrissCrossBlock(nn.Module):
    """
    Single criss-cross attention block for use in larger architectures.
    Can be inserted after LSTM or CNN layers.

    Input:  (batch, seq_len, hidden_dim)
    Output: (batch, seq_len, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-head attention (acts as spatial attention when n_channels is in seq_len)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_norm = nn.LayerNorm(hidden_dim)

        # Temporal attention with causal mask option
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial attention
        attended, _ = self.spatial_attn(x, x, x)
        x = self.spatial_norm(x + attended)

        # Temporal attention
        attended, _ = self.temporal_attn(x, x, x)
        x = self.temporal_norm(x + attended)

        # FFN
        x = self.ffn_norm(x + self.ffn(x))

        return x
