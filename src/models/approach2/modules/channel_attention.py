"""
Channel Attention Modules
-------------------------
SE-style attention for weighting EEG electrode channels.

References:
- Squeeze-and-Excitation Networks (Hu et al., 2018)
- GMAEEG paper for EEG channel weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Lightweight channel-wise attention over EEG electrodes.
    Learns which of the N channels (electrodes) are most
    discriminative for seizure detection.

    Input:  (batch, n_channels, time_steps)
    Output: (batch, n_channels, time_steps) — channel-scaled
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(n_channels // reduction, 2)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (batch, C, 1)
            nn.Flatten(),  # (batch, C)
            nn.Linear(n_channels, mid),
            nn.ReLU(),
            nn.Linear(mid, n_channels),
            nn.Sigmoid(),  # (batch, C) in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time_steps)
        weights = self.gate(x).unsqueeze(-1)  # (batch, channels, 1)
        return x * weights


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for CNN feature maps.
    Recalibrates channel-wise feature responses.

    Input:  (batch, channels, time_steps)
    Output: (batch, channels, time_steps)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time_steps)
        batch, channels, _ = x.size()

        # Squeeze: global average pooling
        y = self.squeeze(x).view(batch, channels)

        # Excitation: FC → ReLU → FC → Sigmoid
        y = self.excitation(y).view(batch, channels, 1)

        # Scale
        return x * y.expand_as(x)


class ChannelAttentionWithContext(nn.Module):
    """
    Enhanced channel attention that considers temporal context.
    Uses both average and max pooling for richer statistics.

    Input:  (batch, n_channels, time_steps)
    Output: (batch, n_channels, time_steps)
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(n_channels // reduction, 2)

        # Shared MLP for both pooling paths
        self.mlp = nn.Sequential(
            nn.Linear(n_channels, mid),
            nn.ReLU(),
            nn.Linear(mid, n_channels),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.size()

        # Average pooling path
        avg_out = self.avg_pool(x).view(batch, channels)
        avg_out = self.mlp(avg_out)

        # Max pooling path
        max_out = self.max_pool(x).view(batch, channels)
        max_out = self.mlp(max_out)

        # Combine and apply sigmoid
        weights = torch.sigmoid(avg_out + max_out).unsqueeze(-1)

        return x * weights


class SpatialChannelAttention(nn.Module):
    """
    Combined spatial and channel attention (CBAM-style).
    First applies channel attention, then spatial attention.

    Input:  (batch, n_channels, time_steps)
    Output: (batch, n_channels, time_steps)
    """

    def __init__(self, n_channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()

        # Channel attention
        self.channel_attn = ChannelAttentionWithContext(n_channels, reduction)

        # Spatial (temporal) attention
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        x = self.channel_attn(x)

        # Spatial attention: concat avg and max across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch, 1, time)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (batch, 1, time)
        spatial_in = torch.cat([avg_out, max_out], dim=1)  # (batch, 2, time)

        spatial_weights = self.spatial_conv(spatial_in)  # (batch, 1, time)

        return x * spatial_weights
