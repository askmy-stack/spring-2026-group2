"""
Channel Attention Modules for LSTM Benchmark Models.

SE-style attention for weighting EEG electrode channels.

References:
    Squeeze-and-Excitation Networks (Hu et al., 2018)
    GMAEEG paper for EEG channel weighting
"""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ChannelAttention(nn.Module):
    """
    Lightweight channel-wise attention over EEG electrodes.

    Learns which of the N channels (electrodes) are most
    discriminative for seizure detection.

    Args:
        n_channels: Number of EEG channels.
        reduction: Bottleneck reduction factor for the gate MLP.

    Example:
        >>> attn = ChannelAttention(n_channels=16, reduction=4)
        >>> out = attn(torch.randn(8, 16, 256))
        >>> assert out.shape == (8, 16, 256)
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(n_channels // reduction, 2)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels, mid),
            nn.ReLU(),
            nn.Linear(mid, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.

        Args:
            eeg_tensor: EEG signal, shape (batch, n_channels, time_steps)

        Returns:
            Channel-scaled tensor, same shape as input.
        """
        channel_weights = self.gate(eeg_tensor).unsqueeze(-1)
        return eeg_tensor * channel_weights


class ChannelAttentionWithContext(nn.Module):
    """
    Enhanced channel attention using both average and max pooling.

    Args:
        n_channels: Number of EEG channels.
        reduction: Bottleneck reduction factor.

    Example:
        >>> attn = ChannelAttentionWithContext(n_channels=16)
        >>> out = attn(torch.randn(4, 16, 256))
        >>> assert out.shape == (4, 16, 256)
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(n_channels // reduction, 2)
        self.mlp = nn.Sequential(
            nn.Linear(n_channels, mid),
            nn.ReLU(),
            nn.Linear(mid, n_channels),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_tensor: shape (batch, n_channels, time_steps)

        Returns:
            Context-attended tensor, same shape as input.
        """
        batch, channels, _ = eeg_tensor.size()
        avg_features = self.mlp(self.avg_pool(eeg_tensor).view(batch, channels))
        max_features = self.mlp(self.max_pool(eeg_tensor).view(batch, channels))
        channel_weights = torch.sigmoid(avg_features + max_features).unsqueeze(-1)
        return eeg_tensor * channel_weights


class SpatialChannelAttention(nn.Module):
    """
    Combined spatial and channel attention (CBAM-style).

    First applies channel attention, then spatial (temporal) attention.

    Args:
        n_channels: Number of EEG channels.
        reduction: Channel attention reduction factor.
        kernel_size: Temporal attention conv kernel size.

    Example:
        >>> attn = SpatialChannelAttention(n_channels=16)
        >>> out = attn(torch.randn(4, 16, 256))
        >>> assert out.shape == (4, 16, 256)
    """

    def __init__(self, n_channels: int, reduction: int = 4, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttentionWithContext(n_channels, reduction)
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid(),
        )

    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_tensor: shape (batch, n_channels, time_steps)

        Returns:
            Spatially and channel-attended tensor, same shape as input.
        """
        channel_attended = self.channel_attn(eeg_tensor)
        avg_spatial = torch.mean(channel_attended, dim=1, keepdim=True)
        max_spatial, _ = torch.max(channel_attended, dim=1, keepdim=True)
        spatial_features = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_weights = self.spatial_conv(spatial_features)
        return channel_attended * spatial_weights
