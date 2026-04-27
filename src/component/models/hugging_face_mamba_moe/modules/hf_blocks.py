"""
Hugging Face CNN Building Blocks for EEG.

Reusable Conv1D primitives used by hf_cnn_models.py architectures.
Ported from ritu_dev branch: src/hugging_face/model/blocks.py.
"""
from __future__ import annotations

import logging
from typing import Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ConvBNAct1d(nn.Sequential):
    """
    Conv1d → BatchNorm1d → ReLU (→ optional Dropout).

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride (default: 1).
        padding: Explicit padding; defaults to kernel_size // 2.
        dropout: Dropout probability appended after activation (default: 0.0).

    Example:
        >>> block = ConvBNAct1d(16, 32, kernel_size=7, dropout=0.3)
        >>> out = block(torch.randn(2, 16, 256))
        >>> assert out.shape == (2, 32, 256)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        dropout: float = 0.0,
    ):
        if padding is None:
            padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        super().__init__(*layers)


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise-separable Conv1d: depthwise + pointwise with BN + ReLU.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Depthwise kernel size.
        dropout: Dropout probability (default: 0.0).

    Example:
        >>> ds = DepthwiseSeparableConv1d(32, 64, kernel_size=3)
        >>> out = ds(torch.randn(2, 32, 256))
        >>> assert out.shape == (2, 64, 256)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels), nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, C, T). Returns: (batch, out_channels, T)."""
        return self.block(x)


class ResidualBlock1d(nn.Module):
    """
    1D residual block with two conv layers and skip connection.

    Args:
        channels: Number of channels (input = output).
        kernel_size: Convolution kernel size (default: 3).
        dropout: Dropout probability (default: 0.0).

    Example:
        >>> blk = ResidualBlock1d(64, kernel_size=5)
        >>> out = blk(torch.randn(2, 64, 128))
        >>> assert out.shape == (2, 64, 128)
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, channels, T). Returns: residual output, same shape."""
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class DilatedResidualBlock1d(nn.Module):
    """
    Dilated 1D residual block for larger receptive fields.

    Args:
        channels: Number of channels.
        kernel_size: Convolution kernel size (default: 3).
        dilation: Dilation factor (default: 2).
        dropout: Dropout probability (default: 0.0).

    Example:
        >>> blk = DilatedResidualBlock1d(64, kernel_size=3, dilation=4)
        >>> out = blk(torch.randn(2, 64, 128))
        >>> assert out.shape == (2, 64, 128)
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 2, dropout: float = 0.0):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, channels, T). Returns: dilated residual output, same shape."""
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class SEBlock1d(nn.Module):
    """
    Squeeze-and-Excitation channel attention for 1D signals.

    Args:
        channels: Number of feature channels.
        reduction: Channel reduction ratio (default: 8).

    Example:
        >>> se = SEBlock1d(128, reduction=8)
        >>> out = se(torch.randn(2, 128, 64))
        >>> assert out.shape == (2, 128, 64)
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False), nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, channels, T). Returns: channel-recalibrated tensor."""
        batch, channels, _ = x.shape
        weights = self.fc(self.pool(x).view(batch, channels)).view(batch, channels, 1)
        return x * weights


class MultiScaleBranch1d(nn.Module):
    """
    Two-conv branch for one kernel scale in a multi-scale CNN.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size for both convs.
        dropout: Dropout probability (default: 0.0).

    Example:
        >>> branch = MultiScaleBranch1d(16, 32, kernel_size=7)
        >>> out = branch(torch.randn(2, 16, 256))
        >>> assert out.shape == (2, 32, 256)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        self.branch = nn.Sequential(
            ConvBNAct1d(in_channels, out_channels, kernel_size, dropout=dropout),
            ConvBNAct1d(out_channels, out_channels, kernel_size, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, in_channels, T). Returns: (batch, out_channels, T)."""
        return self.branch(x)


def ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """
    Assert input is 3D; raise ValueError otherwise.

    Args:
        x: Input tensor.

    Returns:
        Unchanged tensor if shape is (batch, channels, time).

    Raises:
        ValueError: If tensor is not 3D.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected input [batch, channels, time], got {tuple(x.shape)}")
    return x


def make_mlp(head_dims: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    """
    Build a multi-layer perceptron from a dimension sequence.

    Args:
        head_dims: Sequence of layer widths, e.g. (128, 64, 2).
        dropout: Dropout probability between hidden layers (default: 0.0).

    Returns:
        nn.Sequential MLP.

    Example:
        >>> mlp = make_mlp((128, 64, 2), dropout=0.3)
        >>> out = mlp(torch.randn(4, 128))
        >>> assert out.shape == (4, 2)
    """
    layers: list[nn.Module] = []
    for i in range(len(head_dims) - 1):
        layers.append(nn.Linear(head_dims[i], head_dims[i + 1]))
        if i < len(head_dims) - 2:
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
