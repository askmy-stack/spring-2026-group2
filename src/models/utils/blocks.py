from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ConvBNAct1d(nn.Sequential):
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock1d(nn.Module):
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
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class DilatedResidualBlock1d(nn.Module):
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
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.shape
        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1)
        return x * weights


class MultiScaleBranch1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float = 0.0):
        super().__init__()
        self.branch = nn.Sequential(
            ConvBNAct1d(in_channels, out_channels, kernel_size, dropout=dropout),
            ConvBNAct1d(out_channels, out_channels, kernel_size, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


def ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected input with shape [batch, channels, time], got {tuple(x.shape)}")
    return x


def make_mlp(head_dims: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(head_dims) - 1):
        layers.append(nn.Linear(head_dims[i], head_dims[i + 1]))
        if i < len(head_dims) - 2:
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
