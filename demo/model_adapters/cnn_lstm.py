from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _SEBlock1d(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(channels, reduction),
            nn.ReLU(),
            nn.Linear(reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.se(x).unsqueeze(-1)
        return x * weights


class _CheckpointConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        reduction = max(4, out_channels // 8)
        self.se = _SEBlock1d(out_channels, reduction=reduction)
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        x = self.conv(x)
        x = self.se(x)
        return x + residual


class CheckpointCNNLSTM(nn.Module):
    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.4,
    ):
        super().__init__()
        del time_steps
        self.branch_a = _CheckpointConvBlock(n_channels, 32, kernel_size=3, stride=2)
        self.branch_a2 = _CheckpointConvBlock(32, 32, kernel_size=3, stride=1)
        self.branch_b = _CheckpointConvBlock(n_channels, 64, kernel_size=15, stride=2)
        self.branch_b2 = _CheckpointConvBlock(64, 64, kernel_size=3, stride=1)
        self.branch_c = _CheckpointConvBlock(n_channels, 64, kernel_size=31, stride=1)
        self.branch_c2 = _CheckpointConvBlock(64, 64, kernel_size=3, stride=1)

        bilstm_out = hidden_size * 2
        self.lstm = nn.LSTM(
            input_size=160,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(bilstm_out, num_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(bilstm_out)
        self.pool_norm = nn.LayerNorm(bilstm_out * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(bilstm_out * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")

        a = self.branch_a2(self.branch_a(x))
        b = self.branch_b2(self.branch_b(x))
        c = self.branch_c2(self.branch_c(x))
        target_len = min(a.size(2), b.size(2))
        a = F.adaptive_avg_pool1d(a, target_len)
        b = F.adaptive_avg_pool1d(b, target_len)
        c = F.adaptive_avg_pool1d(c, target_len)
        features = torch.cat([a, b, c], dim=1)

        lstm_out, _ = self.lstm(features.permute(0, 2, 1))
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attended = self.attn_norm(attended + lstm_out)

        avg_pooled = attended.mean(dim=1)
        max_pooled = attended.max(dim=1).values
        pooled = torch.cat([avg_pooled, max_pooled], dim=-1)
        pooled = self.dropout(self.pool_norm(pooled))
        return self.fc(pooled)
