"""
Criss-Cross Attention Module for LSTM Benchmark Models.

Separates spatial (channel) and temporal attention for EEG signals.
Inspired by CBraMod (ICLR 2025).

References:
    CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding (ICLR 2025)
    Criss-Cross Attention for Semantic Segmentation (ICCV 2019)
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpatialAttention(nn.Module):
    """
    Multi-head attention across EEG channels at each time step.

    Args:
        hidden_dim: Feature dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.

    Example:
        >>> attn = SpatialAttention(hidden_dim=64)
        >>> out = attn(torch.randn(2, 32, 16, 64))
        >>> assert out.shape == (2, 32, 16, 64)
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

    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_tensor: shape (batch, seq_len, n_channels, hidden_dim)

        Returns:
            Spatially attended tensor, same shape.
        """
        batch, seq_len, n_channels, hidden_dim = eeg_tensor.shape
        residual = eeg_tensor
        reshaped = eeg_tensor.view(batch * seq_len, n_channels, hidden_dim)
        attended = self._multihead_attend(reshaped, batch, seq_len, n_channels, hidden_dim)
        return self.norm(attended + residual)

    def _multihead_attend(
        self, reshaped: torch.Tensor, batch: int, seq_len: int, n_channels: int, hidden_dim: int
    ) -> torch.Tensor:
        """Run multi-head attention and reshape back to 4D."""
        q = self.q_proj(reshaped).view(batch * seq_len, n_channels, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(reshaped).view(batch * seq_len, n_channels, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(reshaped).view(batch * seq_len, n_channels, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = self.dropout(F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1))
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        out = self.out_proj(out.view(batch * seq_len, n_channels, hidden_dim))
        return out.view(batch, seq_len, n_channels, hidden_dim)


class TemporalAttention(nn.Module):
    """
    Multi-head attention across time steps for each EEG channel.

    Args:
        hidden_dim: Feature dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.

    Example:
        >>> attn = TemporalAttention(hidden_dim=64)
        >>> out = attn(torch.randn(2, 32, 16, 64))
        >>> assert out.shape == (2, 32, 16, 64)
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

    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_tensor: shape (batch, seq_len, n_channels, hidden_dim)

        Returns:
            Temporally attended tensor, same shape.
        """
        batch, seq_len, n_channels, hidden_dim = eeg_tensor.shape
        residual = eeg_tensor
        permuted = eeg_tensor.permute(0, 2, 1, 3).contiguous().view(batch * n_channels, seq_len, hidden_dim)
        attended = self._multihead_attend(permuted, batch, seq_len, n_channels, hidden_dim)
        return self.norm(attended + residual)

    def _multihead_attend(
        self, permuted: torch.Tensor, batch: int, seq_len: int, n_channels: int, hidden_dim: int
    ) -> torch.Tensor:
        """Run multi-head attention and reshape back to 4D."""
        q = self.q_proj(permuted).view(batch * n_channels, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(permuted).view(batch * n_channels, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(permuted).view(batch * n_channels, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn_weights = self.dropout(F.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1))
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        out = self.out_proj(out.view(batch * n_channels, seq_len, hidden_dim))
        out = out.view(batch, n_channels, seq_len, hidden_dim).permute(0, 2, 1, 3).contiguous()
        return out


class CrissCrossAttention(nn.Module):
    """
    Criss-Cross Attention: interleaved spatial and temporal attention.

    Args:
        hidden_dim: Internal feature dimension.
        num_heads: Number of attention heads.
        num_layers: Number of criss-cross layers.
        dropout: Dropout probability.
        n_channels: Number of EEG channels (for positional encoding).

    Example:
        >>> attn = CrissCrossAttention(hidden_dim=32, n_channels=16)
        >>> out = attn(torch.randn(2, 16, 256))
        >>> assert out.shape == (2, 16, 256)
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
        self.input_proj = nn.Linear(1, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, n_channels, hidden_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(1, 1024, 1, hidden_dim) * 0.02)
        self.layers = nn.ModuleList([
            _build_criss_cross_layer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_tensor: shape (batch, n_channels, time_steps)

        Returns:
            Attended tensor, same shape as input.
        """
        batch, n_channels, time_steps = eeg_tensor.shape
        hidden = self._embed(eeg_tensor, time_steps)
        hidden = self._apply_layers(hidden)
        return self._project_back(hidden)

    def _embed(self, eeg_tensor: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Project raw EEG to hidden dimension with positional encodings."""
        hidden = self.input_proj(eeg_tensor.permute(0, 2, 1).unsqueeze(-1))
        hidden = hidden + self.spatial_pos[:, :, :self.n_channels, :]
        return hidden + self.temporal_pos[:, :time_steps, :, :]

    def _apply_layers(self, hidden: torch.Tensor) -> torch.Tensor:
        """Pass through all criss-cross layers."""
        for layer in self.layers:
            hidden = layer["spatial"](hidden)
            hidden = layer["temporal"](hidden)
            residual = hidden
            hidden = layer["ffn_norm"](layer["ffn"](hidden) + residual)
        return hidden

    def _project_back(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project back to (batch, n_channels, time_steps)."""
        return self.output_proj(hidden).squeeze(-1).permute(0, 2, 1)


class CrissCrossBlock(nn.Module):
    """
    Single criss-cross attention block for insertion in LSTM architectures.

    Args:
        hidden_dim: Feature dimension.
        num_heads: Attention heads.
        dropout: Dropout probability.

    Example:
        >>> block = CrissCrossBlock(hidden_dim=128)
        >>> out = block(torch.randn(4, 256, 128))
        >>> assert out.shape == (4, 256, 128)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.ffn = _build_ffn(hidden_dim, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, eeg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_tensor: shape (batch, seq_len, hidden_dim)

        Returns:
            Attended tensor, same shape.
        """
        attended, _ = self.spatial_attn(eeg_tensor, eeg_tensor, eeg_tensor)
        hidden = self.spatial_norm(eeg_tensor + attended)
        attended, _ = self.temporal_attn(hidden, hidden, hidden)
        hidden = self.temporal_norm(hidden + attended)
        return self.ffn_norm(hidden + self.ffn(hidden))


def _build_criss_cross_layer(hidden_dim: int, num_heads: int, dropout: float) -> nn.ModuleDict:
    """Build one criss-cross layer dict (spatial + temporal + ffn)."""
    return nn.ModuleDict({
        "spatial": SpatialAttention(hidden_dim, num_heads, dropout),
        "temporal": TemporalAttention(hidden_dim, num_heads, dropout),
        "ffn": _build_ffn(hidden_dim, dropout),
        "ffn_norm": nn.LayerNorm(hidden_dim),
    })


def _build_ffn(hidden_dim: int, dropout: float) -> nn.Sequential:
    """Build feed-forward network block."""
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 4),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 4, hidden_dim),
        nn.Dropout(dropout),
    )
