"""M7: Attention-based BiLSTM — Benchmark Model. Expected F1: 0.72–0.80.

Design: BiLSTM backbone followed by multi-head temporal self-attention and
learnable (Bahdanau-style) attention pooling. This gives every timestep
access to a global temporal context (beyond what BiLSTM alone captures) and
lets the model emphasise the most discriminative moments rather than
treating every timestep equally as avg/max-pool does in m1/m2.

Input:  (batch, n_channels, time_steps)
Output: (batch, 1) logits.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn

from ..modules.channel_attention import ChannelAttention
from .m1_vanilla_lstm import _build_input_proj, _build_lstm, _build_classifier

logger = logging.getLogger(__name__)


class M7_AttentionLSTM(nn.Module):
    """
    Attention-based BiLSTM for EEG seizure detection.

    Architecture::

        Input (B, n_channels, time_steps)
            -> ChannelAttention -> LayerNorm -> Projection
            -> BiLSTM (bidirectional, num_layers)
            -> + learnable positional embedding
            -> MultiHead TemporalSelfAttention (2 blocks w/ residual + LN)
            -> Additive (Bahdanau) attention pooling over time
            -> FC head
            -> Logits

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Timesteps per window (default: 256).
        hidden_size: Hidden dim per LSTM direction (default: 128).
        num_layers: BiLSTM layers (default: 2).
        num_heads: Self-attention heads (default: 4).
        num_attn_blocks: Stacked self-attention blocks (default: 2).
        dropout: Dropout rate (default: 0.3).

    Example:
        >>> model = M7_AttentionLSTM(n_channels=16)
        >>> logits = model(torch.randn(8, 16, 256))
        >>> assert logits.shape == (8, 1)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_attn_blocks: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        bilstm_out = hidden_size * 2  # bidirectional

        self.channel_attn = ChannelAttention(n_channels, reduction=4)
        self.input_norm = nn.LayerNorm(n_channels)
        self.input_proj = _build_input_proj(n_channels, hidden_size, dropout)
        self.lstm = _build_lstm(hidden_size, num_layers, dropout, bidirectional=True)

        self.pos_embedding = _build_pos_embedding(time_steps, bilstm_out)
        self.self_attn_blocks = nn.ModuleList([
            _SelfAttentionBlock(bilstm_out, num_heads, dropout)
            for _ in range(num_attn_blocks)
        ])
        self.attn_pool = _AdditiveAttentionPool(bilstm_out)

        self.pool_norm = nn.LayerNorm(bilstm_out)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = _build_classifier(hidden_size, bilstm_out, dropout)

    def forward(self, eeg_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            eeg_input: shape (batch, n_channels, time_steps).

        Returns:
            Logits, shape (batch, 1).
        """
        if eeg_input.ndim != 3:
            raise ValueError(
                f"M7_AttentionLSTM expects (batch, channels, time), "
                f"got shape {tuple(eeg_input.shape)}"
            )
        attended = self.channel_attn(eeg_input)                          # (B, C, T)
        projected = self.input_proj(self.input_norm(attended.permute(0, 2, 1)))  # (B, T, H)
        lstm_out, _ = self.lstm(projected)                               # (B, T, 2H)
        positioned = self._add_positional(lstm_out)
        x = positioned
        for block in self.self_attn_blocks:
            x = block(x)
        pooled = self.attn_pool(x)                                       # (B, 2H)
        pooled = self.dropout_layer(self.pool_norm(pooled))
        return self.classifier(pooled)

    def _add_positional(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Add learnable positional embedding, truncated to actual seq len."""
        seq_len = lstm_out.size(1)
        return lstm_out + self.pos_embedding[:, :seq_len, :]


class _SelfAttentionBlock(nn.Module):
    """Pre-norm multi-head self-attention block with residual + FFN."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class _AdditiveAttentionPool(nn.Module):
    """Bahdanau-style attention pooling over the temporal axis."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)  -> weights: (B, T, 1) -> pooled: (B, D)
        scores = self.score(x)
        weights = torch.softmax(scores, dim=1)
        return (weights * x).sum(dim=1)


def _build_pos_embedding(time_steps: int, embed_dim: int) -> nn.Parameter:
    """Truncated-normal learnable positional embedding."""
    param = nn.Parameter(torch.zeros(1, time_steps, embed_dim))
    nn.init.trunc_normal_(param, std=0.02)
    return param
