"""M7: VQ-Transformer (EEGFormer-style) — Ensemble Model. Expected F1: 0.80-0.87."""
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.vector_quantizer import VectorQuantizer, VQEncoder

logger = logging.getLogger(__name__)


class M7_VQTransformer(nn.Module):
    """
    Vector-Quantized Transformer for interpretable EEG seizure detection.

    Architecture:
        Input (batch, n_channels, time_steps)
        -> Patch Embedding -> Positional Encoding -> Transformer Encoder
        -> Vector Quantizer (discrete codes) -> Classify via CLS token

    Args:
        n_channels: Number of EEG channels (default: 16).
        time_steps: Timesteps per window (default: 256).
        patch_size: Temporal patch width (default: 32).
        hidden_size: Transformer/VQ embedding dimension (default: 128).
        codebook_size: VQ codebook size (default: 512).
        num_layers: Transformer encoder layers (default: 4).
        num_heads: Attention heads (default: 4).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> model = M7_VQTransformer(n_channels=16, time_steps=256)
        >>> logits = model(torch.randn(4, 16, 256))
        >>> assert logits.shape == (4, 1)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        patch_size: int = 32,
        hidden_size: int = 128,
        codebook_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_patches = time_steps // patch_size
        self.patch_embed = _build_patch_embed(n_channels, patch_size, hidden_size, dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.quantizer = VectorQuantizer(codebook_size, hidden_size)
        self.pre_quant = nn.Linear(hidden_size, hidden_size)
        self.post_quant = nn.Linear(hidden_size, hidden_size)
        self.classifier = _build_classifier(hidden_size, dropout)

    def forward(
        self, eeg_input: torch.Tensor, return_codes: bool = False
    ) -> torch.Tensor:
        """
        Args:
            eeg_input: EEG windows, shape (batch, n_channels, time_steps).
            return_codes: If True, return (logits, codes, vq_loss) instead.

        Returns:
            Logits, shape (batch, 1). Or tuple if return_codes=True.
        """
        batch = eeg_input.size(0)
        patches = self._embed_patches(eeg_input, batch)
        transformer_out = self.transformer(patches)
        cls_features, quantized, vq_loss, codes = self._quantize_classify(transformer_out)
        logits = self.classifier(quantized.mean(dim=1) + cls_features)
        if return_codes:
            return logits, codes, vq_loss
        return logits

    def _embed_patches(self, eeg_input: torch.Tensor, batch: int) -> torch.Tensor:
        """Extract patches, embed, prepend CLS token, add positional encoding."""
        patches = eeg_input.unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(batch, self.num_patches, -1)
        patch_embeds = self.patch_embed(patches)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat([cls_tokens, patch_embeds], dim=1)
        return sequence + self.pos_embed[:, :sequence.size(1), :]

    def _quantize_classify(
        self, transformer_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split CLS/patch tokens, quantize patches, return components."""
        cls_features = transformer_out[:, 0]
        patch_features = transformer_out[:, 1:]
        quantized, vq_loss, codes = self.quantizer(self.pre_quant(patch_features))
        return cls_features, self.post_quant(quantized), vq_loss, codes


def _build_patch_embed(
    n_channels: int, patch_size: int, hidden_size: int, dropout: float
) -> nn.Sequential:
    """Build patch embedding MLP with layer norm."""
    return nn.Sequential(
        nn.Linear(n_channels * patch_size, hidden_size * 2),
        nn.GELU(),
        nn.Dropout(dropout * 0.5),
        nn.Linear(hidden_size * 2, hidden_size),
        nn.LayerNorm(hidden_size),
    )


def _build_classifier(hidden_size: int, dropout: float) -> nn.Sequential:
    """Build classification head with layer norm."""
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, 1),
    )
