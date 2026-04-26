"""
Vector Quantizer for discrete EEG representation learning.

Enables interpretable seizure patterns via codebook indices.

References:
    VQ-VAE: Neural Discrete Representation Learning (van den Oord et al., 2017)
    EEGFormer: Towards Transferable and Interpretable EEG Foundation Model
    LaBraM: Vector-quantized neural spectrum prediction
"""
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA-updated codebook.

    Maps continuous EEG features to nearest discrete codebook entries.

    Args:
        codebook_size: Number of discrete codes (default: 512).
        embedding_dim: Dimension of each code vector (default: 128).
        commitment_cost: Weight for commitment loss beta (default: 0.25).
        decay: EMA decay rate for codebook updates (default: 0.99).
        epsilon: Numerical stability constant (default: 1e-5).

    Example:
        >>> vq = VectorQuantizer(codebook_size=512, embedding_dim=128)
        >>> quantized, loss, indices = vq(torch.randn(4, 8, 128))
        >>> assert quantized.shape == (4, 8, 128)
    """

    def __init__(
        self,
        codebook_size: int = 512,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous features.

        Args:
            latents: Continuous features, shape (batch, seq_len, embedding_dim).

        Returns:
            quantized: Quantized features (same shape as input).
            loss: Commitment + codebook loss scalar.
            indices: Codebook indices, shape (batch, seq_len).
        """
        flat_latents = latents.view(-1, self.embedding_dim)
        indices = self._find_nearest_codes(flat_latents).view(latents.shape[:-1])
        quantized = self.embedding(indices)
        loss = self._compute_loss(latents, quantized, flat_latents)
        quantized = latents + (quantized - latents).detach()
        return quantized, loss, indices

    def _find_nearest_codes(self, flat_latents: torch.Tensor) -> torch.Tensor:
        """Return index of nearest codebook entry for each flat latent vector."""
        distances = (
            flat_latents.pow(2).sum(1, keepdim=True)
            + self.embedding.weight.pow(2).sum(1)
            - 2.0 * torch.matmul(flat_latents, self.embedding.weight.t())
        )
        return distances.argmin(dim=1)

    def _compute_loss(
        self, latents: torch.Tensor, quantized: torch.Tensor, flat_latents: torch.Tensor
    ) -> torch.Tensor:
        """Compute commitment + codebook loss; update EMA in training."""
        if self.training:
            self._ema_update(flat_latents, quantized)
            commitment_loss = self.commitment_cost * F.mse_loss(latents, quantized.detach())
            codebook_loss = F.mse_loss(quantized, latents.detach())
            return commitment_loss + codebook_loss
        return torch.tensor(0.0, device=latents.device)

    def _ema_update(self, flat_latents: torch.Tensor, quantized: torch.Tensor) -> None:
        """Update codebook via exponential moving average."""
        indices_flat = self._find_nearest_codes(flat_latents)
        encodings = F.one_hot(indices_flat, self.codebook_size).float()
        self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * encodings.sum(0)
        dw = torch.matmul(encodings.t(), flat_latents)
        self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw
        n = self.ema_cluster_size.sum()
        cluster_size = (self.ema_cluster_size + self.epsilon) / (n + self.codebook_size * self.epsilon) * n
        self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)

    def get_codebook_usage(self, indices: torch.Tensor) -> float:
        """Return percentage of codebook codes actively used in indices."""
        return float(indices.unique().numel() / self.codebook_size * 100)


class VQEncoder(nn.Module):
    """
    Vector-Quantized Encoder: EEG patches -> Transformer -> VQ -> discrete codes.

    Args:
        n_channels: Number of EEG channels (default: 16).
        patch_size: Temporal patch width (default: 32).
        embedding_dim: Embedding dimension (default: 128).
        codebook_size: Codebook size (default: 512).
        num_layers: Transformer encoder layers (default: 4).
        num_heads: Attention heads (default: 4).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> enc = VQEncoder(n_channels=16, patch_size=32, embedding_dim=128)
        >>> recon, loss, codes, z = enc(torch.randn(2, 16, 256))
    """

    def __init__(
        self,
        n_channels: int = 16,
        patch_size: int = 32,
        embedding_dim: int = 128,
        codebook_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.patch_embed = _build_patch_embed(n_channels, patch_size, embedding_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, embedding_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=embedding_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.quantizer = VectorQuantizer(codebook_size, embedding_dim)
        self.decoder = _build_decoder(embedding_dim, n_channels, patch_size)

    def forward(
        self, eeg_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            eeg_input: shape (batch, n_channels, time_steps)

        Returns:
            (reconstructed_patches, total_loss, codebook_indices, latents)
        """
        latents, original_patches = self._encode(eeg_input)
        quantized, vq_loss, indices = self.quantizer(latents)
        reconstructed = self.decoder(quantized)
        total_loss = F.mse_loss(reconstructed, original_patches) + vq_loss
        return reconstructed, total_loss, indices, latents

    def _encode(self, eeg_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract patches, embed, and pass through transformer."""
        batch, n_channels, time_steps = eeg_input.shape
        num_patches = time_steps // self.patch_size
        patches = eeg_input.unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(batch, num_patches, -1)
        latents = self.transformer(self.patch_embed(patches) + self.pos_embed[:, :num_patches, :])
        return latents, patches


def _build_patch_embed(n_channels: int, patch_size: int, embedding_dim: int) -> nn.Sequential:
    """Build patch embedding MLP."""
    return nn.Sequential(
        nn.Linear(n_channels * patch_size, embedding_dim * 2),
        nn.GELU(),
        nn.Linear(embedding_dim * 2, embedding_dim),
        nn.LayerNorm(embedding_dim),
    )


def _build_decoder(embedding_dim: int, n_channels: int, patch_size: int) -> nn.Sequential:
    """Build patch reconstruction decoder."""
    return nn.Sequential(
        nn.Linear(embedding_dim, embedding_dim * 2),
        nn.GELU(),
        nn.Linear(embedding_dim * 2, n_channels * patch_size),
    )
