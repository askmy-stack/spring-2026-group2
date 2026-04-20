"""
Vector Quantizer Module
-----------------------
Discrete representation learning for EEG signals.
Inspired by EEGFormer and VQ-VAE for interpretable seizure patterns.

Key insight: Discrete codes allow:
1. Interpretable seizure patterns (n-gram analysis)
2. Compression for efficient storage
3. Better generalization through discrete bottleneck

References:
- EEGFormer: Towards Transferable and Interpretable EEG Foundation Model
- VQ-VAE: Neural Discrete Representation Learning (van den Oord et al., 2017)
- LaBraM: Vector-quantized neural spectrum prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer with EMA updates.
    Maps continuous EEG features to discrete codebook entries.

    Args:
        codebook_size: Number of discrete codes (e.g., 512, 1024, 2048)
        embedding_dim: Dimension of each code vector
        commitment_cost: Weight for commitment loss (beta)
        decay: EMA decay rate for codebook updates
        epsilon: Small constant for numerical stability
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

        # Codebook embeddings
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

        # EMA variables
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous features to discrete codes.

        Args:
            z: Continuous features (batch, seq_len, embedding_dim)

        Returns:
            quantized: Quantized features (same shape as z)
            loss: Commitment + codebook loss
            indices: Codebook indices (batch, seq_len)
        """
        # Flatten input
        z_flat = z.view(-1, self.embedding_dim)

        # Compute distances to codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z.e
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )

        # Get nearest codebook entry
        indices = torch.argmin(distances, dim=1)
        indices = indices.view(z.shape[:-1])

        # Quantize
        quantized = self.embedding(indices)

        # Loss
        if self.training:
            # EMA update
            encodings = F.one_hot(indices.view(-1), self.codebook_size).float()
            
            # Update cluster sizes
            self.ema_cluster_size = (
                self.decay * self.ema_cluster_size
                + (1 - self.decay) * encodings.sum(0)
            )

            # Update embeddings
            dw = torch.matmul(encodings.t(), z_flat)
            self.ema_w = self.decay * self.ema_w + (1 - self.decay) * dw

            # Normalize
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.codebook_size * self.epsilon)
                * n
            )
            self.embedding.weight.data = self.ema_w / cluster_size.unsqueeze(1)

            # Commitment loss: encourages encoder to commit to codebook
            commitment_loss = self.commitment_cost * F.mse_loss(z, quantized.detach())

            # Codebook loss: moves codebook towards encoder outputs (handled by EMA)
            codebook_loss = F.mse_loss(quantized, z.detach())

            loss = commitment_loss + codebook_loss
        else:
            loss = torch.tensor(0.0, device=z.device)

        # Straight-through estimator: copy gradients from quantized to z
        quantized = z + (quantized - z).detach()

        return quantized, loss, indices

    def get_codebook_usage(self, indices: torch.Tensor) -> float:
        """Compute percentage of codebook being used."""
        unique = torch.unique(indices)
        return len(unique) / self.codebook_size * 100


class VQEncoder(nn.Module):
    """
    Vector-Quantized Encoder for EEG signals.
    Encodes EEG patches into discrete tokens.

    Architecture:
        EEG → Patch Embedding → Transformer → VQ → Discrete Codes

    Args:
        n_channels: Number of EEG channels
        patch_size: Size of each temporal patch
        embedding_dim: Dimension of embeddings
        codebook_size: Number of discrete codes
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
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

        # Patch embedding: project each patch to embedding_dim
        self.patch_embed = nn.Sequential(
            nn.Linear(n_channels * patch_size, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, embedding_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Vector quantizer
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=embedding_dim,
        )

        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, n_channels * patch_size),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode EEG to continuous features.

        Args:
            x: EEG signal (batch, n_channels, time_steps)

        Returns:
            z: Continuous features (batch, num_patches, embedding_dim)
            patches: Original patches for reconstruction loss
        """
        batch, n_channels, time_steps = x.shape
        num_patches = time_steps // self.patch_size

        # Create patches: (batch, num_patches, n_channels * patch_size)
        patches = x.unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(batch, num_patches, -1)

        # Embed patches
        z = self.patch_embed(patches)

        # Add positional encoding
        z = z + self.pos_embed[:, :num_patches, :]

        # Transformer
        z = self.transformer(z)

        return z, patches

    def quantize(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous features to discrete codes.

        Args:
            z: Continuous features (batch, num_patches, embedding_dim)

        Returns:
            quantized: Quantized features
            vq_loss: Vector quantization loss
            indices: Codebook indices
        """
        return self.quantizer(z)

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized features back to EEG patches.

        Args:
            quantized: Quantized features (batch, num_patches, embedding_dim)

        Returns:
            reconstructed: Reconstructed patches (batch, num_patches, n_channels * patch_size)
        """
        return self.decoder(quantized)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode, quantize, decode.

        Args:
            x: EEG signal (batch, n_channels, time_steps)

        Returns:
            reconstructed: Reconstructed patches
            vq_loss: Vector quantization loss
            indices: Codebook indices
            z: Continuous features (before quantization)
        """
        # Encode
        z, original_patches = self.encode(x)

        # Quantize
        quantized, vq_loss, indices = self.quantize(z)

        # Decode
        reconstructed = self.decode(quantized)

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, original_patches)

        # Total loss
        total_loss = recon_loss + vq_loss

        return reconstructed, total_loss, indices, z

    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get discrete codes for EEG signal (for inference).

        Args:
            x: EEG signal (batch, n_channels, time_steps)

        Returns:
            indices: Codebook indices (batch, num_patches)
        """
        z, _ = self.encode(x)
        _, _, indices = self.quantize(z)
        return indices

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get quantized features for downstream tasks.

        Args:
            x: EEG signal (batch, n_channels, time_steps)

        Returns:
            features: Quantized features (batch, num_patches, embedding_dim)
        """
        z, _ = self.encode(x)
        quantized, _, _ = self.quantize(z)
        return quantized


class VQSeizureClassifier(nn.Module):
    """
    VQ-Transformer for seizure classification.
    Uses discrete codes + transformer for interpretable detection.

    Architecture:
        EEG → VQEncoder → Discrete Codes → Classifier
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        patch_size: int = 32,
        embedding_dim: int = 128,
        codebook_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vq_encoder = VQEncoder(
            n_channels=n_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            codebook_size=codebook_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1),
        )

        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for seizure classification.

        Args:
            x: EEG signal (batch, n_channels, time_steps)

        Returns:
            logits: Classification logits (batch, 1)
            vq_loss: Vector quantization loss
            indices: Codebook indices for interpretability
        """
        # Get quantized features
        _, vq_loss, indices, _ = self.vq_encoder(x)
        features = self.vq_encoder.get_features(x)

        # Pool over patches
        features = features.permute(0, 2, 1)  # (batch, embedding_dim, num_patches)
        pooled = self.pool(features).squeeze(-1)  # (batch, embedding_dim)

        # Classify
        logits = self.classifier(pooled)

        return logits, vq_loss, indices
