"""
M7: VQ-Transformer (EEGFormer-style)
------------------------------------
Vector-quantized transformer for discrete, interpretable EEG encoding.

Expected F1: 0.80-0.87 (with pre-training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from modules.vector_quantizer import VectorQuantizer, VQEncoder
from modules.pretrained_encoders import load_pretrained_encoder


class M7_VQTransformer(nn.Module):
    """
    VQ-Transformer for EEG.

    Encodes EEG into discrete codes using vector quantization,
    enabling interpretable seizure pattern analysis.

    Architecture:
        Input (batch, 16, 256)
        → [Optional: Pretrained Encoder]
        → Patch Embedding
        → Transformer Encoder
        → Vector Quantizer (discrete codes)
        → Classification Head
        → Logits

    Args:
        n_channels: Number of EEG channels (default: 16)
        time_steps: Number of time steps (default: 256)
        patch_size: Size of each temporal patch (default: 32)
        hidden_size: Hidden size (default: 128)
        codebook_size: Number of discrete codes (default: 512)
        num_layers: Number of transformer layers (default: 4)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
        use_pretrained: Whether to use pretrained encoder
        pretrained_encoder: Name of pretrained encoder
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
        use_pretrained: bool = False,
        pretrained_encoder: Optional[str] = "cbramod",
    ):
        super().__init__()
        self.n_channels = n_channels
        self.time_steps = time_steps
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.codebook_size = codebook_size
        self.use_pretrained = use_pretrained

        # Number of patches
        self.num_patches = time_steps // patch_size

        # Optional pretrained encoder
        if use_pretrained and pretrained_encoder:
            self.pretrained = load_pretrained_encoder(
                pretrained_encoder,
                n_channels=n_channels,
                time_steps=time_steps,
                hidden_dim=hidden_size,
                pretrained=True,
                freeze=True,
            )
        else:
            self.pretrained = None

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(n_channels * patch_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, hidden_size) * 0.02
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Vector quantizer
        self.quantizer = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=hidden_size,
            commitment_cost=0.25,
        )

        # Pre-quantization projection
        self.pre_quant = nn.Linear(hidden_size, hidden_size)

        # Post-quantization projection
        self.post_quant = nn.Linear(hidden_size, hidden_size)

        # Classification head
        fc_input_size = hidden_size
        if use_pretrained:
            fc_input_size += hidden_size

        self.fc = nn.Sequential(
            nn.LayerNorm(fc_input_size),
            nn.Linear(fc_input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # For reconstruction (optional, for pre-training)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, n_channels * patch_size),
        )

    def forward(
        self, x: torch.Tensor, return_codes: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: EEG signal (batch, n_channels, time_steps)
            return_codes: Whether to return discrete codes

        Returns:
            logits: Classification logits (batch, 1)
            [Optional] codes: Discrete codes (batch, num_patches)
            [Optional] vq_loss: Vector quantization loss
        """
        batch = x.shape[0]

        # Get pretrained features if available
        pretrained_feat = None
        if self.pretrained is not None:
            pretrained_feat = self.pretrained(x)

        # Create patches: (batch, num_patches, n_channels * patch_size)
        patches = x.unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(batch, self.num_patches, -1)

        # Embed patches
        x = self.patch_embed(patches)  # (batch, num_patches, hidden_size)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, hidden_size)

        # Add positional encoding
        x = x + self.pos_embed[:, :x.size(1), :]

        # Transformer
        x = self.transformer(x)  # (batch, num_patches + 1, hidden_size)

        # Separate CLS and patch tokens
        cls_out = x[:, 0]  # (batch, hidden_size)
        patch_out = x[:, 1:]  # (batch, num_patches, hidden_size)

        # Vector quantization on patch tokens
        patch_out = self.pre_quant(patch_out)
        quantized, vq_loss, codes = self.quantizer(patch_out)
        quantized = self.post_quant(quantized)

        # Pool quantized features
        pooled = quantized.mean(dim=1)  # (batch, hidden_size)

        # Combine with CLS token
        combined = pooled + cls_out

        # Combine with pretrained features
        if pretrained_feat is not None:
            combined = torch.cat([combined, pretrained_feat], dim=1)

        # Classify
        logits = self.fc(combined)

        if return_codes:
            return logits, codes, vq_loss
        return logits

    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Get discrete codes for interpretability."""
        _, codes, _ = self.forward(x, return_codes=True)
        return codes

    def reconstruct(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct EEG from discrete codes (for pre-training)."""
        batch = x.shape[0]

        # Create and embed patches
        patches = x.unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        original_patches = patches.view(batch, self.num_patches, -1)

        x = self.patch_embed(original_patches)

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]

        x = self.transformer(x)

        patch_out = x[:, 1:]
        patch_out = self.pre_quant(patch_out)
        quantized, vq_loss, codes = self.quantizer(patch_out)
        quantized = self.post_quant(quantized)

        # Decode
        reconstructed = self.decoder(quantized)

        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, original_patches)

        total_loss = recon_loss + vq_loss

        return reconstructed, total_loss

    def get_codebook_usage(self, x: torch.Tensor) -> float:
        """Get percentage of codebook being used."""
        codes = self.get_codes(x)
        return self.quantizer.get_codebook_usage(codes)


if __name__ == "__main__":
    # Test
    model = M7_VQTransformer(n_channels=16, time_steps=256, use_pretrained=False)
    x = torch.randn(8, 16, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with codes
    out, codes, vq_loss = model(x, return_codes=True)
    print(f"Codes shape: {codes.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")

    # Test reconstruction
    recon, loss = model.reconstruct(x)
    print(f"Reconstruction loss: {loss.item():.4f}")
