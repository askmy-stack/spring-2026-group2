"""
Hugging Face CNN Architectures for EEG seizure detection.

Ported from ritu_dev branch: src/hugging_face/model/architectures.py.
All sys.path hacks removed; imports are relative.

Models:
    BaselineCNN1D           — Simple 3-layer 1D CNN baseline.
    EnhancedCNN1D           — Stem + dilated residual + SE attention CNN.
    EEGNetLocal             — Local EEGNet implementation (no HF dependency).
    EEGNetHFPretrained      — EEGNet weights from HuggingFace Hub.
    DeepConvNet             — EEGNet-style 2D conv on (1, C, T) input.
    MultiScaleCNN           — Parallel multi-scale branches + fusion.
    MultiScaleAttentionCNN  — MultiScaleCNN with SE attention in fusion.
    STEEGFormerPretrained   — ST-EEGFormer loaded from eugenehp/ST-EEGFormer.
"""
from __future__ import annotations

import logging
import math
from typing import Tuple

import torch
from torch import nn

from ..modules.hf_blocks import (
    ConvBNAct1d,
    DepthwiseSeparableConv1d,
    DilatedResidualBlock1d,
    MultiScaleBranch1d,
    ResidualBlock1d,
    SEBlock1d,
    ensure_3d,
    make_mlp,
)

logger = logging.getLogger(__name__)


class BaselineCNN1D(nn.Module):
    """
    Three-layer 1D CNN baseline for EEG classification.

    Args:
        in_channels: EEG channels (default: 16).
        num_classes: Output classes (default: 2).
        hidden_dim: FC hidden dimension (default: 64).
        dropout: Dropout probability (default: 0.3).

    Example:
        >>> model = BaselineCNN1D(in_channels=16, num_classes=2)
        >>> out = model(torch.randn(4, 16, 256))
        >>> assert out.shape == (4, 2)
    """

    def __init__(self, in_channels: int = 16, num_classes: int = 2,
                 hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct1d(in_channels, 32, kernel_size=7, dropout=dropout),
            nn.MaxPool1d(kernel_size=2),
            ConvBNAct1d(32, 64, kernel_size=5, dropout=dropout),
            nn.MaxPool1d(kernel_size=2),
            ConvBNAct1d(64, 128, kernel_size=3, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = make_mlp((128, hidden_dim, num_classes), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, C, T). Returns: logits (batch, num_classes)."""
        return self.classifier(self.features(ensure_3d(x)).flatten(1))


class EnhancedCNN1D(nn.Module):
    """
    Enhanced 1D CNN with dilated residual blocks and SE channel attention.

    Args:
        in_channels: EEG channels (default: 16).
        num_classes: Output classes (default: 2).
        hidden_dim: FC hidden dimension (default: 64).
        dropout: Dropout probability (default: 0.35).

    Example:
        >>> model = EnhancedCNN1D(in_channels=16)
        >>> out = model(torch.randn(4, 16, 256))
        >>> assert out.shape == (4, 2)
    """

    def __init__(self, in_channels: int = 16, num_classes: int = 2,
                 hidden_dim: int = 64, dropout: float = 0.35):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct1d(in_channels, 32, kernel_size=15, dropout=dropout),
            nn.MaxPool1d(kernel_size=2),
            ConvBNAct1d(32, 64, kernel_size=9, dropout=dropout),
        )
        self.encoder = nn.Sequential(
            ResidualBlock1d(64, kernel_size=5, dropout=dropout),
            DilatedResidualBlock1d(64, kernel_size=3, dilation=2, dropout=dropout),
            nn.MaxPool1d(kernel_size=2),
            ConvBNAct1d(64, 128, kernel_size=5, dropout=dropout),
            DilatedResidualBlock1d(128, kernel_size=3, dilation=4, dropout=dropout),
            SEBlock1d(128, reduction=8),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = make_mlp((128, hidden_dim, num_classes), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, C, T). Returns: logits (batch, num_classes)."""
        x = ensure_3d(x)
        return self.classifier(self.encoder(self.stem(x)).flatten(1))


class EEGNetLocal(nn.Module):
    """
    Local EEGNet implementation with no external dependency.

    Args:
        in_channels: EEG channels (default: 16).
        num_classes: Output classes (default: 2).
        temporal_filters: Temporal filter count (default: 8).
        depth_multiplier: Spatial filter multiplier (default: 2).
        separable_filters: Separable conv output filters (default: 32).
        dropout: Dropout probability (default: 0.25).

    Example:
        >>> model = EEGNetLocal(in_channels=16)
        >>> out = model(torch.randn(4, 16, 256))
        >>> assert out.shape == (4, 2)
    """

    def __init__(self, in_channels: int = 16, num_classes: int = 2,
                 temporal_filters: int = 8, depth_multiplier: int = 2,
                 separable_filters: int = 32, dropout: float = 0.25):
        super().__init__()
        spatial_filters = temporal_filters * depth_multiplier
        self.temporal = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(temporal_filters),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(temporal_filters, spatial_filters, kernel_size=(in_channels, 1),
                      groups=temporal_filters, bias=False),
            nn.BatchNorm2d(spatial_filters), nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)), nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=(1, 16), padding=(0, 8),
                      groups=spatial_filters, bias=False),
            nn.Conv2d(spatial_filters, separable_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(separable_filters), nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 8)), nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(separable_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, C, T). Returns: logits (batch, num_classes)."""
        x = ensure_3d(x).unsqueeze(1)
        return self.classifier(self.separable(self.spatial(self.temporal(x))).flatten(1))


class EEGNetHFPretrained(nn.Module):
    """
    EEGNetv4 with weights loaded from HuggingFace Hub.

    Requires: `pip install braindecode huggingface_hub`

    Args:
        in_channels: EEG channels (default: 16).
        num_classes: Output classes (default: 2).
        n_times: Timesteps per window (default: 256).
        freeze_backbone: Freeze all but classifier if True (default: False).
        pretrained_repo: HF repo id (default: guido151/EEGNetv4).
        pretrained_file: Weights filename in repo.

    Example:
        >>> model = EEGNetHFPretrained(in_channels=16)  # downloads from HF Hub
    """

    def __init__(self, in_channels: int = 16, num_classes: int = 2, n_times: int = 256,
                 freeze_backbone: bool = False,
                 pretrained_repo: str = "guido151/EEGNetv4",
                 pretrained_file: str = "EEGNetv4_Lee2019_ERP/params.pt"):
        super().__init__()
        try:
            from braindecode.models import EEGNetv4
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "EEGNetHFPretrained requires braindecode and huggingface_hub. "
                "Install: pip install braindecode huggingface_hub"
            ) from exc
        self.model = EEGNetv4(n_chans=in_channels, n_outputs=num_classes, n_times=n_times, drop_prob=0.25)
        state = torch.load(hf_hub_download(repo_id=pretrained_repo, filename=pretrained_file), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        _load_partial_state_dict(self.model, state)
        if freeze_backbone:
            _freeze_backbone(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, C, T). Returns: logits (batch, num_classes)."""
        return self.model(ensure_3d(x))


class DeepConvNet(nn.Module):
    """
    DeepConvNet: EEGNet-style 2D convolutions on (1, C, T) input.

    Args:
        in_channels: EEG channels (default: 16).
        num_classes: Output classes (default: 2).
        dropout: Dropout probability (default: 0.4).

    Example:
        >>> model = DeepConvNet(in_channels=16)
        >>> out = model(torch.randn(4, 16, 256))
        >>> assert out.shape == (4, 2)
    """

    def __init__(self, in_channels: int = 16, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), padding=(0, 5), bias=False),
            nn.Conv2d(25, 25, kernel_size=(in_channels, 1), bias=False),
            nn.BatchNorm2d(25), nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)), nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            _make_dcn_block(25, 50, dropout), _make_dcn_block(50, 100, dropout),
            _make_dcn_block(100, 200, dropout), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(200, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, C, T). Returns: logits (batch, num_classes)."""
        x = ensure_3d(x).unsqueeze(1)
        return self.classifier(self.blocks(self.stem(x)).flatten(1))


class MultiScaleCNN(nn.Module):
    """
    Parallel multi-scale 1D CNN branches fused with residual block.

    Args:
        in_channels: EEG channels (default: 16).
        num_classes: Output classes (default: 2).
        branch_channels: Per-branch output channels (default: 32).
        kernel_sizes: Kernel size per branch (default: (3, 7, 15, 31)).
        dropout: Dropout probability (default: 0.3).

    Example:
        >>> model = MultiScaleCNN(in_channels=16)
        >>> out = model(torch.randn(4, 16, 256))
        >>> assert out.shape == (4, 2)
    """

    def __init__(self, in_channels: int = 16, num_classes: int = 2,
                 branch_channels: int = 32,
                 kernel_sizes: Tuple[int, ...] = (3, 7, 15, 31),
                 dropout: float = 0.3):
        super().__init__()
        self.branches = nn.ModuleList([
            MultiScaleBranch1d(in_channels, branch_channels, k, dropout) for k in kernel_sizes
        ])
        fusion_in = branch_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            ConvBNAct1d(fusion_in, 128, kernel_size=3, dropout=dropout),
            ResidualBlock1d(128, kernel_size=3, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = make_mlp((128, 64, num_classes), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, C, T). Returns: logits (batch, num_classes)."""
        x = ensure_3d(x)
        fused = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.classifier(self.fusion(fused).flatten(1))


class MultiScaleAttentionCNN(nn.Module):
    """
    MultiScaleCNN with SE channel attention blocks in the fusion stage.

    Args:
        in_channels: EEG channels (default: 16).
        num_classes: Output classes (default: 2).
        branch_channels: Per-branch output channels (default: 32).
        kernel_sizes: Kernel size per branch (default: (3, 7, 15, 31)).
        dropout: Dropout probability (default: 0.35).

    Example:
        >>> model = MultiScaleAttentionCNN(in_channels=16)
        >>> out = model(torch.randn(4, 16, 256))
        >>> assert out.shape == (4, 2)
    """

    def __init__(self, in_channels: int = 16, num_classes: int = 2,
                 branch_channels: int = 32,
                 kernel_sizes: Tuple[int, ...] = (3, 7, 15, 31),
                 dropout: float = 0.35):
        super().__init__()
        self.branches = nn.ModuleList([
            MultiScaleBranch1d(in_channels, branch_channels, k, dropout) for k in kernel_sizes
        ])
        fusion_in = branch_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            ConvBNAct1d(fusion_in, 128, kernel_size=3, dropout=dropout),
            SEBlock1d(128, reduction=8),
            ResidualBlock1d(128, kernel_size=3, dropout=dropout),
            SEBlock1d(128, reduction=8),
            DepthwiseSeparableConv1d(128, 160, kernel_size=5, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = make_mlp((160, 64, num_classes), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, C, T). Returns: logits (batch, num_classes)."""
        x = ensure_3d(x)
        fused = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.classifier(self.fusion(fused).flatten(1))


class STEEGFormerPretrained(nn.Module):
    """
    ST-EEGFormer loaded from eugenehp/ST-EEGFormer on HuggingFace Hub.

    Requires: pip install huggingface_hub safetensors

    Args:
        in_channels: EEG channels (default: 16).
        num_classes: Output classes (default: 2).
        n_times: Timesteps — must be divisible by patch_size (default: 768).
        sfreq: Must be 128 (default: 128).
        freeze_backbone: Freeze encoder weights (default: False).
        pretrained_repo: HF repo id (default: eugenehp/ST-EEGFormer).
        variant: Model size — small/base/large/largev2 (default: small).
        patch_size: Patch size in samples (default: 16).
        max_channels: Max channel embedding size (default: 145).
        max_patches: Max temporal patches (default: 512).
        channel_embed_mode: 'mean' or 'index' (default: mean).

    Example:
        >>> model = STEEGFormerPretrained(in_channels=16, n_times=768, sfreq=128)
    """

    VARIANTS = {
        "small":   {"embed_dim": 512,  "depth": 8,  "num_heads": 8,  "weights": "ST-EEGFormer_small_encoder.safetensors"},
        "base":    {"embed_dim": 768,  "depth": 12, "num_heads": 12, "weights": "ST-EEGFormer_base_encoder.safetensors"},
        "large":   {"embed_dim": 1024, "depth": 24, "num_heads": 16, "weights": "ST-EEGFormer_large_encoder.safetensors"},
        "largev2": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "weights": "ST-EEGFormer_largeV2_encoder.safetensors"},
    }

    def __init__(self, in_channels: int = 16, num_classes: int = 2,
                 n_times: int = 768, sfreq: int = 128,
                 freeze_backbone: bool = False,
                 pretrained_repo: str = "eugenehp/ST-EEGFormer",
                 variant: str = "small", patch_size: int = 16,
                 max_channels: int = 145, max_patches: int = 512,
                 channel_embed_mode: str = "mean"):
        super().__init__()
        if sfreq != 128:
            raise ValueError(f"ST-EEGFormer requires sfreq=128, got {sfreq}.")
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose: {sorted(self.VARIANTS)}.")
        if channel_embed_mode not in {"mean", "index"}:
            raise ValueError("channel_embed_mode must be 'mean' or 'index'.")
        spec = self.VARIANTS[variant]
        embed_dim = spec["embed_dim"]
        self.channel_embed_mode = channel_embed_mode
        self.patch_embed = _STEEGPatchEmbed(patch_size, embed_dim)
        self.enc_channel_emd = _STEEGChannelEmbedding(max_channels, embed_dim)
        self.enc_temporal_emd = _STEEGTemporalEncoding(max_patches, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            _STEEGBlock(embed_dim, spec["num_heads"]) for _ in range(spec["depth"])
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._load_pretrained_weights(pretrained_repo, spec["weights"])
        if freeze_backbone:
            for name, param in self.named_parameters():
                if not name.startswith("classifier."):
                    param.requires_grad = False

    def _load_pretrained_weights(self, repo: str, filename: str) -> None:
        """Download and load safetensors weights from HF Hub."""
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "ST-EEGFormer requires huggingface_hub and safetensors. "
                "Install: pip install huggingface_hub safetensors"
            ) from exc
        state = load_file(hf_hub_download(repo_id=repo, filename=filename))
        missing, unexpected = self.load_state_dict(state, strict=False)
        if self.channel_embed_mode == "mean":
            with torch.no_grad():
                mean_emb = self.enc_channel_emd.channel_transformation.weight.mean(dim=0, keepdim=True)
                n = self.enc_channel_emd.channel_transformation.num_embeddings
                self.enc_channel_emd.channel_transformation.weight.copy_(mean_emb.repeat(n, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, C, T). Returns: logits (batch, num_classes)."""
        x = ensure_3d(x)
        batch, channels, _ = x.shape
        tokens, n_patches = self.patch_embed(x)
        temporal = self.enc_temporal_emd(batch, channels, n_patches)
        if self.channel_embed_mode == "mean":
            channel = self.enc_channel_emd.channel_transformation.weight.mean(dim=0)
            channel = channel.view(1, 1, -1).expand(batch, channels * n_patches, -1)
        else:
            channel = self.enc_channel_emd(batch, channels, n_patches, x.device)
        tokens = tokens + temporal + channel
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.classifier(self.norm(tokens[:, 0]))


class _STEEGPatchEmbed(nn.Module):
    """Split (batch, C, T) into patches and project to embed_dim."""
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        batch, channels, samples = x.shape
        if samples % self.patch_size != 0:
            raise ValueError(f"samples={samples} must be divisible by patch_size={self.patch_size}.")
        n_patches = samples // self.patch_size
        x = x.view(batch, channels, n_patches, self.patch_size)
        return self.proj(x).reshape(batch, channels * n_patches, -1), n_patches


class _STEEGChannelEmbedding(nn.Module):
    """Learnable channel embedding table."""
    def __init__(self, num_embeddings: int, embed_dim: int):
        super().__init__()
        self.channel_transformation = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, batch: int, n_channels: int, n_patches: int, device: torch.device) -> torch.Tensor:
        ids = torch.arange(n_channels, device=device).clamp_max(self.channel_transformation.num_embeddings - 1)
        emb = self.channel_transformation(ids).unsqueeze(1).expand(n_channels, n_patches, -1)
        return emb.reshape(1, n_channels * n_patches, -1).expand(batch, -1, -1)


class _STEEGTemporalEncoding(nn.Module):
    """Sinusoidal temporal positional encoding."""
    def __init__(self, max_patches: int, embed_dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_patches, embed_dim), requires_grad=False)
        position = torch.arange(max_patches, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_patches, embed_dim)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.pe.data.copy_(pe.unsqueeze(0))

    def forward(self, batch: int, n_channels: int, n_patches: int) -> torch.Tensor:
        pe = self.pe[:, :n_patches, :].unsqueeze(1).expand(batch, n_channels, n_patches, -1)
        return pe.reshape(batch, n_channels * n_patches, -1)


class _STEEGAttention(nn.Module):
    """Multi-head self-attention for ST-EEGFormer blocks."""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(batch, tokens, embed_dim))


class _STEEGBlock(nn.Module):
    """Pre-norm transformer block for ST-EEGFormer."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _STEEGAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# ─── Private helpers ──────────────────────────────────────────────────────────

def _make_dcn_block(in_filters: int, out_filters: int, dropout: float) -> nn.Sequential:
    """Build one DeepConvNet expansion block."""
    return nn.Sequential(
        nn.Conv2d(in_filters, out_filters, kernel_size=(1, 10), padding=(0, 5), bias=False),
        nn.BatchNorm2d(out_filters), nn.ELU(inplace=True),
        nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)), nn.Dropout(dropout),
    )


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters except classifier/final_layer."""
    for name, param in model.named_parameters():
        if not any(k in name for k in ("final_layer", "classification", "classifier")):
            param.requires_grad = False


def _load_partial_state_dict(
    model: nn.Module, source_state: dict[str, torch.Tensor]
) -> None:
    """Load matching keys only; skip shape mismatches silently."""
    target_state = model.state_dict()
    patched = {
        k: source_state[k].to(dtype=target_state[k].dtype)
        for k in target_state
        if k in source_state and source_state[k].shape == target_state[k].shape
    }
    model.load_state_dict(patched, strict=False)
    logger.info("Loaded %d / %d keys from pretrained state dict.", len(patched), len(target_state))
