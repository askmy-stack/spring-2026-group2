from __future__ import annotations

import math

import torch
from torch import nn


def ensure_3d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"Expected (batch, channels, samples) or (channels, samples), got shape {tuple(x.shape)}.")
    return x


class _STEEGPatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        batch, channels, samples = x.shape
        if samples % self.patch_size != 0:
            raise ValueError(f"samples={samples} must be divisible by patch_size={self.patch_size}.")
        n_patches = samples // self.patch_size
        x = x.view(batch, channels, n_patches, self.patch_size)
        return self.proj(x).reshape(batch, channels * n_patches, -1), n_patches


class _STEEGChannelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embed_dim: int):
        super().__init__()
        self.channel_transformation = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, batch: int, n_channels: int, n_patches: int, device: torch.device) -> torch.Tensor:
        ids = torch.arange(n_channels, device=device).clamp_max(self.channel_transformation.num_embeddings - 1)
        emb = self.channel_transformation(ids).unsqueeze(1).expand(n_channels, n_patches, -1)
        return emb.reshape(1, n_channels * n_patches, -1).expand(batch, -1, -1)


class _STEEGTemporalEncoding(nn.Module):
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
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _STEEGAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class HFSTEEGFormerPretrainedModel(nn.Module):
    VARIANTS = {
        "small": {"embed_dim": 512, "depth": 8, "num_heads": 8},
        "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
        "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
        "largev2": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    }

    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        n_times: int = 128,
        sfreq: int = 128,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        variant: str = "small",
        patch_size: int = 16,
        max_channels: int = 145,
        max_patches: int = 512,
        channel_embed_mode: str = "mean",
        **_: object,
    ):
        super().__init__()
        del dropout
        del in_channels
        if sfreq != 128:
            raise ValueError(f"HF ST-EEGFormer requires sfreq=128, got {sfreq}.")
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. Choose: {sorted(self.VARIANTS)}.")
        if channel_embed_mode not in {"mean", "index"}:
            raise ValueError("channel_embed_mode must be 'mean' or 'index'.")

        spec = self.VARIANTS[variant]
        embed_dim = spec["embed_dim"]
        self.n_times = int(n_times)
        self.channel_embed_mode = channel_embed_mode
        self.patch_embed = _STEEGPatchEmbed(patch_size, embed_dim)
        self.enc_channel_emd = _STEEGChannelEmbedding(max_channels, embed_dim)
        self.enc_temporal_emd = _STEEGTemporalEncoding(max_patches, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([_STEEGBlock(embed_dim, spec["num_heads"]) for _ in range(spec["depth"])])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        if freeze_backbone:
            for name, param in self.named_parameters():
                if not name.startswith("classifier."):
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        batch, channels, samples = x.shape
        if samples != self.n_times:
            raise ValueError(f"HF ST-EEGFormer expects n_times={self.n_times}, got {samples}.")
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
