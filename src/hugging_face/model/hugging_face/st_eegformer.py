from __future__ import annotations

import math

import torch
from torch import nn

from model.blocks import ensure_3d


class _PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        batch, channels, samples = x.shape
        if samples % self.patch_size != 0:
            raise ValueError(
                f"ST-EEGFormer expects samples divisible by patch size {self.patch_size}, got {samples}."
            )
        n_patches = samples // self.patch_size
        x = x.view(batch, channels, n_patches, self.patch_size)
        x = self.proj(x)
        return x.reshape(batch, channels * n_patches, -1), n_patches


class _ChannelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embed_dim: int):
        super().__init__()
        self.channel_transformation = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, batch: int, n_channels: int, n_patches: int, device: torch.device) -> torch.Tensor:
        ids = torch.arange(n_channels, device=device, dtype=torch.long)
        ids = ids.clamp_max(self.channel_transformation.num_embeddings - 1)
        emb = self.channel_transformation(ids)
        emb = emb.unsqueeze(1).expand(n_channels, n_patches, emb.shape[-1]).reshape(1, n_channels * n_patches, -1)
        return emb.expand(batch, -1, -1)


class _TemporalEncoding(nn.Module):
    def __init__(self, max_patches: int, embed_dim: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_patches, embed_dim), requires_grad=False)
        self._init_sinusoidal()

    def _init_sinusoidal(self):
        _, max_patches, embed_dim = self.pe.shape
        position = torch.arange(max_patches, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_patches, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe.data.copy_(pe.unsqueeze(0))

    def forward(self, batch: int, n_channels: int, n_patches: int) -> torch.Tensor:
        if n_patches > self.pe.shape[1]:
            raise ValueError(f"ST-EEGFormer max patches is {self.pe.shape[1]}, got {n_patches}.")
        pe = self.pe[:, :n_patches, :]
        pe = pe.unsqueeze(1).expand(batch, n_channels, n_patches, pe.shape[-1]).reshape(batch, n_channels * n_patches, -1)
        return pe


class _Attention(nn.Module):
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
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, embed_dim)
        return self.proj(x)


class _Mlp(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _Mlp(embed_dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HuggingFaceSTEEGFormer(nn.Module):
    VARIANTS = {
        "small": {"embed_dim": 512, "depth": 8, "num_heads": 8, "weights": "ST-EEGFormer_small_encoder.safetensors"},
        "base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "weights": "ST-EEGFormer_base_encoder.safetensors"},
        "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "weights": "ST-EEGFormer_large_encoder.safetensors"},
        "largev2": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "weights": "ST-EEGFormer_largeV2_encoder.safetensors"},
    }

    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        n_times: int = 768,
        sfreq: int = 128,
        freeze_backbone: bool = False,
        pretrained_repo: str = "eugenehp/ST-EEGFormer",
        variant: str = "small",
        patch_size: int = 16,
        max_channels: int = 145,
        max_patches: int = 512,
        channel_embed_mode: str = "mean",
    ):
        super().__init__()
        if sfreq != 128:
            raise ValueError(f"ST-EEGFormer is documented for 128 Hz input. Got sfreq={sfreq}.")
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown ST-EEGFormer variant '{variant}'. Choose from {sorted(self.VARIANTS)}.")
        if channel_embed_mode not in {"mean", "index"}:
            raise ValueError("channel_embed_mode must be 'mean' or 'index'.")

        spec = self.VARIANTS[variant]
        embed_dim = spec["embed_dim"]
        self.channel_embed_mode = channel_embed_mode
        self.variant = variant
        self.pretrained_repo = pretrained_repo
        self.patch_embed = _PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.enc_channel_emd = _ChannelEmbedding(num_embeddings=max_channels, embed_dim=embed_dim)
        self.enc_temporal_emd = _TemporalEncoding(max_patches=max_patches, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([_Block(embed_dim=embed_dim, num_heads=spec["num_heads"]) for _ in range(spec["depth"])])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._load_pretrained_weights(filename=spec["weights"])

        if freeze_backbone:
            for name, param in self.named_parameters():
                if not name.startswith("classifier."):
                    param.requires_grad = False

    def _load_pretrained_weights(self, filename: str):
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Hugging Face ST-EEGFormer requires `huggingface_hub` and `safetensors`. "
                "Install them with `pip install huggingface_hub safetensors`."
            ) from exc

        weights_path = hf_hub_download(repo_id=self.pretrained_repo, filename=filename)
        state_dict = load_file(weights_path)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected ST-EEGFormer keys: {unexpected}")
        if self.channel_embed_mode == "mean":
            with torch.no_grad():
                mean_embedding = self.enc_channel_emd.channel_transformation.weight.mean(dim=0, keepdim=True)
                self.enc_channel_emd.channel_transformation.weight.copy_(
                    mean_embedding.repeat(self.enc_channel_emd.channel_transformation.num_embeddings, 1)
                )
        self._missing_keys = missing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        batch, channels, _ = x.shape
        tokens, n_patches = self.patch_embed(x)
        temporal = self.enc_temporal_emd(batch, channels, n_patches)
        if self.channel_embed_mode == "mean":
            channel = self.enc_channel_emd.channel_transformation.weight.mean(dim=0, keepdim=True)
            channel = channel.view(1, 1, -1).expand(batch, channels * n_patches, -1)
        else:
            channel = self.enc_channel_emd(batch, channels, n_patches, x.device)
        tokens = tokens + temporal + channel
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.classifier(self.norm(tokens[:, 0]))
