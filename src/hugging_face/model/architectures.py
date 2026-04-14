from __future__ import annotations

import math

import torch
from torch import nn

from .blocks import (
    ConvBNAct1d,
    DepthwiseSeparableConv1d,
    DilatedResidualBlock1d,
    MultiScaleBranch1d,
    ResidualBlock1d,
    SEBlock1d,
    ensure_3d,
    make_mlp,
)


class BaselineCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
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
        x = ensure_3d(x)
        x = self.features(x).flatten(1)
        return self.classifier(x)


class EnhancedCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.35,
    ):
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
        x = ensure_3d(x)
        x = self.stem(x)
        x = self.encoder(x).flatten(1)
        return self.classifier(x)


class BIOTPretrained(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        n_times: int = 256,
        sfreq: int = 256,
        freeze_backbone: bool = False,
        pretrained_repo: str = "braindecode/biot-pretrained-prest-16chs",
    ):
        super().__init__()
        try:
            from braindecode.models import BIOT
        except ImportError as exc:
            raise ImportError(
                "BIOT requires braindecode with Hugging Face support. "
                "Install it with `pip install braindecode[hug] huggingface_hub`."
            ) from exc

        self.model = BIOT.from_pretrained(
            pretrained_repo,
            n_chans=in_channels,
            n_outputs=num_classes,
            n_times=n_times,
            sfreq=sfreq,
        )
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "final_layer" not in name and "classification" not in name and "classifier" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        return self.model(x)


class BENDRPretrained(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        n_times: int = 256,
        sfreq: int = 256,
        freeze_backbone: bool = False,
        pretrained_repo: str = "braindecode/braindecode-bendr",
    ):
        super().__init__()
        try:
            from braindecode.models import BENDR
        except ImportError as exc:
            raise ImportError(
                "BENDR requires braindecode with Hugging Face support. "
                "Install it with `pip install braindecode[hug] huggingface_hub`."
            ) from exc

        self.model = BENDR.from_pretrained(
            pretrained_repo,
            n_chans=in_channels,
            n_outputs=num_classes,
            n_times=n_times,
            sfreq=sfreq,
        )
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "final_layer" not in name and "classification" not in name and "classifier" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        return self.model(x)


class EEGPTPretrained(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        n_times: int = 256,
        sfreq: int = 256,
        freeze_backbone: bool = False,
        pretrained_repo: str = "braindecode/eegpt-pretrained",
    ):
        super().__init__()
        try:
            from braindecode.models import EEGPT
        except ImportError as exc:
            raise ImportError(
                "EEGPT requires a braindecode version that exposes EEGPT, plus Hugging Face support. "
                "Install it with `pip install --upgrade braindecode[hug] huggingface_hub`."
            ) from exc

        self.model = EEGPT.from_pretrained(
            pretrained_repo,
            n_chans=in_channels,
            n_outputs=num_classes,
            n_times=n_times,
            sfreq=sfreq,
        )
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "final_layer" not in name and "classification" not in name and "classifier" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        return self.model(x)


class EEGNetHFPretrained(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        n_times: int = 256,
        freeze_backbone: bool = False,
        pretrained_repo: str = "guido151/EEGNetv4",
        pretrained_file: str = "EEGNetv4_Lee2019_ERP/params.pt",
    ):
        super().__init__()
        try:
            from braindecode.models import EEGNetv4
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "Hugging Face EEGNet requires braindecode and huggingface_hub. "
                "Install it with `pip install braindecode[hug] huggingface_hub`."
            ) from exc

        self.model = EEGNetv4(
            n_chans=in_channels,
            n_outputs=num_classes,
            n_times=n_times,
            drop_prob=0.25,
        )
        model_path = hf_hub_download(repo_id=pretrained_repo, filename=pretrained_file)
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        _load_partial_state_dict(self.model, state_dict)

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "final_layer" not in name and "classification" not in name and "classifier" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        return self.model(x)


class EEGNetLocal(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        separable_filters: int = 32,
        dropout: float = 0.25,
    ):
        super().__init__()
        spatial_filters = temporal_filters * depth_multiplier
        self.temporal = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(temporal_filters),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(in_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, separable_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(separable_filters),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(separable_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = x.flatten(1)
        return self.classifier(x)


class _STEEGPatchEmbed(nn.Module):
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
        x = x.reshape(batch, channels * n_patches, -1)
        return x, n_patches


class _STEEGChannelEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embed_dim: int):
        super().__init__()
        self.channel_transformation = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, batch: int, n_channels: int, n_patches: int, device: torch.device) -> torch.Tensor:
        ids = torch.arange(n_channels, device=device, dtype=torch.long)
        ids = ids.clamp_max(self.channel_transformation.num_embeddings - 1)
        emb = self.channel_transformation(ids)
        emb = emb.unsqueeze(1).expand(n_channels, n_patches, emb.shape[-1]).reshape(1, n_channels * n_patches, -1)
        return emb.expand(batch, -1, -1)


class _STEEGTemporalEncoding(nn.Module):
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
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, embed_dim)
        return self.proj(x)


class _STEEGMLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _STEEGBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = _STEEGAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = _STEEGMLP(embed_dim, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class STEEGFormerPretrained(nn.Module):
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
        n_times: int = 128,
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
        self.patch_embed = _STEEGPatchEmbed(patch_size=patch_size, embed_dim=embed_dim)
        self.enc_channel_emd = _STEEGChannelEmbedding(num_embeddings=max_channels, embed_dim=embed_dim)
        self.enc_temporal_emd = _STEEGTemporalEncoding(max_patches=max_patches, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList(
            [_STEEGBlock(embed_dim=embed_dim, num_heads=spec["num_heads"]) for _ in range(spec["depth"])]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._load_pretrained_weights(pretrained_repo=pretrained_repo, filename=spec["weights"])

        if freeze_backbone:
            for name, param in self.named_parameters():
                if not name.startswith("classifier."):
                    param.requires_grad = False

    def _load_pretrained_weights(self, pretrained_repo: str, filename: str):
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "ST-EEGFormer requires `huggingface_hub` and `safetensors`. "
                "Install them with `pip install huggingface_hub safetensors`."
            ) from exc

        weights_path = hf_hub_download(repo_id=pretrained_repo, filename=filename)
        state_dict = load_file(weights_path)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected ST-EEGFormer keys: {unexpected}")
        if self.channel_embed_mode == "mean":
            with torch.no_grad():
                mean_embedding = self.enc_channel_emd.channel_transformation.weight.mean(dim=0, keepdim=True)
                self.enc_channel_emd.channel_transformation.weight.copy_(mean_embedding.repeat(
                    self.enc_channel_emd.channel_transformation.num_embeddings, 1
                ))
        self._missing_keys = missing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        batch, channels, samples = x.shape
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
        cls_out = self.norm(tokens[:, 0])
        return self.classifier(cls_out)


def _load_partial_state_dict(model: nn.Module, source_state: dict[str, torch.Tensor]):
    target_state = model.state_dict()
    patched: dict[str, torch.Tensor] = {}

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            continue
        if source_tensor.shape == target_tensor.shape:
            patched[key] = source_tensor.to(dtype=target_tensor.dtype)

    model.load_state_dict(patched, strict=False)


class DeepConvNet(nn.Module):
    def __init__(self, in_channels: int = 16, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 10), padding=(0, 5), bias=False),
            nn.Conv2d(25, 25, kernel_size=(in_channels, 1), bias=False),
            nn.BatchNorm2d(25),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            self._make_block(25, 50, dropout),
            self._make_block(50, 100, dropout),
            self._make_block(100, 200, dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(200, num_classes)

    @staticmethod
    def _make_block(in_filters: int, out_filters: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_filters, out_filters, kernel_size=(1, 10), padding=(0, 5), bias=False),
            nn.BatchNorm2d(out_filters),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        x = x.flatten(1)
        return self.classifier(x)


class MultiScaleCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        branch_channels: int = 32,
        kernel_sizes: tuple[int, ...] = (3, 7, 15, 31),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [MultiScaleBranch1d(in_channels, branch_channels, kernel_size=k, dropout=dropout) for k in kernel_sizes]
        )
        fusion_channels = branch_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            ConvBNAct1d(fusion_channels, 128, kernel_size=3, dropout=dropout),
            ResidualBlock1d(128, kernel_size=3, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = make_mlp((128, 64, num_classes), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        multi_scale = [branch(x) for branch in self.branches]
        x = torch.cat(multi_scale, dim=1)
        x = self.fusion(x).flatten(1)
        return self.classifier(x)


class MultiScaleAttentionCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        branch_channels: int = 32,
        kernel_sizes: tuple[int, ...] = (3, 7, 15, 31),
        dropout: float = 0.35,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [MultiScaleBranch1d(in_channels, branch_channels, kernel_size=k, dropout=dropout) for k in kernel_sizes]
        )
        fusion_channels = branch_channels * len(kernel_sizes)
        self.fusion = nn.Sequential(
            ConvBNAct1d(fusion_channels, 128, kernel_size=3, dropout=dropout),
            SEBlock1d(128, reduction=8),
            ResidualBlock1d(128, kernel_size=3, dropout=dropout),
            SEBlock1d(128, reduction=8),
            DepthwiseSeparableConv1d(128, 160, kernel_size=5, dropout=dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = make_mlp((160, 64, num_classes), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        multi_scale = [branch(x) for branch in self.branches]
        x = torch.cat(multi_scale, dim=1)
        x = self.fusion(x).flatten(1)
        return self.classifier(x)
