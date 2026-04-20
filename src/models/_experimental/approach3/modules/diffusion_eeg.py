"""
Diffusion Model for EEG Data Augmentation
-----------------------------------------
Generate synthetic seizure EEG to balance imbalanced datasets.

Key use cases:
1. Generate realistic seizure samples (~5% → balanced)
2. Denoise noisy EEG recordings
3. Anomaly detection (seizure as deviation from learned distribution)

References:
- Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- Diffusion Models for Time Series (Tashiro et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock1D(nn.Module):
    """1D Convolutional block with GroupNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_groups: int = 8,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock1D(nn.Module):
    """Residual block with time embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
    ):
        super().__init__()
        self.conv1 = ConvBlock1D(in_channels, out_channels, num_groups=num_groups)
        self.conv2 = ConvBlock1D(out_channels, out_channels, num_groups=num_groups)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_mlp(t)[:, :, None]
        h = self.conv2(h)
        return h + self.residual(x)


class UNet1D(nn.Module):
    """
    1D U-Net for EEG diffusion.

    Architecture:
        Encoder (downsampling) → Bottleneck → Decoder (upsampling)
        with skip connections and time embeddings.
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.n_channels = n_channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Initial projection
        self.init_conv = nn.Conv1d(n_channels, base_channels, 3, padding=1)

        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        channels = [base_channels]
        in_ch = base_channels

        for mult in channel_mults:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.encoder.append(ResBlock1D(in_ch, out_ch, time_emb_dim))
                in_ch = out_ch
            channels.append(in_ch)
            self.downsample.append(nn.Conv1d(in_ch, in_ch, 3, stride=2, padding=1))

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResBlock1D(in_ch, in_ch, time_emb_dim),
            ResBlock1D(in_ch, in_ch, time_emb_dim),
        ])

        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.upsample.append(nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1))
            for _ in range(num_res_blocks):
                self.decoder.append(ResBlock1D(in_ch + channels.pop(), out_ch, time_emb_dim))
                in_ch = out_ch

        # Final projection
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, n_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy EEG (batch, n_channels, time_steps)
            t: Timesteps (batch,)

        Returns:
            Predicted noise (batch, n_channels, time_steps)
        """
        # Time embedding
        t_emb = self.time_mlp(t)

        # Initial
        h = self.init_conv(x)

        # Encoder
        skips = [h]
        enc_idx = 0
        for i, downsample in enumerate(self.downsample):
            for _ in range(2):  # num_res_blocks
                h = self.encoder[enc_idx](h, t_emb)
                enc_idx += 1
            skips.append(h)
            h = downsample(h)

        # Bottleneck
        for block in self.bottleneck:
            h = block(h, t_emb)

        # Decoder
        dec_idx = 0
        for i, upsample in enumerate(self.upsample):
            h = upsample(h)
            # Handle size mismatch
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="linear")
            h = torch.cat([h, skip], dim=1)
            for _ in range(2):  # num_res_blocks
                h = self.decoder[dec_idx](h, t_emb)
                dec_idx += 1

        # Final
        h = torch.cat([h, skips.pop()], dim=1) if skips else h
        return self.final_conv(h)


class DiffusionScheduler:
    """
    Diffusion noise scheduler.
    Implements linear and cosine schedules.
    """

    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        self.num_steps = num_steps

        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule == "cosine":
            s = 0.008
            steps = torch.linspace(0, num_steps, num_steps + 1)
            alphas_cumprod = torch.cos((steps / num_steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def add_noise(
        self, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to samples at timestep t."""
        if noise is None:
            noise = torch.randn_like(x)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        noisy = sqrt_alpha.to(x.device) * x + sqrt_one_minus_alpha.to(x.device) * noise
        return noisy, noise


class EEGDiffusion(nn.Module):
    """
    Complete diffusion model for EEG generation.

    Can be used for:
    1. Generating synthetic seizure data
    2. Denoising EEG
    3. Anomaly detection
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        num_diffusion_steps: int = 1000,
        base_channels: int = 64,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.time_steps = time_steps
        self.num_diffusion_steps = num_diffusion_steps

        # U-Net denoiser
        self.unet = UNet1D(
            n_channels=n_channels,
            time_steps=time_steps,
            base_channels=base_channels,
        )

        # Scheduler
        self.scheduler = DiffusionScheduler(num_diffusion_steps)

        # Class embedding for conditional generation
        self.class_embed = nn.Embedding(2, 128)  # 0: background, 1: seizure

    def forward(
        self, x: torch.Tensor, class_label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Training forward pass.
        Returns MSE loss between predicted and actual noise.
        """
        batch = x.shape[0]
        device = x.device

        # Sample random timesteps
        t = torch.randint(0, self.num_diffusion_steps, (batch,), device=device)

        # Add noise
        noisy, noise = self.scheduler.add_noise(x, t.cpu())
        noisy = noisy.to(device)
        noise = noise.to(device)

        # Predict noise
        pred_noise = self.unet(noisy, t.float())

        # MSE loss
        loss = F.mse_loss(pred_noise, noise)

        return loss

    @torch.no_grad()
    def generate(
        self,
        batch_size: int = 1,
        class_label: int = 1,  # 1 = seizure
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generate synthetic EEG samples."""
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.eval()

        # Start from pure noise
        x = torch.randn(batch_size, self.n_channels, self.time_steps, device=device)

        # Reverse diffusion
        for t in reversed(range(self.num_diffusion_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.float)

            # Predict noise
            pred_noise = self.unet(x, t_batch)

            # Denoise step
            alpha = self.scheduler.alphas[t].to(device)
            alpha_cumprod = self.scheduler.alphas_cumprod[t].to(device)
            beta = self.scheduler.betas[t].to(device)

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
            ) + torch.sqrt(beta) * noise

        return x


if __name__ == "__main__":
    # Test
    model = EEGDiffusion(n_channels=16, time_steps=256, num_diffusion_steps=100)
    x = torch.randn(8, 16, 256)

    # Training
    loss = model(x)
    print(f"Training loss: {loss.item():.4f}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generation
    generated = model.generate(batch_size=2, device="cpu")
    print(f"Generated shape: {generated.shape}")
