"""
Tiny Seizure Net: Distilled Lightweight Models
-----------------------------------------------
For edge deployment on wearables and embedded devices.

Target specs:
- Model size: <500 KB
- Inference: <10 ms
- Power: <1 mW

References:
- Knowledge Distillation (Hinton et al., 2015)
- MobileNets for efficient inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution for efficiency.
    Reduces parameters by ~9× compared to standard conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        # Depthwise: one filter per input channel
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels,
        )
        # Pointwise: 1×1 conv to mix channels
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class TinyConvBlock(nn.Module):
    """Tiny convolutional block with minimal parameters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = DepthwiseSeparableConv1d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU6(inplace=True)  # ReLU6 for quantization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TinySeizureNet(nn.Module):
    """
    Tiny Seizure Detection Network.

    Optimized for edge deployment with <100K parameters.

    Architecture:
        Input (batch, 16, 256)
        → 3 × TinyConvBlock (depthwise separable)
        → Global Average Pooling
        → Small FC Head
        → Logits

    Args:
        n_channels: Number of EEG channels (default: 16)
        time_steps: Number of time steps (default: 256)
        base_filters: Base number of filters (default: 16)
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        base_filters: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Tiny CNN
        self.features = nn.Sequential(
            TinyConvBlock(n_channels, base_filters, kernel_size=7, stride=2),
            TinyConvBlock(base_filters, base_filters * 2, kernel_size=5, stride=2),
            TinyConvBlock(base_filters * 2, base_filters * 4, kernel_size=3, stride=2),
        )

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Tiny classifier
        self.classifier = nn.Sequential(
            nn.Linear(base_filters * 4, base_filters * 2),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_filters * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: EEG signal (batch, n_channels, time_steps)

        Returns:
            logits: Classification logits (batch, 1)
        """
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_model_size_kb(self) -> float:
        """Get model size in KB (assuming float32)."""
        return self.count_parameters() * 4 / 1024


class MicroSeizureNet(nn.Module):
    """
    Micro Seizure Net: Even smaller for ultra-low-power devices.

    Target: <50K parameters, <250 KB.

    Architecture:
        Input → Channel Reduction → Single Conv → Pool → Linear
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
        hidden_dim: int = 32,
    ):
        super().__init__()

        # Channel reduction
        self.channel_reduce = nn.Linear(n_channels, 4)

        # Single conv layer
        self.conv = nn.Sequential(
            nn.Conv1d(4, hidden_dim, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(),
            nn.AdaptiveAvgPool1d(4),
        )

        # Tiny classifier
        self.classifier = nn.Linear(hidden_dim * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]

        # Channel reduction: (batch, n_channels, time) → (batch, time, 4)
        x = x.permute(0, 2, 1)
        x = self.channel_reduce(x)
        x = x.permute(0, 2, 1)  # (batch, 4, time)

        # Conv
        x = self.conv(x)  # (batch, hidden, 4)

        # Flatten and classify
        x = x.view(batch, -1)
        return self.classifier(x)


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss.
    Combines hard labels with soft teacher predictions.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,  # Weight for soft loss
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.hard_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss.

        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            labels: Ground truth labels

        Returns:
            Combined loss
        """
        # Hard loss (student vs ground truth)
        hard_loss = self.hard_loss(student_logits, labels)

        # Soft loss (student vs teacher, with temperature)
        soft_student = torch.sigmoid(student_logits / self.temperature)
        soft_teacher = torch.sigmoid(teacher_logits / self.temperature)

        soft_loss = F.mse_loss(soft_student, soft_teacher)

        # Temperature scaling for soft loss
        soft_loss = soft_loss * (self.temperature ** 2)

        # Combined loss
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


class MultiTeacherDistillation(nn.Module):
    """
    Distill from multiple teacher models.
    Each teacher contributes to soft labels.
    """

    def __init__(
        self,
        teachers: List[nn.Module],
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
    ):
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
        self.student = student
        self.temperature = temperature
        self.alpha = alpha

        # Freeze teachers
        for teacher in self.teachers:
            for param in teacher.parameters():
                param.requires_grad = False

        self.hard_loss = nn.BCEWithLogitsLoss()

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with distillation.

        Returns:
            student_logits: Student predictions
            loss: Distillation loss
        """
        # Get teacher predictions
        teacher_probs = []
        for teacher in self.teachers:
            teacher.eval()
            with torch.no_grad():
                t_out = teacher(x)
                if isinstance(t_out, tuple):
                    t_out = t_out[0]
                t_prob = torch.sigmoid(t_out / self.temperature)
                teacher_probs.append(t_prob)

        # Average teacher predictions
        avg_teacher_prob = torch.stack(teacher_probs, dim=0).mean(dim=0)

        # Student forward
        student_logits = self.student(x)
        student_prob = torch.sigmoid(student_logits / self.temperature)

        # Losses
        hard_loss = self.hard_loss(student_logits, labels)
        soft_loss = F.mse_loss(student_prob, avg_teacher_prob) * (self.temperature ** 2)

        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return student_logits, loss


class QuantizedTinyNet(nn.Module):
    """
    Quantization-ready tiny network.
    Uses operations compatible with INT8 quantization.
    """

    def __init__(
        self,
        n_channels: int = 16,
        time_steps: int = 256,
    ):
        super().__init__()

        # Quantization stubs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Quantization-friendly architecture
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU6(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU6(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse Conv-BN-ReLU for better quantization."""
        torch.quantization.fuse_modules(
            self.features,
            [["0", "1", "2"], ["3", "4", "5"]],
            inplace=True,
        )


if __name__ == "__main__":
    # Test TinySeizureNet
    print("Testing TinySeizureNet...")
    model = TinySeizureNet(n_channels=16, time_steps=256)
    x = torch.randn(8, 16, 256)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Model size: {model.get_model_size_kb():.1f} KB")

    # Test MicroSeizureNet
    print("\nTesting MicroSeizureNet...")
    micro = MicroSeizureNet(n_channels=16, time_steps=256)
    out = micro(x)
    print(f"Output shape: {out.shape}")
    params = sum(p.numel() for p in micro.parameters())
    print(f"Parameters: {params:,}")
    print(f"Model size: {params * 4 / 1024:.1f} KB")
