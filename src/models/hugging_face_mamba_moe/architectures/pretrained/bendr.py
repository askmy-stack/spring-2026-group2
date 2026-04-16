"""BENDR pretrained model wrapper. Requires: pip install braindecode[hug] huggingface_hub"""
from __future__ import annotations
import logging
import torch
from torch import nn
from ...modules.hf_blocks import ensure_3d
from .base import PretrainedInputSpec, validate_pretrained_inputs

logger = logging.getLogger(__name__)


class BENDRPretrainedModel(nn.Module):
    """
    BENDR loaded from braindecode/braindecode-bendr on HuggingFace Hub.

    Args:
        in_channels: EEG channels — must be 16 (default: 16).
        num_classes: Output classes (default: 2).
        n_times: Timesteps per window (default: 256).
        sfreq: Sampling frequency (default: 256).
        freeze_backbone: Freeze encoder weights if True (default: False).
        pretrained_repo: HF repo id (default: braindecode/braindecode-bendr).

    Example:
        >>> model = BENDRPretrainedModel(in_channels=16)  # downloads from HF Hub
    """

    SPEC = PretrainedInputSpec(
        model_id="bendr_pretrained",
        repo_id="braindecode/braindecode-bendr",
        required_channels=16,
        notes="BENDR accepts varying lengths; matching pretraining setup is recommended.",
    )

    def __init__(self, in_channels: int = 16, num_classes: int = 2,
                 n_times: int = 256, sfreq: int = 256,
                 freeze_backbone: bool = False,
                 pretrained_repo: str = "braindecode/braindecode-bendr"):
        super().__init__()
        validate_pretrained_inputs(self.SPEC, in_channels=in_channels, n_times=n_times, sfreq=sfreq)
        try:
            from braindecode.models import BENDR
        except ImportError as exc:
            raise ImportError(
                "BENDR requires braindecode[hug] and huggingface_hub. "
                "Install: pip install 'braindecode[hug]' huggingface_hub"
            ) from exc
        self.model = BENDR.from_pretrained(pretrained_repo, n_chans=in_channels,
                                           n_outputs=num_classes, n_times=n_times, sfreq=sfreq)
        if freeze_backbone:
            _freeze_backbone(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, C, T). Returns: logits (batch, num_classes)."""
        return self.model(ensure_3d(x))


def _freeze_backbone(model: nn.Module) -> None:
    """Freeze all params except classifier/final_layer."""
    for name, param in model.named_parameters():
        if not any(k in name for k in ("final_layer", "classification", "classifier")):
            param.requires_grad = False
