"""BIOT pretrained model wrapper. Requires: pip install braindecode[hug] huggingface_hub"""
from __future__ import annotations
import logging
import torch
from torch import nn
from ...modules.hf_blocks import ensure_3d
from .base import PretrainedInputSpec, validate_pretrained_inputs

logger = logging.getLogger(__name__)


class BIOTPretrainedModel(nn.Module):
    """
    BIOT loaded from braindecode/biot-pretrained-prest-16chs on HuggingFace Hub.

    Note: BIOT requires sfreq=200 and in_channels=16.

    Args:
        in_channels: Must be 16 (default: 16).
        num_classes: Output classes (default: 2).
        n_times: Timesteps — must match sfreq * window_sec (default: 200).
        sfreq: Must be 200 (default: 200).
        freeze_backbone: Freeze encoder weights if True (default: False).
        pretrained_repo: HF repo id.

    Example:
        >>> model = BIOTPretrainedModel(in_channels=16, sfreq=200, n_times=200)
    """

    SPEC = PretrainedInputSpec(
        model_id="biot_pretrained",
        repo_id="braindecode/biot-pretrained-prest-16chs",
        required_sfreq=200,
        required_channels=16,
        recommended_window_sec=1.0,
        notes="BIOT Prest checkpoint is shape-sensitive; use BIOT-specific config.",
    )

    def __init__(self, in_channels: int = 16, num_classes: int = 2,
                 n_times: int = 200, sfreq: int = 200,
                 freeze_backbone: bool = False,
                 pretrained_repo: str = "braindecode/biot-pretrained-prest-16chs"):
        super().__init__()
        validate_pretrained_inputs(self.SPEC, in_channels=in_channels, n_times=n_times, sfreq=sfreq)
        try:
            from braindecode.models import BIOT
        except ImportError as exc:
            raise ImportError(
                "BIOT requires braindecode[hug] and huggingface_hub. "
                "Install: pip install 'braindecode[hug]' huggingface_hub"
            ) from exc
        self.model = BIOT.from_pretrained(pretrained_repo, n_chans=in_channels,
                                          n_outputs=num_classes, n_times=n_times, sfreq=sfreq)
        if freeze_backbone:
            _freeze_backbone(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, C, T). Returns: logits (batch, num_classes)."""
        return self.model(ensure_3d(x))


def _freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not any(k in name for k in ("final_layer", "classification", "classifier")):
            param.requires_grad = False
