"""EEGPT pretrained model wrapper. Requires: pip install --upgrade braindecode[hug] huggingface_hub"""
from __future__ import annotations
import logging
import torch
from torch import nn
from ...modules.hf_blocks import ensure_3d
from .base import PretrainedInputSpec, validate_pretrained_inputs

logger = logging.getLogger(__name__)


class EEGPTPretrainedModel(nn.Module):
    """
    EEGPT loaded from braindecode/eegpt-pretrained on HuggingFace Hub.

    Args:
        in_channels: Must be 16 (default: 16).
        num_classes: Output classes (default: 2).
        n_times: Timesteps per window (default: 256).
        sfreq: Sampling frequency (default: 256).
        freeze_backbone: Freeze encoder weights if True (default: False).
        pretrained_repo: HF repo id.

    Example:
        >>> model = EEGPTPretrainedModel(in_channels=16)
    """

    SPEC = PretrainedInputSpec(
        model_id="eegpt_pretrained",
        repo_id="braindecode/eegpt-pretrained",
        required_channels=16,
        notes="Validate against chosen checkpoint setup before fine-tuning.",
    )

    def __init__(self, in_channels: int = 16, num_classes: int = 2,
                 n_times: int = 256, sfreq: int = 256,
                 freeze_backbone: bool = False,
                 pretrained_repo: str = "braindecode/eegpt-pretrained"):
        super().__init__()
        validate_pretrained_inputs(self.SPEC, in_channels=in_channels, n_times=n_times, sfreq=sfreq)
        try:
            from braindecode.models import EEGPT
        except ImportError as exc:
            raise ImportError(
                "EEGPT requires an updated braindecode[hug] + huggingface_hub. "
                "Install: pip install --upgrade 'braindecode[hug]' huggingface_hub"
            ) from exc
        # chs_info=None overrides the 64-ch montage baked into the pretrained config,
        # letting us fine-tune on arbitrary in_channels (16 for CHB-MIT).
        self.model = EEGPT.from_pretrained(pretrained_repo, n_chans=in_channels,
                                           n_outputs=num_classes, n_times=n_times,
                                           sfreq=sfreq, chs_info=None)
        if freeze_backbone:
            _freeze_backbone(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (batch, C, T). Returns: logits (batch, num_classes)."""
        return self.model(ensure_3d(x))


def _freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not any(k in name for k in ("final_layer", "classification", "classifier")):
            param.requires_grad = False
