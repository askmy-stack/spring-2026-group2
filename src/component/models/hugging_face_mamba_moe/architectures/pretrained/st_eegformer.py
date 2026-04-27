"""ST-EEGFormer pretrained wrapper. Requires: pip install huggingface_hub safetensors"""
from __future__ import annotations
from ...architectures.hf_cnn_models import STEEGFormerPretrained  # noqa: F401 — re-export alias
from .base import PretrainedInputSpec, validate_pretrained_inputs


class HFSTEEGFormerPretrainedModel(STEEGFormerPretrained):
    """
    ST-EEGFormer with SPEC-validated inputs for registry integration.

    Constraints: sfreq=128, in_channels=16, max_window=6s.

    Args:
        in_channels: Must be 16 (default: 16).
        n_times: Timesteps — must be ≤ 128*6=768 (default: 768).
        sfreq: Must be 128 (default: 128).

    Example:
        >>> model = HFSTEEGFormerPretrainedModel(in_channels=16, n_times=768, sfreq=128)
    """

    SPEC = PretrainedInputSpec(
        model_id="hf_st_eegformer",
        repo_id="eugenehp/ST-EEGFormer",
        required_sfreq=128,
        required_channels=16,
        recommended_window_sec=6.0,
        max_window_sec=6.0,
        notes="Official model card: 128 Hz, up to 6-second windows.",
    )

    def __init__(self, *args, in_channels: int = 16, n_times: int = 768, sfreq: int = 128, **kwargs):
        validate_pretrained_inputs(self.SPEC, in_channels=in_channels, n_times=n_times, sfreq=sfreq)
        super().__init__(*args, in_channels=in_channels, n_times=n_times, sfreq=sfreq, **kwargs)
