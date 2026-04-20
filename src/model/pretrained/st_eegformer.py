from __future__ import annotations

from model.hugging_face.st_eegformer import HuggingFaceSTEEGFormer

from .base import PretrainedInputSpec, validate_pretrained_inputs


class HFSTEEGFormerPretrainedModel(HuggingFaceSTEEGFormer):
    SPEC = PretrainedInputSpec(
        model_id="hf_st_eegformer",
        repo_id="eugenehp/ST-EEGFormer",
        required_sfreq=128,
        required_channels=16,
        recommended_window_sec=6.0,
        max_window_sec=6.0,
        notes="The official ST-EEGFormer model card documents 128 Hz input and up to 6-second windows.",
    )

    def __init__(self, *args, in_channels: int = 16, n_times: int = 768, sfreq: int = 128, **kwargs):
        validate_pretrained_inputs(self.SPEC, in_channels=in_channels, n_times=n_times, sfreq=sfreq)
        super().__init__(*args, in_channels=in_channels, n_times=n_times, sfreq=sfreq, **kwargs)
