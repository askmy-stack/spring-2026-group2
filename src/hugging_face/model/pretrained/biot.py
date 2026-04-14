from __future__ import annotations

from torch import nn

from model.blocks import ensure_3d

from .base import PretrainedInputSpec, validate_pretrained_inputs


class BIOTPretrainedModel(nn.Module):
    SPEC = PretrainedInputSpec(
        model_id="biot_pretrained",
        repo_id="braindecode/biot-pretrained-prest-16chs",
        required_sfreq=200,
        required_channels=16,
        recommended_window_sec=1.0,
        notes="The official BIOT Prest checkpoint is shape-sensitive and should use the BIOT-specific config.",
    )

    def __init__(
        self,
        in_channels: int = 16,
        num_classes: int = 2,
        n_times: int = 200,
        sfreq: int = 200,
        freeze_backbone: bool = False,
        pretrained_repo: str = "braindecode/biot-pretrained-prest-16chs",
    ):
        super().__init__()
        validate_pretrained_inputs(self.SPEC, in_channels=in_channels, n_times=n_times, sfreq=sfreq)
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

    def forward(self, x):
        x = ensure_3d(x)
        return self.model(x)
