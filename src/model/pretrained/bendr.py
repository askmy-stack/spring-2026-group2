from __future__ import annotations

from torch import nn

from model.blocks import ensure_3d

from .base import PretrainedInputSpec, validate_pretrained_inputs


class BENDRPretrainedModel(nn.Module):
    SPEC = PretrainedInputSpec(
        model_id="bendr_pretrained",
        repo_id="braindecode/braindecode-bendr",
        required_channels=16,
        notes="BENDR can accept varying lengths, but matching the pretraining setup is recommended.",
    )

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
        validate_pretrained_inputs(self.SPEC, in_channels=in_channels, n_times=n_times, sfreq=sfreq)
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

    def forward(self, x):
        x = ensure_3d(x)
        return self.model(x)
