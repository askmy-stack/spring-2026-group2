from __future__ import annotations

from .base import PretrainedInputSpec
from .bendr import BENDRPretrainedModel
from .biot import BIOTPretrainedModel
from .eegpt import EEGPTPretrainedModel
from .st_eegformer import HFSTEEGFormerPretrainedModel


PRETRAINED_SPECS: dict[str, PretrainedInputSpec] = {
    "bendr_pretrained": BENDRPretrainedModel.SPEC,
    "biot_pretrained": BIOTPretrainedModel.SPEC,
    "eegpt_pretrained": EEGPTPretrainedModel.SPEC,
    "hf_st_eegformer": HFSTEEGFormerPretrainedModel.SPEC,
    "st_eegformer": HFSTEEGFormerPretrainedModel.SPEC,
}


def list_pretrained_specs() -> dict[str, PretrainedInputSpec]:
    return PRETRAINED_SPECS.copy()
