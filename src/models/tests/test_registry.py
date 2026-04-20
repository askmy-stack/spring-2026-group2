from __future__ import annotations

import torch

from models.utils.registry import create_model, list_models
from models.architectures.st_eegformer import HFSTEEGFormerPretrainedModel


def test_selected_models_are_registered():
    names = set(list_models())
    assert "baseline_cnn_1d" in names
    assert "deepconvnet" in names
    assert "enhanced_cnn_1d" in names
    assert "multiscale_attention_cnn" in names
    assert "st_eegformer" in names


def test_create_model_returns_expected_cnn():
    model = create_model("enhanced_cnn_1d", in_channels=16, num_classes=2, dropout=0.2)
    x = torch.randn(2, 16, 256)
    logits = model(x)
    assert logits.shape == (2, 2)


def test_create_model_builds_st_eegformer(monkeypatch):
    def _skip_weights(self, pretrained_repo: str, filename: str):
        self._missing_keys = []

    monkeypatch.setattr(HFSTEEGFormerPretrainedModel, "_load_pretrained_weights", _skip_weights)
    model = create_model("st_eegformer", in_channels=16, num_classes=2, n_times=768, sfreq=128)
    x = torch.randn(1, 16, 768)
    logits = model(x)
    assert logits.shape == (1, 2)


def test_create_model_rejects_unknown_name():
    try:
        create_model("unknown_model")
    except ValueError as exc:
        assert "Unknown model" in str(exc)
    else:
        raise AssertionError("Expected create_model to reject unknown model names.")
