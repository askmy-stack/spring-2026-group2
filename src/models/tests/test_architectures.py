from __future__ import annotations

import torch

from models.architectures.baseline_cnn_1d import BaselineCNN1D
from models.architectures.deepconvnet import DeepConvNet
from models.architectures.enhanced_cnn_1d import EnhancedCNN1D
from models.architectures.multiscale_attention_cnn import MultiScaleAttentionCNN
from models.architectures.st_eegformer import HFSTEEGFormerPretrainedModel


def test_baseline_cnn_forward_shape():
    model = BaselineCNN1D(in_channels=16, num_classes=2, dropout=0.1)
    x = torch.randn(4, 16, 256)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_deepconvnet_forward_shape():
    model = DeepConvNet(in_channels=16, num_classes=2, dropout=0.1)
    x = torch.randn(4, 16, 256)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_enhanced_cnn_forward_shape():
    model = EnhancedCNN1D(in_channels=16, num_classes=2, dropout=0.1)
    x = torch.randn(4, 16, 256)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_multiscale_attention_cnn_forward_shape():
    model = MultiScaleAttentionCNN(in_channels=16, num_classes=2, dropout=0.1)
    x = torch.randn(4, 16, 256)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_st_eegformer_forward_shape_without_hf_download(monkeypatch):
    def _skip_weights(self, pretrained_repo: str, filename: str):
        self._missing_keys = []

    monkeypatch.setattr(HFSTEEGFormerPretrainedModel, "_load_pretrained_weights", _skip_weights)
    model = HFSTEEGFormerPretrainedModel(
        in_channels=16,
        num_classes=2,
        n_times=768,
        sfreq=128,
        freeze_backbone=False,
    )
    x = torch.randn(2, 16, 768)
    logits = model(x)
    assert logits.shape == (2, 2)


def test_st_eegformer_rejects_wrong_sampling_rate(monkeypatch):
    def _skip_weights(self, pretrained_repo: str, filename: str):
        self._missing_keys = []

    monkeypatch.setattr(HFSTEEGFormerPretrainedModel, "_load_pretrained_weights", _skip_weights)
    try:
        HFSTEEGFormerPretrainedModel(in_channels=16, num_classes=2, n_times=768, sfreq=256)
    except ValueError as exc:
        assert "128 Hz" in str(exc)
    else:
        raise AssertionError("Expected ST-EEGFormer to reject non-128 Hz input.")
