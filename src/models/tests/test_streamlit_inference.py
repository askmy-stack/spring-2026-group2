from __future__ import annotations

import numpy as np
import torch
from torch import nn

from streamlit import app as streamlit_app


class _ConstantLogitModel(nn.Module):
    def __init__(self, logits: tuple[float, float]):
        super().__init__()
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._logits.expand(x.shape[0], -1)


def test_predict_single_window_returns_seizure(monkeypatch):
    monkeypatch.setattr(
        streamlit_app,
        "load_inference_bundle",
        lambda model_name: {
            "model": _ConstantLogitModel((0.1, 2.0)),
            "threshold": 0.55,
            "config": {"model": model_name},
        },
    )
    prepared = np.zeros((16, 256), dtype=np.float32)
    result = streamlit_app.predict_single_window("enhanced_cnn_1d", prepared)
    assert result["prediction"] == "seizure"
    assert result["seizure_prob"] > result["threshold"]


def test_predict_single_window_returns_background(monkeypatch):
    monkeypatch.setattr(
        streamlit_app,
        "load_inference_bundle",
        lambda model_name: {
            "model": _ConstantLogitModel((2.0, 0.1)),
            "threshold": 0.85,
            "config": {"model": model_name},
        },
    )
    prepared = np.zeros((16, 256), dtype=np.float32)
    result = streamlit_app.predict_single_window("multiscale_attention_cnn", prepared)
    assert result["prediction"] == "background"
    assert result["seizure_prob"] < result["threshold"]
