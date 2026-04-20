from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch
import yaml
from torch import nn

from models.utils import artifacts as artifacts_module
from models.utils import train_eval
from streamlit import app as streamlit_app


class _TinyNet(nn.Module):
    def __init__(self, in_channels: int = 16, num_classes: int = 2, **_: object):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 8, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_batches() -> list[tuple[torch.Tensor, torch.Tensor]]:
    x1 = torch.randn(4, 16, 8)
    y1 = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    x2 = torch.randn(4, 16, 8)
    y2 = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    return [(x1, y1), (x2, y2)]


def test_train_model_writes_submission_artifacts(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "channels": {"target_count": 16},
                "signal": {"target_sfreq": 8},
                "windowing": {"window_sec": 1.0},
            },
            f,
        )

    def _fake_get_dataloaders(**_: object):
        batches = _make_batches()
        return batches, batches, batches

    def _fake_create_model(name: str, **kwargs: object):
        assert name == "enhanced_cnn_1d"
        return _TinyNet(**kwargs)

    def _fake_ensure_artifact_dirs(model_name: str, version: str = "v1"):
        base = tmp_path / "outputs"
        return artifacts_module.ModelArtifactPaths(
            model_name=model_name,
            checkpoint_path=base / "models" / model_name / f"{model_name}_{version}.pt",
            results_dir=base / "results" / model_name,
            logs_dir=base / "logs" / model_name,
            summary_path=base / "results" / model_name / "summary.json",
            test_metrics_path=base / "results" / model_name / "test_metrics.json",
            config_path=base / "logs" / model_name / "config.json",
            history_path=base / "logs" / model_name / "history.csv",
        )

    monkeypatch.setattr(train_eval, "get_dataloaders", _fake_get_dataloaders)
    monkeypatch.setattr(train_eval, "create_model", _fake_create_model)
    monkeypatch.setattr(train_eval, "ensure_artifact_dirs", _fake_ensure_artifact_dirs)

    args = Namespace(
        run_name="workflow_test",
        epochs=1,
        batch_size=4,
        num_workers=0,
        lr=1e-3,
        weight_decay=1e-4,
        dropout=0.1,
        device="cpu",
        seed=42,
        patience=5,
        save_every=1,
        channels=16,
        samples=8,
        sfreq=8,
        config_path=str(config_path),
        num_classes=2,
        freeze_backbone=False,
        train_augment=False,
        loss="ce",
        focal_gamma=2.0,
        focal_alpha=0.25,
        threshold_mode="fixed",
        decision_threshold=0.5,
        smoothing_mode="none",
        smoothing_window=3,
        min_positive_run=1,
        log_interval=0,
        eval_log_interval=0,
        max_train_batches=0,
        max_val_batches=0,
        max_test_batches=0,
        grad_clip_norm=0.0,
    )

    exit_code = train_eval.train_model("enhanced_cnn_1d", args)
    assert exit_code == 0

    model_name = "enhanced_cnn_1d__workflow_test"
    paths = _fake_ensure_artifact_dirs(model_name)
    assert paths.checkpoint_path.exists()
    assert paths.summary_path.exists()
    assert paths.test_metrics_path.exists()
    assert paths.config_path.exists()
    assert paths.history_path.exists()


def test_list_available_artifacts_reads_saved_outputs(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    model_name = "st_eegformer"
    (outputs / "models" / model_name).mkdir(parents=True)
    (outputs / "results" / model_name).mkdir(parents=True)
    (outputs / "logs" / model_name).mkdir(parents=True)
    (outputs / "models" / model_name / f"{model_name}_v1.pt").write_bytes(b"checkpoint")
    (outputs / "results" / model_name / "summary.json").write_text('{"best_val_f1": 0.12}')
    (outputs / "results" / model_name / "test_metrics.json").write_text('{"f1": 0.15, "auroc": 0.82}')
    (outputs / "logs" / model_name / "config.json").write_text('{"channels": 16, "samples": 768, "sfreq": 128}')

    monkeypatch.setattr(artifacts_module, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(artifacts_module, "MODELS_DIR", outputs / "models")
    monkeypatch.setattr(artifacts_module, "RESULTS_DIR", outputs / "results")
    monkeypatch.setattr(artifacts_module, "LOGS_DIR", outputs / "logs")

    discovered = artifacts_module.list_available_artifacts()
    assert len(discovered) == 1
    assert discovered[0]["model_name"] == model_name
    assert discovered[0]["test_metrics"]["f1"] == 0.15
    assert discovered[0]["config"]["samples"] == 768


def test_find_window_index_dir_prefers_project_results(tmp_path, monkeypatch):
    project_results = tmp_path / "results" / "dataloader"
    src_results = tmp_path / "src" / "results" / "dataloader"
    project_results.mkdir(parents=True)
    src_results.mkdir(parents=True)

    monkeypatch.setattr(streamlit_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(streamlit_app, "SRC_DIR", tmp_path / "src")

    resolved = streamlit_app._find_window_index_dir()
    assert resolved == project_results


def test_find_window_index_dir_falls_back_to_src_results(tmp_path, monkeypatch):
    src_results = tmp_path / "src" / "results" / "dataloader"
    src_results.mkdir(parents=True)

    monkeypatch.setattr(streamlit_app, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(streamlit_app, "SRC_DIR", tmp_path / "src")

    resolved = streamlit_app._find_window_index_dir()
    assert resolved == src_results
