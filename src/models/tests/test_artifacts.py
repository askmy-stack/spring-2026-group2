from __future__ import annotations

from pathlib import Path

from models.utils.artifacts import PROJECT_ROOT, SRC_DIR, artifact_filename, artifact_paths


def test_artifact_filename_uses_required_pattern():
    assert artifact_filename("deepconvnet") == "deepconvnet_v1.pt"


def test_artifact_paths_match_outputs_contract():
    paths = artifact_paths("st_eegformer")
    assert str(paths.checkpoint_path).endswith("outputs/models/st_eegformer/st_eegformer_v1.pt")
    assert str(paths.summary_path).endswith("outputs/results/st_eegformer/summary.json")
    assert str(paths.test_metrics_path).endswith("outputs/results/st_eegformer/test_metrics.json")
    assert str(paths.config_path).endswith("outputs/logs/st_eegformer/config.json")
    assert str(paths.history_path).endswith("outputs/logs/st_eegformer/history.csv")


def test_project_root_resolves_to_repo_root():
    assert SRC_DIR.name == "src"
    assert PROJECT_ROOT == SRC_DIR.parent
    assert Path(PROJECT_ROOT / "outputs").name == "outputs"
