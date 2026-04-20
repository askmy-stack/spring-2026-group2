from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parents[1]
PROJECT_ROOT = SRC_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"
LOGS_DIR = OUTPUTS_DIR / "logs"


@dataclass(frozen=True)
class ModelArtifactPaths:
    model_name: str
    checkpoint_path: Path
    results_dir: Path
    logs_dir: Path
    summary_path: Path
    test_metrics_path: Path
    config_path: Path
    history_path: Path


def artifact_filename(model_name: str, version: str = "v1") -> str:
    return f"{model_name}_{version}.pt"


def artifact_paths(model_name: str, version: str = "v1") -> ModelArtifactPaths:
    return ModelArtifactPaths(
        model_name=model_name,
        checkpoint_path=MODELS_DIR / model_name / artifact_filename(model_name, version),
        results_dir=RESULTS_DIR / model_name,
        logs_dir=LOGS_DIR / model_name,
        summary_path=RESULTS_DIR / model_name / "summary.json",
        test_metrics_path=RESULTS_DIR / model_name / "test_metrics.json",
        config_path=LOGS_DIR / model_name / "config.json",
        history_path=LOGS_DIR / model_name / "history.csv",
    )


def ensure_artifact_dirs(model_name: str, version: str = "v1") -> ModelArtifactPaths:
    paths = artifact_paths(model_name=model_name, version=version)
    paths.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    return paths


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def list_available_artifacts() -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    if not MODELS_DIR.exists():
        return discovered
    for model_dir in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir()):
        checkpoint_candidates = sorted(model_dir.glob("*.pt"))
        if not checkpoint_candidates:
            continue
        model_name = model_dir.name
        paths = artifact_paths(model_name)
        discovered.append(
            {
                "model_name": model_name,
                "checkpoint_path": str(checkpoint_candidates[0]),
                "summary": load_json_if_exists(paths.summary_path),
                "test_metrics": load_json_if_exists(paths.test_metrics_path),
                "config": load_json_if_exists(paths.config_path),
                "history_path": str(paths.history_path) if paths.history_path.exists() else None,
            }
        )
    return discovered
