# src/run_pipeline.py
from __future__ import annotations

from pathlib import Path
import yaml

from pipeline import BIDSEDAPreprocessPipeline


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    cfg = load_yaml("preprocess_eda_config.yaml")
    BIDSEDAPreprocessPipeline(cfg).run()
