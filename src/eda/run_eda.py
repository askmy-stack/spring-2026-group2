# src/run_pipeline.py
from __future__ import annotations

import yaml

from src.eda.eda_pipeline import BIDSEDAPreprocessPipeline


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    cfg = load_yaml("preprocess_eda_config.yaml")
    BIDSEDAPreprocessPipeline(cfg).run()
