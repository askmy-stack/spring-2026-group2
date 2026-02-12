# src/run_pipeline.py
from __future__ import annotations

from io_utils import load_yaml_config
from eda_preprocess_pipeline import EDAPreprocessPipeline


if __name__ == "__main__":
    # YAML-only config (no classes)
    cfg = load_yaml_config("preprocess_eda_config.yaml")

    # Run pipeline
    EDAPreprocessPipeline(cfg).run()
