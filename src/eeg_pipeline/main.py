# src/eeg_pipeline/main.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from eeg_pipeline.core.yaml_utils import load_yaml
from eeg_pipeline.pipeline.run_pipeline import run_from_dataloader_index


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run EEG Pipeline (window-level preprocessing/QC/EDA/overview + optional BIDS export)."
    )
    ap.add_argument(
        "--config",
        default="src/eeg_pipeline/configs/config.yaml",
        help="Path to EEG pipeline YAML config.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg: Dict[str, Any] = load_yaml(str(cfg_path))
    run_from_dataloader_index(cfg)


if __name__ == "__main__":
    main()
