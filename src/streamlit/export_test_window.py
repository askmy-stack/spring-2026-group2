from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader.dataset.factory import create_loader


def main() -> int:
    parser = argparse.ArgumentParser(description="Export one cached EEG window for Streamlit inference testing.")
    parser.add_argument("--config-path", default=str(SRC_DIR / "data_loader" / "config.yaml"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default=str(SRC_DIR / "streamlit" / "sample_window.npy"))
    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset = create_loader(
        "cached",
        config_path=str(config_path),
        mode=args.split,
        cache_memory_mb=512,
        augment_data=False,
    )
    x, y = dataset[args.index]
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, x.numpy())

    channels = int(cfg.get("channels", {}).get("target_count", x.shape[0]))
    sfreq = int(cfg.get("signal", {}).get("target_sfreq", 0))
    window_sec = float(cfg.get("windowing", {}).get("window_sec", 0))

    print(f"Saved sample window: {output_path}")
    print(f"Split             : {args.split}")
    print(f"Dataset index     : {args.index}")
    print(f"Label             : {int(y.item())}")
    print(f"Shape             : {tuple(x.shape)}")
    print(f"Config channels   : {channels}")
    print(f"Config sfreq      : {sfreq}")
    print(f"Config window_sec : {window_sec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
