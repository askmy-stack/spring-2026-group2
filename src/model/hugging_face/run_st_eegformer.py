from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).parent
SRC_DIR = THIS_DIR.parent.parent
DEFAULT_CONFIG = SRC_DIR / "data_loader" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check Hugging Face ST-EEGFormer on one batch.")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG), help="Path to dataloader config.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"), help="Dataset split to inspect.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze encoder weights.")
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        "-m",
        "model.run",
        "--model",
        "hf_st_eegformer",
        "--config-path",
        str(Path(args.config_path)),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--split",
        args.split,
    ]
    if args.freeze_backbone:
        cmd.append("--freeze-backbone")
    raise SystemExit(subprocess.run(cmd, cwd=SRC_DIR).returncode)


if __name__ == "__main__":
    main()
