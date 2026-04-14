from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).parent
SRC_DIR = THIS_DIR.parent.parent
DEFAULT_CONFIG = SRC_DIR / "data_loader" / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Hugging Face ST-EEGFormer fine-tuning with validation threshold tuning and one final test pass."
    )
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG), help="Path to dataloader config.")
    parser.add_argument("--epochs", type=int, default=5, help="Total training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--max-train-batches", type=int, default=0, help="Optional train batch cap. 0 means full train split.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Optional validation batch cap. 0 means full validation split.")
    parser.add_argument("--max-test-batches", type=int, default=0, help="Optional test batch cap. 0 means full test split.")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Positive-class alpha for focal loss.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma.")
    parser.add_argument("--run-name", default="hf_small", help="Optional run-name suffix.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze encoder and train only the head.")
    return parser.parse_args()


def main():
    args = parse_args()
    cmd = [
        sys.executable,
        "-m",
        "model.train",
        "--model",
        "hf_st_eegformer",
        "--config-path",
        str(Path(args.config_path)),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--loss",
        "focal",
        "--focal-alpha",
        str(args.focal_alpha),
        "--focal-gamma",
        str(args.focal_gamma),
        "--threshold-mode",
        "tune",
        "--save-every",
        "1",
        "--run-name",
        args.run_name,
    ]
    if args.max_train_batches:
        cmd.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_val_batches:
        cmd.extend(["--max-val-batches", str(args.max_val_batches)])
    if args.max_test_batches:
        cmd.extend(["--max-test-batches", str(args.max_test_batches)])
    if args.freeze_backbone:
        cmd.append("--freeze-backbone")
    raise SystemExit(subprocess.run(cmd, cwd=SRC_DIR).returncode)


if __name__ == "__main__":
    main()
