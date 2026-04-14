from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

THIS_DIR = Path(__file__).parent
SRC_DIR = THIS_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from data_loader.dataset.factory import create_loader
from model.factory import create_model, list_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CNN EEG model on one batch from the dataloader.")
    parser.add_argument(
        "--model",
        default="baseline_cnn_1d",
        choices=list_models(),
        help="Model name to instantiate.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for the dataloader.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "val", "test"),
        help="Dataset split to use for the sanity-check batch.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
        help="Execution device.",
    )
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes.")
    parser.add_argument("--channels", type=int, default=16, help="Number of EEG channels per sample.")
    parser.add_argument("--samples", type=int, default=128, help="Number of samples per EEG window.")
    parser.add_argument("--sfreq", type=int, default=128, help="Sampling frequency of EEG windows.")
    parser.add_argument("--config-path", default=str(SRC_DIR / "data_loader" / "config.yaml"), help="Path to dataloader config.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze pretrained backbone when supported.")
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_dataloaders(batch_size: int, num_workers: int, config_path: Path):
    datasets = {
        "train": create_loader("cached", config_path=str(config_path), mode="train", cache_memory_mb=2048, augment_data=False),
        "val": create_loader("cached", config_path=str(config_path), mode="val", cache_memory_mb=512, augment_data=False),
        "test": create_loader("cached", config_path=str(config_path), mode="test", cache_memory_mb=512, augment_data=False),
    }

    for split, dataset in datasets.items():
        if len(dataset) == 0:
            output_dir = getattr(dataset, "output_dir", SRC_DIR.parent / "results" / "dataloader")
            raise RuntimeError(
                f"The '{split}' dataset is empty. Expected window index files under {output_dir}. "
                f"Run the dataloader pipeline first and confirm window_index_{split}.csv exists with rows."
            )

    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        for split, dataset in datasets.items()
    }


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config_path = Path(args.config_path).resolve()

    print(f"Loading dataloaders on {device} ...")
    dataloaders = build_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, config_path=config_path)
    batch_x, batch_y = next(iter(dataloaders[args.split]))

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    channels = int(cfg.get("channels", {}).get("target_count", args.channels))
    sfreq = int(cfg.get("signal", {}).get("target_sfreq", args.sfreq))
    window_sec = float(cfg.get("windowing", {}).get("window_sec", 1.0))
    samples = int(window_sec * sfreq)

    model = create_model(
        args.model,
        in_channels=channels,
        num_classes=args.num_classes,
        n_times=samples,
        sfreq=sfreq,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(batch_x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

    print("=" * 72)
    print(f"Model            : {args.model}")
    print(f"Trainable params : {count_parameters(model):,}")
    print(f"Input batch      : {tuple(batch_x.shape)}")
    print(f"Labels           : {tuple(batch_y.shape)}")
    print(f"Logits           : {tuple(logits.shape)}")
    print(f"Predictions      : {tuple(preds.shape)}")
    print(f"Pred class count : {(preds == 1).sum().item()} seizure / {(preds == 0).sum().item()} background")
    print(f"Prob range       : min={probs.min().item():.4f} max={probs.max().item():.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
