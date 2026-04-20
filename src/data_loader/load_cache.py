"""
Load Pre-Cached EEG Tensors for Modeling
==========================================
Loads tensors directly from the pre-built disk cache.
No EDF files are read — every __getitem__ hits the .pkl cache.

Run from src/data_loader on the server:
    cd /home/amir/Desktop/GWU/Research/EEG/src/data_loader
    python3 load_cache.py

To use in your model training script, import get_dataloaders() directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).parent          # src/data_loader
sys.path.insert(0, str(THIS_DIR))

from dataset.factory import create_pytorch_dataloaders

CONFIG_PATH = str(THIS_DIR / "config.yaml")


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    augment_train: bool = False,   # set True to enable augmentation during training
):
    """
    Returns (train_dl, val_dl, test_dl) backed by the pre-built pkl cache.
    Tensors shape : (batch, 16, 256)
    Labels shape  : (batch,)  — 0=background, 1=seizure
    """
    train_dl, val_dl, test_dl = create_pytorch_dataloaders(
        config_path=CONFIG_PATH,
        loader_type="cached",       # uses CachedEEGLoader → reads from .pkl files
        augment_train=augment_train,
    )

    # Override batch_size / num_workers if caller specified them
    def _rebuild(ds, shuffle):
        return DataLoader(
            ds.dataset if hasattr(ds, "dataset") else ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    train_dl = _rebuild(train_dl, shuffle=True)
    val_dl   = _rebuild(val_dl,   shuffle=False)
    test_dl  = _rebuild(test_dl,  shuffle=False)

    return train_dl, val_dl, test_dl


def print_summary(train_dl, val_dl, test_dl):
    print("=" * 55)
    print("DataLoader Summary (cache-backed)")
    print("=" * 55)
    for name, dl in [("train", train_dl), ("val", val_dl), ("test", test_dl)]:
        ds = dl.dataset
        n  = len(ds)
        pos = int((ds.get_labels() == 1).sum()) if hasattr(ds, "get_labels") else "?"
        neg = n - pos if isinstance(pos, int) else "?"
        print(f"  {name:5s} | windows={n:>9,} | seizure={pos:>7} | background={neg:>9}")
    print(f"\n  Tensor shape : (batch, 16, 256)  [channels x samples]")
    print(f"  Label values : 0 = background,  1 = seizure")
    print("=" * 55)


def demo_batch(dl: DataLoader, split: str):
    """Grab one batch and print shape + label distribution."""
    x, y = next(iter(dl))
    n_pos = (y == 1).sum().item()
    n_neg = (y == 0).sum().item()
    print(f"\n  [{split}] batch x={tuple(x.shape)}  y={tuple(y.shape)}  "
          f"seizure={n_pos}  background={n_neg}  "
          f"x.min={x.min():.4f}  x.max={x.max():.4f}")


if __name__ == "__main__":
    print("Loading cached dataloaders ...")
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=32, num_workers=4)

    print_summary(train_dl, val_dl, test_dl)

    print("\nSample batches:")
    demo_batch(train_dl, "train")
    demo_batch(val_dl,   "val")
    demo_batch(test_dl,  "test")

    print("\nDone. Use get_dataloaders() in your training script.")