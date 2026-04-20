"""
Load Pre-Cached EEG Tensors for Modeling
Portable version — resolves all relative paths from the config file location.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

THIS_DIR    = Path(__file__).parent.resolve()   # src/data_loader/
CONFIG_PATH = THIS_DIR / "config.yaml"


def _resolve_paths_in_config(cfg: dict, base: Path) -> dict:
    """Resolve relative paths in config relative to base (config file dir)."""
    import copy
    cfg = copy.deepcopy(cfg)

    def resolve(val):
        p = Path(val).expanduser()
        if not p.is_absolute():
            p = (base / p).resolve()
        return str(p)

    ds = cfg.get("dataset", {})
    for key in ["raw_root", "results_root", "bids_root", "sample_data_dir"]:
        if key in ds:
            ds[key] = resolve(ds[key])

    cache = cfg.get("caching", {})
    if "disk_cache_dir" in cache:
        cache["disk_cache_dir"] = resolve(cache["disk_cache_dir"])

    export = cfg.get("export", {})
    for key in ["metadata_dir", "reports_dir"]:
        if key in export:
            export[key] = resolve(export[key])

    return cfg


sys.path.insert(0, str(THIS_DIR))
from dataset.factory import create_pytorch_dataloaders


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    augment_train: bool = False,
):
    """
    Returns (train_dl, val_dl, test_dl) backed by the pre-built pkl cache.
    Tensors shape : (batch, 16, 256)
    Labels shape  : (batch,)  — 0=background, 1=seizure
    """
    # Load and resolve all paths relative to config file location
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_paths_in_config(cfg, CONFIG_PATH.parent)

    # Write resolved config to a temp file for create_pytorch_dataloaders
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                     delete=False) as tmp:
        yaml.dump(cfg, tmp, default_flow_style=False)
        tmp_path = tmp.name

    try:
        train_dl, val_dl, test_dl = create_pytorch_dataloaders(
            config_path=tmp_path,
            loader_type="cached",
            augment_train=augment_train,
        )
    finally:
        os.unlink(tmp_path)

    def _rebuild(ds, shuffle):
        return DataLoader(
            ds.dataset if hasattr(ds, "dataset") else ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=False,
            prefetch_factor=1,
        )

    return _rebuild(train_dl, True), _rebuild(val_dl, False), _rebuild(test_dl, False)


def print_summary(train_dl, val_dl, test_dl):
    print("=" * 55)
    print("DataLoader Summary (cache-backed)")
    print("=" * 55)
    for name, dl in [("train", train_dl), ("val", val_dl), ("test", test_dl)]:
        ds  = dl.dataset
        n   = len(ds)
        pos = int((ds.get_labels() == 1).sum()) if hasattr(ds, "get_labels") else "?"
        neg = n - pos if isinstance(pos, int) else "?"
        print(f"  {name:5s} | windows={n:>9,} | seizure={pos:>7} | background={neg:>9}")
    print(f"\n  Tensor shape : (batch, 16, 256)")
    print(f"  Label values : 0 = background,  1 = seizure")
    print("=" * 55)


if __name__ == "__main__":
    print("Loading cached dataloaders ...")
    train_dl, val_dl, test_dl = get_dataloaders(batch_size=32, num_workers=4)
    print_summary(train_dl, val_dl, test_dl)
    print("\nDone.")
