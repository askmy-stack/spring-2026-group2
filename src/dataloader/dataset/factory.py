from __future__ import annotations

from typing import Any

from dataset.loaders import (
    StandardEEGLoader,
    CachedEEGLoader,
    ParallelEEGLoader,
    EnhancedEEGLoader,
    BaseEEGDataset,
)


_LOADERS = {
    "standard": StandardEEGLoader,
    "cached": CachedEEGLoader,
    "parallel": ParallelEEGLoader,
    "enhanced": EnhancedEEGLoader,
}


def create_loader(loader_type: str = "enhanced", **kwargs) -> BaseEEGDataset:
    if loader_type not in _LOADERS:
        raise ValueError(f"Unknown loader_type '{loader_type}'. Choose from: {list(_LOADERS)}")
    return _LOADERS[loader_type](**kwargs)


def create_pytorch_dataloaders(
    config_path: str = "config.yaml",
    loader_type: str = "enhanced",
    augment_train: bool = True,
):
    import torch
    from torch.utils.data import DataLoader

    import yaml
    from pathlib import Path

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pt = cfg.get("pytorch", {})

    batch_size = int(pt.get("batch_size", 32))
    num_workers = int(pt.get("num_workers", 4))
    pin_memory = bool(pt.get("pin_memory", True))
    drop_last = bool(pt.get("drop_last", False))

    cache_mb = int(cfg.get("caching", {}).get("max_memory_mb", 2048))

    train_ds = create_loader(
        loader_type,
        config_path=config_path,
        mode="train",
        cache_memory_mb=cache_mb,
        augment_data=augment_train,
    )
    val_ds = create_loader(
        loader_type,
        config_path=config_path,
        mode="val",
        cache_memory_mb=cache_mb // 4,
        augment_data=False,
    )
    test_ds = create_loader(
        loader_type,
        config_path=config_path,
        mode="test",
        cache_memory_mb=cache_mb // 4,
        augment_data=False,
    )

    prefetch = int(pt.get("prefetch_factor", 2)) if num_workers > 0 else None
    persistent = bool(pt.get("persistent_workers", True)) and num_workers > 0

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=bool(pt.get("shuffle_train", True)),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch,
        persistent_workers=persistent,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=prefetch,
        persistent_workers=persistent,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=prefetch,
        persistent_workers=persistent,
    )

    return train_dl, val_dl, test_dl