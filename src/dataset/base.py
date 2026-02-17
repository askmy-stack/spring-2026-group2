from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


DEFAULT_CONFIG: Dict[str, Any] = {
    "signal": {"target_sfreq": 256, "bandpass": [1.0, 50.0], "notch": 60.0, "reference": "average"},
    "windowing": {"window_sec": 1.0, "stride_sec": 1.0},
    "balance": {"seizure_ratio": 0.3},
    "channels": {"target_count": 16},
    "dataset": {"raw_root": "data/raw", "bids_root": "results/bids_dataset", "results_root": "results"},
    "caching": {"enable": True, "max_memory_mb": 2048, "disk_cache_dir": "results/cache"},
    "augmentation": {"enable": False},
}


def load_config(config_path: str | Path) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.exists():
        return DEFAULT_CONFIG
    with open(p) as f:
        return yaml.safe_load(f)


class BaseEEGDataset(Dataset, ABC):
    def __init__(self, config_path: str = "config.yaml", mode: str = "train"):
        super().__init__()
        self.config_path = Path(config_path)
        self.mode = mode
        self.cfg = load_config(config_path)
        self._init_paths()
        self.data_index: pd.DataFrame = pd.DataFrame()

    def _init_paths(self):
        ds = self.cfg.get("dataset", {})
        self.raw_root = Path(ds.get("raw_root", "data/raw"))
        self.bids_root = Path(ds.get("bids_root", "results/bids_dataset"))
        self.results_root = Path(ds.get("results_root", "results"))
        self.output_dir = self.results_root / "dataloader"
        cache_dir = self.cfg.get("caching", {}).get("disk_cache_dir", "results/cache")
        self.cache_dir = Path(cache_dir)
        for p in [self.bids_root, self.output_dir, self.cache_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def _load_index(self):
        idx = self.output_dir / f"window_index_{self.mode}.csv"
        if idx.exists():
            self.data_index = pd.read_csv(idx)

    @property
    def target_sfreq(self) -> int:
        return int(self.cfg.get("signal", {}).get("target_sfreq", 256))

    @property
    def window_samples(self) -> int:
        w = float(self.cfg.get("windowing", {}).get("window_sec", 1.0))
        return int(w * self.target_sfreq)

    @property
    def n_channels(self) -> int:
        return int(self.cfg.get("channels", {}).get("target_count", 16))

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def get_class_weights(self) -> torch.Tensor:
        if self.data_index.empty or "label" not in self.data_index.columns:
            return torch.tensor([1.0, 1.0])
        counts = self.data_index["label"].value_counts().sort_index()
        if len(counts) < 2:
            return torch.tensor([1.0, 1.0])
        total = len(self.data_index)
        w = total / (len(counts) * counts.values)
        w = w / w.sum() * len(counts)
        return torch.tensor(w, dtype=torch.float32)

    def get_summary(self) -> Dict[str, Any]:
        if self.data_index.empty:
            return {"mode": self.mode, "total_windows": 0, "labels": {}, "n_subjects": 0,
                    "target_sfreq": self.target_sfreq, "window_sec": float(self.cfg.get("windowing", {}).get("window_sec", 1.0)),
                    "n_channels": self.n_channels}
        labels = self.data_index["label"].value_counts().to_dict() if "label" in self.data_index.columns else {}
        n_subjects = self.data_index["subject_id"].nunique() if "subject_id" in self.data_index.columns else 0
        return {
            "mode": self.mode,
            "total_windows": len(self.data_index),
            "target_sfreq": self.target_sfreq,
            "window_sec": float(self.cfg.get("windowing", {}).get("window_sec", 1.0)),
            "n_channels": self.n_channels,
            "labels": labels,
            "n_subjects": n_subjects,
        }