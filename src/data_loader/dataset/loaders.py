from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.base import BaseEEGDataset
from core.io import read_raw
from core.cache import PickleCacher, make_cache_key, CacheStats
from core.augment import augment
from core.signal import normalize_signal

try:
    import dask
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import swifter
    SWIFTER_AVAILABLE = True
except ImportError:
    SWIFTER_AVAILABLE = False


class StandardEEGLoader(BaseEEGDataset):
    def __init__(self, config_path: str = "config.yaml", mode: str = "train"):
        super().__init__(config_path, mode)
        self._load_index()

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data_index.iloc[idx]
        data = self._load_window(row)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(int(row["label"]), dtype=torch.long)

    def _load_window(self, row) -> np.ndarray:
        raw = read_raw(row["path"], preload=True)
        sfreq = raw.info["sfreq"]
        start = int(float(row["start_sec"]) * sfreq)
        end = int(float(row["end_sec"]) * sfreq)
        end = min(end, raw.n_times)
        data, _ = raw[:, start:end]
        data = _pad_or_trim(data, self.n_channels, self.window_samples)
        norm_method = self.cfg.get("signal", {}).get("normalize", None)
        if norm_method:
            data = normalize_signal(data, method=norm_method)
        return data


class CachedEEGLoader(BaseEEGDataset):
    def __init__(
        self,
        config_path: str = "config.yaml",
        mode: str = "train",
        cache_memory_mb: int = 2048,
        augment_data: bool = False,
    ):
        super().__init__(config_path, mode)
        self.cache_memory_mb = cache_memory_mb
        self.do_augment = augment_data and mode == "train"
        self.cacher = PickleCacher(self.cache_dir / "pickle", cache_memory_mb)
        self._load_index()

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data_index.iloc[idx]
        label = int(row["label"])
        cache_key = make_cache_key(self.mode, row["path"], row["start_sec"], row["end_sec"])

        tensor_cached = self.cacher.get(cache_key, ns="tensor")
        if tensor_cached is not None:
            data_t, label_t = tensor_cached
            if self.do_augment:
                data_np = data_t.numpy()
                data_np = augment(data_np, label, self.cfg)
                data_t = torch.from_numpy(data_np).float()
            return data_t, label_t

        raw_cached = self.cacher.get(cache_key, ns="raw")
        if raw_cached is not None:
            data = raw_cached
        else:
            raw = read_raw(row["path"], preload=True)
            sfreq = raw.info["sfreq"]
            start = int(float(row["start_sec"]) * sfreq)
            end = min(int(float(row["end_sec"]) * sfreq), raw.n_times)
            data, _ = raw[:, start:end]
            data = _pad_or_trim(data, self.n_channels, self.window_samples)
            norm_method = self.cfg.get("signal", {}).get("normalize", None)
            if norm_method:
                data = normalize_signal(data, method=norm_method)
            self.cacher.put(cache_key, data, ns="raw")

        if self.do_augment:
            data = augment(data, label, self.cfg)

        data_t = torch.tensor(data, dtype=torch.float32)
        label_t = torch.tensor(label, dtype=torch.long)

        if not self.do_augment:
            self.cacher.put(cache_key, (data_t, label_t), ns="tensor")

        return data_t, label_t

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "hits": self.cacher.stats.hits,
            "misses": self.cacher.stats.misses,
            "hit_rate": self.cacher.stats.hit_rate,
            "memory_usage_mb": self.cacher.memory_usage_mb(),
            "cached_items": self.cacher.cached_items(),
        }

    def clear_cache(self, ns: Optional[str] = None):
        self.cacher.clear(ns)


class ParallelEEGLoader(CachedEEGLoader):
    def __init__(
        self,
        config_path: str = "config.yaml",
        mode: str = "train",
        cache_memory_mb: int = 2048,
        n_workers: int = 4,
        augment_data: bool = False,
    ):
        super().__init__(config_path, mode, cache_memory_mb, augment_data)
        self.use_dask = DASK_AVAILABLE
        self.n_workers = n_workers


class EnhancedEEGLoader(ParallelEEGLoader):
    def __init__(
        self,
        config_path: str = "config.yaml",
        mode: str = "train",
        cache_memory_mb: int = 2048,
        n_workers: int = 4,
        augment_data: bool = False,
    ):
        super().__init__(config_path, mode, cache_memory_mb, n_workers, augment_data)
        self.use_swifter = SWIFTER_AVAILABLE

    def benchmark_cache(self, num_samples: int = 50) -> Dict[str, Any]:
        n = min(num_samples, len(self))
        if n == 0:
            return {"error": "empty dataset"}
        self.cacher.stats.reset()

        t0 = time.time()
        for i in range(n):
            _ = self[i]
        cold = time.time() - t0

        self.cacher.stats.reset()
        t0 = time.time()
        for i in range(n):
            _ = self[i]
        warm = time.time() - t0

        speedup = cold / warm if warm > 0 else 0.0
        return {"num_samples": n, "cold_time": cold, "warm_time": warm, "speedup": speedup}


def _pad_or_trim(data: np.ndarray, n_channels: int, n_samples: int) -> np.ndarray:
    c, t = data.shape
    if c < n_channels:
        pad_rows = np.zeros((n_channels - c, t))
        data = np.vstack([data, pad_rows])
    elif c > n_channels:
        data = data[:n_channels]

    if t < n_samples:
        pad_cols = np.zeros((n_channels, n_samples - t))
        data = np.hstack([data, pad_cols])
    elif t > n_samples:
        data = data[:, :n_samples]

    return data