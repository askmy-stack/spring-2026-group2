from __future__ import annotations

import hashlib
import pickle
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import dask
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    total_load_time: float = 0.0
    cached_load_time: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset(self):
        self.hits = 0
        self.misses = 0
        self.total_load_time = 0.0
        self.cached_load_time = 0.0


class BaseCacher(ABC):
    @abstractmethod
    def get(self, key: str, ns: str = "default") -> Optional[Any]:
        pass

    @abstractmethod
    def put(self, key: str, value: Any, ns: str = "default"):
        pass

    @abstractmethod
    def exists(self, key: str, ns: str = "default") -> bool:
        pass

    @abstractmethod
    def clear(self, ns: Optional[str] = None):
        pass


class PickleCacher(BaseCacher):
    def __init__(self, cache_dir: Path, memory_limit_mb: int = 2048):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self._mem: OrderedDict[str, Any] = OrderedDict()
        self._mem_usage = 0
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def _disk_path(self, key: str, ns: str) -> Path:
        ns_dir = self.cache_dir / ns
        ns_dir.mkdir(exist_ok=True)
        return ns_dir / f"{key}.pkl"

    def _obj_size(self, obj: Any) -> int:
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 0

    def _mem_key(self, key: str, ns: str) -> str:
        return f"{ns}:{key}"

    def exists(self, key: str, ns: str = "default") -> bool:
        return self._mem_key(key, ns) in self._mem or self._disk_path(key, ns).exists()

    def get(self, key: str, ns: str = "default") -> Optional[Any]:
        t0 = time.time()
        mkey = self._mem_key(key, ns)

        with self._lock:
            if mkey in self._mem:
                self._mem.move_to_end(mkey)
                self.stats.hits += 1
                self.stats.cached_load_time += time.time() - t0
                return self._mem[mkey]

        disk = self._disk_path(key, ns)
        if disk.exists():
            try:
                with open(disk, "rb") as f:
                    obj = pickle.load(f)
                with self._lock:
                    self._mem_add(mkey, obj)
                    self.stats.hits += 1
                    self.stats.cached_load_time += time.time() - t0
                return obj
            except Exception:
                disk.unlink(missing_ok=True)

        with self._lock:
            self.stats.misses += 1
        return None

    def put(self, key: str, value: Any, ns: str = "default"):
        disk = self._disk_path(key, ns)
        try:
            with open(disk, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return
        mkey = self._mem_key(key, ns)
        with self._lock:
            self._mem_add(mkey, value)

    def _mem_add(self, mkey: str, obj: Any):
        size = self._obj_size(obj)
        if size > self.memory_limit_bytes:
            return
        while self._mem_usage + size > self.memory_limit_bytes and self._mem:
            _, evicted = self._mem.popitem(last=False)
            self._mem_usage -= self._obj_size(evicted)
        self._mem[mkey] = obj
        self._mem_usage += size

    def clear(self, ns: Optional[str] = None):
        if ns is None:
            for p in self.cache_dir.rglob("*.pkl"):
                p.unlink(missing_ok=True)
            self._mem.clear()
            self._mem_usage = 0
        else:
            ns_dir = self.cache_dir / ns
            if ns_dir.exists():
                for p in ns_dir.glob("*.pkl"):
                    p.unlink(missing_ok=True)
            keys = [k for k in self._mem if k.startswith(f"{ns}:")]
            for k in keys:
                obj = self._mem.pop(k, None)
                if obj is not None:
                    self._mem_usage -= self._obj_size(obj)

    def memory_usage_mb(self) -> float:
        return self._mem_usage / (1024 * 1024)

    def cached_items(self) -> int:
        return len(self._mem)


def make_cache_key(*parts) -> str:
    combined = ":".join(str(p) for p in parts)
    return hashlib.md5(combined.encode()).hexdigest()


def load_with_dask(pkl_path: Path):
    if not DASK_AVAILABLE:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    import dask
    return dask.delayed(pickle.load)(open(pkl_path, "rb"))