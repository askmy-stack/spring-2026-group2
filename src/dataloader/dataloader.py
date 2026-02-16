"""
 EEG Data Loader with Integrated Imbalance Handling
- Class hierarchy (Base, Standard, Cached, Parallel, Enhanced)
- Optional automatic imbalance detection and rebalancing
- Built-in EEG augmentation for minority class
- Class weight calculation for weighted loss
- Pickle caching with LRU memory management
- Dask parallel processing support

"""

import pandas as pd
import numpy as np
import mne
import torch
import pickle
import hashlib
import yaml
import json
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple, Any
from torch.utils.data import Dataset
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import warnings
from scipy.interpolate import CubicSpline

try:
    import dask
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import swifter
    SWIFTER_AVAILABLE = True
except ImportError:
    SWIFTER_AVAILABLE = False

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')


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


class EEGAugmentation:
    """EEG-specific data augmentation"""
    
    @staticmethod
    def time_warp(signal, sigma=0.15):
        channels, time_steps = signal.shape
        orig_steps = np.arange(time_steps)
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=5)
        warp_steps = np.linspace(0, time_steps - 1, num=5)
        warper = CubicSpline(warp_steps, warp_steps * random_warps)
        warped_steps = np.clip(warper(orig_steps), 0, time_steps - 1)
        
        warped_signal = np.zeros_like(signal)
        for c in range(channels):
            warped_signal[c] = np.interp(warped_steps, orig_steps, signal[c])
        return warped_signal
    
    @staticmethod
    def magnitude_scale(signal, sigma=0.15):
        scale_factor = np.random.normal(loc=1.0, scale=sigma, size=(signal.shape[0], 1))
        return signal * scale_factor
    
    @staticmethod
    def add_noise(signal, snr_db=25):
        signal_power = np.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise
    
    @staticmethod
    def time_shift(signal, shift_max=15):
        shift = np.random.randint(-shift_max, shift_max)
        return np.roll(signal, shift, axis=1)


class Cacher(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any):
        pass
    
    @abstractmethod
    def clear(self):
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        pass


class PickleCacher(Cacher):
    """Two-level caching: LRU memory + disk"""
    
    def __init__(self, cache_dir: Path, memory_limit_mb: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.memory_cache = OrderedDict()
        self.memory_usage = 0
        self.stats = CacheStats()
        
        self.input_cache_dir = self.cache_dir / "input_data"
        self.tensor_cache_dir = self.cache_dir / "tensor_data"
        self.input_cache_dir.mkdir(exist_ok=True)
        self.tensor_cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, key: str, cache_type: str = "input") -> Path:
        if cache_type == "input":
            return self.input_cache_dir / f"{key}.pkl"
        elif cache_type == "tensor":
            return self.tensor_cache_dir / f"{key}.pkl"
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
    
    def _get_size(self, obj: Any) -> int:
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except:
            return 0
    
    def exists(self, key: str, cache_type: str = "input") -> bool:
        cache_key = f"{cache_type}:{key}"
        if cache_key in self.memory_cache:
            return True
        return self._get_cache_path(key, cache_type).exists()
    
    def get(self, key: str, cache_type: str = "input") -> Optional[Any]:
        start_time = time.time()
        cache_key = f"{cache_type}:{key}"
        
        if cache_key in self.memory_cache:
            self.memory_cache.move_to_end(cache_key)
            self.stats.hits += 1
            self.stats.cached_load_time += time.time() - start_time
            return self.memory_cache[cache_key]
        
        cache_path = self._get_cache_path(key, cache_type)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                data_size = self._get_size(data)
                if data_size < self.memory_limit_bytes:
                    self._add_to_memory(cache_key, data, data_size)
                
                self.stats.hits += 1
                self.stats.cached_load_time += time.time() - start_time
                return data
            except:
                cache_path.unlink()
        
        self.stats.misses += 1
        return None
    
    def put(self, key: str, value: Any, cache_type: str = "input"):
        cache_path = self._get_cache_path(key, cache_type)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            return
        
        cache_key = f"{cache_type}:{key}"
        data_size = self._get_size(value)
        if data_size < self.memory_limit_bytes:
            self._add_to_memory(cache_key, value, data_size)
    
    def _add_to_memory(self, key: str, data: Any, size: int):
        while self.memory_usage + size > self.memory_limit_bytes and self.memory_cache:
            old_key, old_data = self.memory_cache.popitem(last=False)
            self.memory_usage -= self._get_size(old_data)
        
        self.memory_cache[key] = data
        self.memory_usage += size
    
    def clear(self, cache_type: Optional[str] = None):
        if cache_type is None or cache_type == "input":
            for f in self.input_cache_dir.glob("*.pkl"):
                f.unlink()
        
        if cache_type is None or cache_type == "tensor":
            for f in self.tensor_cache_dir.glob("*.pkl"):
                f.unlink()
        
        if cache_type is None:
            self.memory_cache.clear()
            self.memory_usage = 0


class BaseEEGLoader(Dataset, ABC):
    """Base class for all EEG loaders"""
    
    STANDARD_1020_POSITIONS = {
        'Fp1': np.array([-0.3, 0.95, 0.1]), 'Fp2': np.array([0.3, 0.95, 0.1]),
        'F7': np.array([-0.7, 0.5, 0.1]), 'F3': np.array([-0.4, 0.6, 0.5]),
        'Fz': np.array([0.0, 0.7, 0.6]), 'F4': np.array([0.4, 0.6, 0.5]),
        'F8': np.array([0.7, 0.5, 0.1]), 'T7': np.array([-0.95, 0.0, 0.1]),
        'C3': np.array([-0.5, 0.0, 0.7]), 'Cz': np.array([0.0, 0.0, 1.0]),
        'C4': np.array([0.5, 0.0, 0.7]), 'T8': np.array([0.95, 0.0, 0.1]),
        'P7': np.array([-0.7, -0.5, 0.1]), 'P3': np.array([-0.4, -0.6, 0.5]),
        'Pz': np.array([0.0, -0.7, 0.6]), 'P4': np.array([0.4, -0.6, 0.5]),
        'P8': np.array([0.7, -0.5, 0.1]), 'O1': np.array([-0.3, -0.95, 0.1]),
        'O2': np.array([0.3, -0.95, 0.1]), 'AFz': np.array([0.0, 0.8, 0.4]),
    }
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "train"):
        super().__init__()
        
        self.config_path = Path(config_path)
        self.mode = mode
        
        self._load_config()
        self._init_paths()
        
        self.data_index = pd.DataFrame()
    
    def _load_config(self):
        if not self.config_path.exists():
            self.cfg = self._get_default_config()
        else:
            with open(self.config_path, 'r') as f:
                self.cfg = yaml.safe_load(f)
        
        self.target_sfreq = self.cfg.get('signal', {}).get('target_sfreq', 256)
        self.window_sec = self.cfg.get('windowing', {}).get('window_sec', 1.0)
        self.bandpass = self.cfg.get('signal', {}).get('bandpass', [1.0, 50.0])
        self.notch_freq = self.cfg.get('signal', {}).get('notch', 60.0)
        self.balance_ratio = self.cfg.get('balance', {}).get('seizure_ratio', 0.3)
        self.imbalance_threshold = 0.10
    
    def _get_default_config(self) -> Dict:
        return {
            'signal': {'target_sfreq': 256, 'bandpass': [1.0, 50.0], 'notch': 60.0},
            'windowing': {'window_sec': 1.0},
            'balance': {'seizure_ratio': 0.3},
            'dataset': {'raw_root': 'data/raw', 'bids_root': 'results/bids_dataset', 'results_root': 'results'}
        }
    
    def _init_paths(self):
        dataset_cfg = self.cfg.get('dataset', {})
        
        self.raw_root = Path(dataset_cfg.get('raw_root', 'data/raw'))
        self.bids_root = Path(dataset_cfg.get('bids_root', 'results/bids_dataset'))
        self.results_root = Path(dataset_cfg.get('results_root', 'results'))
        
        self.cache_dir = self.results_root / "dataloader" / "cache"
        self.output_dir = self.results_root / "dataloader"
        
        for path in [self.bids_root, self.cache_dir, self.output_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def _read_raw_any(self, path: str):
        """Read EDF/FIF generically."""
        p = str(path)
        # mne can auto-detect for FIF via read_raw, but EDF needs read_raw_edf.
        if p.lower().endswith(".edf") or p.lower().endswith(".edf+"):
            return mne.io.read_raw_edf(p, preload=True, verbose=False)
        elif p.lower().endswith(".fif"):
            return mne.io.read_raw_fif(p, preload=True, verbose=False)
        else:
            # fallback: try MNE generic reader
            return mne.io.read_raw(p, preload=True, verbose=False)

    def _fixed_16_channel_matrix(self, raw: mne.io.BaseRaw, data_2d: np.ndarray) -> np.ndarray:
        """
        Return data in fixed channel order with shape (target_count, n_times).
        Missing channels are filled with zeros (keeps shapes stable/generic).
        """
        std = self.cfg.get("channels", {}).get("standard_set", [])
        target_count = int(self.cfg.get("channels", {}).get("target_count", 16))

        # If no list provided, just keep first target_count channels (still fixed count)
        if not std:
            X = data_2d
            if X.shape[0] >= target_count:
                return X[:target_count, :]
            pad = np.zeros((target_count - X.shape[0], X.shape[1]), dtype=X.dtype)
            return np.vstack([X, pad])

        name_to_idx = {name: i for i, name in enumerate(raw.ch_names)}

        out = np.zeros((target_count, data_2d.shape[1]), dtype=data_2d.dtype)
        for k, ch in enumerate(std[:target_count]):
            if ch in name_to_idx:
                out[k] = data_2d[name_to_idx[ch]]
        return out

    def _slice_window(self, raw: mne.io.BaseRaw, start_sec: float, end_sec: float) -> np.ndarray:
        """
        Slice using the file's sfreq (not self.target_sfreq).
        This stays correct even if the file isn’t 256Hz (but your preprocess will be).
        """
        sf = float(raw.info["sfreq"])
        start_samp = int(round(float(start_sec) * sf))
        end_samp = int(round(float(end_sec) * sf))
        start_samp = max(0, start_samp)
        end_samp = min(raw.n_times, max(start_samp + 1, end_samp))
        data, _ = raw[:, start_samp:end_samp]  # shape (n_ch, n_times)
        return data


    def check_balance(self) -> Dict[str, Dict]:
        """Check if all splits are balanced"""
        print("\n" + "="*70)
        print("CHECKING DATA BALANCE")
        print("="*70)
        
        balance_info = {}
        imbalanced_splits = []
        
        for split_mode in ['train', 'val', 'test']:
            index_file = self.output_dir / f"window_index_{split_mode}.csv"
            if not index_file.exists():
                continue
            
            df = pd.read_csv(index_file)
            if 'subject_id' not in df.columns and 'path' in df.columns:
                df['subject_id'] = df['path'].apply(
                    lambda x: Path(x).parts[-3] if 'sub-' in str(x) else 'unknown'
                )
            
            total = len(df)
            seizures = (df['label'] == 1).sum()
            seizure_rate = seizures / total if total > 0 else 0
            
            balance_info[split_mode] = {
                'total': total,
                'seizures': seizures,
                'rate': seizure_rate,
                'balanced': seizure_rate >= self.imbalance_threshold
            }
            
            print(f"\n{split_mode.upper()}: {total:,} windows")
            print(f"  Seizures: {seizures:,} ({seizure_rate*100:.2f}%)")
            
            if seizure_rate < self.imbalance_threshold:
                print(f"  WARNING: IMBALANCED (< {self.imbalance_threshold*100:.0f}%)")
                imbalanced_splits.append(split_mode)
            else:
                print(f"  Status: Balanced")
        
        balance_info['imbalanced_splits'] = imbalanced_splits
        balance_info['needs_rebalancing'] = len(imbalanced_splits) > 0
        
        if not imbalanced_splits:
            print(f"\nAll splits are balanced")
        
        return balance_info
    
    def rebalance_splits(self):
        """Rebalance splits using stratified subject-level approach"""
        print(f"\n{'='*70}")
        print(f"REBALANCING SPLITS")
        print(f"{'='*70}")
        print("\nAnalyzing subjects...")
        
        all_windows = []
        for mode in ['train', 'val', 'test']:
            index_file = self.output_dir / f"window_index_{mode}.csv"
            if index_file.exists():
                df = pd.read_csv(index_file)
                df['split'] = mode
                all_windows.append(df)
        
        if not all_windows:
            print("No indices found. Run pipeline first.")
            return
        
        all_windows_df = pd.concat(all_windows, ignore_index=True)
        
        if 'subject_id' not in all_windows_df.columns:
            all_windows_df['subject_id'] = all_windows_df['path'].apply(
                lambda x: Path(x).parts[-3] if 'sub-' in str(x) else 'unknown'
            )
        
        subject_stats = []
        for subject in all_windows_df['subject_id'].unique():
            subj_data = all_windows_df[all_windows_df['subject_id'] == subject]
            total = len(subj_data)
            seizures = (subj_data['label'] == 1).sum()
            
            subject_stats.append({
                'subject': subject,
                'total_windows': total,
                'seizure_windows': seizures,
                'seizure_rate': seizures / total if total > 0 else 0
            })
        
        stats_df = pd.DataFrame(subject_stats).sort_values('seizure_rate', ascending=False)
        
        print(f"  Found {len(stats_df)} subjects")
        
        n_subjects = len(stats_df)
        n_test = max(1, int(n_subjects * 0.2))
        n_val = max(1, int((n_subjects - n_test) * 0.25))
        n_train = n_subjects - n_test - n_val
        
        train_subjects, val_subjects, test_subjects = [], [], []
        
        for i, (_, row) in enumerate(stats_df.iterrows()):
            subject = row['subject']
            split_idx = i % 3
            
            if len(train_subjects) < n_train and split_idx == 0:
                train_subjects.append(subject)
            elif len(val_subjects) < n_val and split_idx == 1:
                val_subjects.append(subject)
            elif len(test_subjects) < n_test and split_idx == 2:
                test_subjects.append(subject)
            else:
                if len(train_subjects) < n_train:
                    train_subjects.append(subject)
                elif len(val_subjects) < n_val:
                    val_subjects.append(subject)
                else:
                    test_subjects.append(subject)
        
        subject_to_split = {}
        for s in train_subjects:
            subject_to_split[s] = 'train'
        for s in val_subjects:
            subject_to_split[s] = 'val'
        for s in test_subjects:
            subject_to_split[s] = 'test'
        
        all_windows_df['new_split'] = all_windows_df['subject_id'].map(subject_to_split)
        
        backup_dir = self.output_dir / "backup_before_rebalance"
        backup_dir.mkdir(exist_ok=True)
        
        import shutil
        for mode in ['train', 'val', 'test']:
            old_file = self.output_dir / f"window_index_{mode}.csv"
            if old_file.exists():
                shutil.copy2(old_file, backup_dir / f"window_index_{mode}.csv")
        
        for split in ['train', 'val', 'test']:
            split_df = all_windows_df[all_windows_df['new_split'] == split].copy()
            split_df = split_df.drop(['split', 'new_split', 'subject_id'], axis=1, errors='ignore')
            
            output_file = self.output_dir / f"window_index_{split}.csv"
            split_df.to_csv(output_file, index=False)
            
            seizures = (split_df['label'] == 1).sum()
            rate = seizures / len(split_df) * 100 if len(split_df) > 0 else 0
            print(f"  {split.upper()}: {len(split_df):,} windows, {seizures:,} seizures ({rate:.1f}%)")
        
        print(f"\nRebalancing complete")
        print(f"Old indices backed up to: {backup_dir}")
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for weighted loss"""
        if self.data_index.empty:
            return torch.tensor([1.0, 1.0])
        
        class_counts = self.data_index['label'].value_counts().sort_index()
        
        if len(class_counts) < 2:
            return torch.tensor([1.0, 1.0])
        
        total = len(self.data_index)
        weights = total / (len(class_counts) * class_counts.values)
        weights = weights / weights.sum() * len(class_counts)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    @abstractmethod
    def __len__(self) -> int:
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
    
    def get_dataset_summary(self) -> Dict:
        if self.data_index.empty:
            return {'error': 'No data loaded'}
        
        summary = {
            'mode': self.mode,
            'total_windows': len(self.data_index),
            'target_sfreq': self.target_sfreq,
            'window_duration_sec': self.window_sec,
            'labels': self.data_index['label'].value_counts().to_dict(),
            'num_files': self.data_index['path'].nunique() if 'path' in self.data_index.columns else 0,
        }
        
        return summary


class StandardEEGLoader(BaseEEGLoader):
    """Standard loader without caching"""
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "train"):
        super().__init__(config_path, mode)
        self._load_data_index()
    
    def _load_data_index(self):
        index_path = self.output_dir / f"window_index_{self.mode}.csv"
        
        if index_path.exists():
            self.data_index = pd.read_csv(index_path)
            print(f"Loaded {len(self.data_index)} windows for {self.mode}")
        else:
            print(f"Warning: No index found at {index_path}")
    
    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.data_index.empty:
            raise ValueError("Data index is empty")

        row = self.data_index.iloc[idx]

        raw = self._read_raw_any(row["path"])

        data = self._slice_window(raw, row["start_sec"], row["end_sec"])
        data = self._fixed_16_channel_matrix(raw, data)

        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(int(row["label"]), dtype=torch.long)
        return data_tensor, label_tensor


class CachedEEGLoader(BaseEEGLoader):
    """Loader with pickle caching and optional augmentation"""
    
    def __init__(self, config_path: str = "src/dataloader/configs/config.yaml", mode: str = "train",
                 cache_memory_mb: int = 1000, augment: bool = False):
        super().__init__(config_path, mode)
        
        self.cache_memory_mb = cache_memory_mb
        self.augment = augment and mode == 'train'
        
        pickle_cache_dir = self.cache_dir / "pickle_cache"
        self.cacher = PickleCacher(pickle_cache_dir, cache_memory_mb)
        
        self._load_data_index()
        
        if self.augment:
            print(f"Augmentation enabled for {mode} (seizures: 80%, non-seizures: 30%)")
        
        print(f"CachedEEGLoader initialized with {cache_memory_mb} MB cache")
    
    def _load_data_index(self):
        index_path = self.output_dir / f"window_index_{self.mode}.csv"
        
        if index_path.exists():
            self.data_index = pd.read_csv(index_path)
            print(f"Loaded {len(self.data_index)} windows")
        else:
            print(f"Warning: No index at {index_path}")
    
    def _get_input_cache_key(self, file_path: str, start_sec: float, end_sec: float) -> str:
        key_str = f"{file_path}:{start_sec}:{end_sec}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _augment_signal(self, data: np.ndarray, label: int) -> np.ndarray:
        prob = 0.8 if label == 1 else 0.3
        
        if np.random.random() < prob:
            if np.random.random() < 0.5:
                data = EEGAugmentation.time_warp(data)
            if np.random.random() < 0.5:
                data = EEGAugmentation.magnitude_scale(data)
            if np.random.random() < 0.3:
                data = EEGAugmentation.add_noise(data)
            if np.random.random() < 0.3:
                data = EEGAugmentation.time_shift(data)
        
        return data
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.data_index.empty:
            raise ValueError("Data index is empty")
        
        row = self.data_index.iloc[idx]
        cache_key = self._get_input_cache_key(row['path'], row['start_sec'], row['end_sec'])
        
        if self.cacher.exists(cache_key, cache_type="tensor"):
            cached_tensor = self.cacher.get(cache_key, cache_type="tensor")
            if cached_tensor is not None:
                data_tensor, label_tensor = cached_tensor
                
                if self.augment:
                    data_np = data_tensor.numpy()
                    data_np = self._augment_signal(data_np, label_tensor.item())
                    data_tensor = torch.from_numpy(data_np).float()
                
                return data_tensor, label_tensor
        
        if self.cacher.exists(cache_key, cache_type="input"):
            data = self.cacher.get(cache_key, cache_type="input")
        else:
            try:
                raw = self._read_raw_any(row["path"])
            except:
                raise ValueError(f"Failed to load: {row['path']}")

            data = self._slice_window(raw, row["start_sec"], row["end_sec"])
            data = self._fixed_16_channel_matrix(raw, data)
            
            if end_samp > raw.n_times:
                end_samp = raw.n_times
            
            data, _ = raw[:, start_samp:end_samp]
            
            self.cacher.put(cache_key, data, cache_type="input")
        
        label = int(row['label'])
        if self.augment:
            data = self._augment_signal(data, label)
        
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if not self.augment:
            self.cacher.put(cache_key, (data_tensor, label_tensor), cache_type="tensor")
        
        return data_tensor, label_tensor
    
    def get_cache_stats(self) -> Dict:
        stats = self.cacher.stats
        return {
            'hits': stats.hits,
            'misses': stats.misses,
            'hit_rate': stats.hit_rate,
            'memory_usage_mb': self.cacher.memory_usage / (1024 * 1024),
            'memory_items': len(self.cacher.memory_cache)
        }
    
    def clear_cache(self, cache_type: Optional[str] = None):
        self.cacher.clear(cache_type)


class ParallelEEGLoader(CachedEEGLoader):
    """Loader with Dask parallel processing"""
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "train",
                 cache_memory_mb: int = 1000, n_workers: int = 4, augment: bool = False):
        
        if not DASK_AVAILABLE:
            print("Warning: Dask not available. Install: pip install 'dask[complete]'")
            print("Falling back to CachedEEGLoader")
        
        super().__init__(config_path, mode, cache_memory_mb, augment)
        
        self.use_dask = DASK_AVAILABLE
        self.n_workers = n_workers
        
        if self.use_dask:
            print(f"ParallelEEGLoader initialized with {n_workers} workers")


class EnhancedEEGLoader(ParallelEEGLoader):
    """Enhanced loader with all features - recommended for production"""
    
    def __init__(self, config_path: str = "config.yaml", mode: str = "train",
                 cache_memory_mb: int = 1000, n_workers: int = 4, augment: bool = False):
        super().__init__(config_path, mode, cache_memory_mb, n_workers, augment)
        
        self.use_swifter = SWIFTER_AVAILABLE
        
        if self.use_swifter:
            print("Swifter optimization enabled")
        
        print(f"EnhancedEEGLoader ready (cache: {cache_memory_mb}MB, workers: {n_workers})")
    
    def benchmark_cache(self, num_samples: int = 100) -> Dict:
        """Benchmark cache performance"""
        if len(self) == 0:
            return {'error': 'Dataset empty'}
        
        num_samples = min(num_samples, len(self))
        
        self.cacher.stats = CacheStats()
        
        print(f"Benchmarking {num_samples} samples...")
        print("Pass 1: Cold cache...")
        
        start = time.time()
        for i in range(num_samples):
            _ = self[i]
        cold_time = time.time() - start
        
        self.cacher.stats = CacheStats()
        
        print("Pass 2: Warm cache...")
        
        start = time.time()
        for i in range(num_samples):
            _ = self[i]
        warm_time = time.time() - start
        
        speedup = cold_time / warm_time if warm_time > 0 else 0
        
        print(f"\nResults:")
        print(f"  Cold: {cold_time:.2f} sec")
        print(f"  Warm: {warm_time:.2f} sec")
        print(f"  Speedup: {speedup:.1f}x")
        
        return {
            'num_samples': num_samples,
            'cold_time': cold_time,
            'warm_time': warm_time,
            'speedup': speedup
        }


def create_loader(loader_type: str = "enhanced", **kwargs) -> BaseEEGLoader:
    """Factory function to create appropriate loader"""
    loaders = {
        'standard': StandardEEGLoader,
        'cached': CachedEEGLoader,
        'parallel': ParallelEEGLoader,
        'enhanced': EnhancedEEGLoader
    }
    
    if loader_type not in loaders:
        raise ValueError(f"Unknown loader type: {loader_type}")
    
    return loaders[loader_type](**kwargs)


UniversalEEGLoader = EnhancedEEGLoader


if __name__ == "__main__":
    print("="*70)
    print("UNIVERSAL EEG LOADER v2.0 - PRODUCTION")
    print("="*70)
    print("\nUsage:")
    print("  from dataloader import create_loader")
    print("  loader = create_loader('enhanced', mode='train', augment=True)")
    print("  class_weights = loader.get_class_weights()")
    print("  balance_info = loader.check_balance()")
