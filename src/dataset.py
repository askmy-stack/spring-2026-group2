import torch
import pandas as pd
import numpy as np
import mne
import yaml
from torch.utils.data import Dataset
from pathlib import Path
from collections import OrderedDict


class LRUCache:
    """
    Simple Least-Recently-Used Cache to keep memory safe.
    Only keeps 'capacity' number of items.
    """

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)  # Mark as recently used
        return self.cache[key]

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove oldest


class EEGDataset(Dataset):
    def __init__(self, index_csv, config_path="config.yaml", mode='train'):
        self.index = pd.read_csv(index_csv)
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # MEMORY SAFETY: Limit open files
        self.cache = LRUCache(capacity=self.cfg['system'].get('cache_size', 10))
        self.normalize = self.cfg['signal'].get('normalize', False)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]

        # 1. Memory Safe Load
        raw = self.cache.get(row['recording_id'])
        if raw is None:
            # Load with preload=True for speed, but Cache limits total RAM usage
            try:
                raw = mne.io.read_raw_fif(row['path'], preload=True, verbose='ERROR')
                self.cache.put(row['recording_id'], raw)
            except Exception:
                # Fallback for training stability
                return torch.zeros(2, 2560), torch.tensor(0)

        # 2. Slice
        sfreq = raw.info['sfreq']
        start = int(row['start_sec'] * sfreq)
        end = int(row['end_sec'] * sfreq)
        data = raw._data[:, start:end]

        # 3. Robust Normalization (Signal Quality)
        # Subtract Median / Divide by IQR (Immune to massive artifact spikes)
        if self.normalize:
            # axis=1 (time)
            median = np.median(data, axis=1, keepdims=True)
            q75, q25 = np.percentile(data, [75, 25], axis=1, keepdims=True)
            iqr = q75 - q25
            # Avoid div by zero
            iqr[iqr == 0] = 1.0
            data = (data - median) / iqr

        # 4. Tensor Conversion
        # Ensure float32 (standard for PyTorch)
        data_tensor = torch.from_numpy(data.astype(np.float32))

        # Pad if short
        expected = int((row['end_sec'] - row['start_sec']) * sfreq)
        if data_tensor.shape[1] < expected:
            pad = expected - data_tensor.shape[1]
            data_tensor = torch.nn.functional.pad(data_tensor, (0, pad))

        return data_tensor, torch.tensor(row['label'], dtype=torch.long)