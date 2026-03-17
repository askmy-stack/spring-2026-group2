"""
Tensor Writer + TensorDataset

Converts window_index CSVs → pre-saved PyTorch tensors.
This eliminates all EDF I/O during training.

Output:
    results/tensors/{dataset}/{split}/
        data.pt     — shape (N, 16, 256)
        labels.pt   — shape (N,)
        metadata.pt — dict with subject_ids, counts, etc.
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _load_window(
    path: str,
    start_sec: float,
    end_sec: float,
    n_channels: int = 16,
    n_samples: int = 256,
    raw_cache: Optional[Dict] = None,
) -> np.ndarray:
    """Load a single window from an EDF file, normalize, return (n_channels, n_samples)."""
    import mne

    if raw_cache is not None and path in raw_cache:
        raw = raw_cache[path]
    else:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        if raw_cache is not None:
            raw_cache[path] = raw
            # Bound cache to prevent OOM
            if len(raw_cache) > 15:
                oldest = next(iter(raw_cache))
                del raw_cache[oldest]

    sfreq = raw.info["sfreq"]
    start = int(start_sec * sfreq)
    end = int(end_sec * sfreq)
    end = min(end, raw.n_times)
    data, _ = raw[:, start:end]

    # Pad or trim to exact shape
    c, t = data.shape
    if c < n_channels:
        data = np.vstack([data, np.zeros((n_channels - c, t))])
    elif c > n_channels:
        data = data[:n_channels]

    if t < n_samples:
        data = np.hstack([data, np.zeros((n_channels, n_samples - t))])
    elif t > n_samples:
        data = data[:, :n_samples]

    # Z-score normalize per channel
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    std[std == 0] = 1.0
    data = (data - mean) / std

    return data.astype(np.float32)


def write_tensors(
    csv_path: Path,
    output_dir: Path,
    n_channels: int = 16,
    n_samples: int = 256,
    batch_size: int = 500,
) -> Dict[str, int]:
    """
    Read window_index CSV and convert all windows to saved tensors.

    Args:
        csv_path: Path to window_index_{split}.csv
        output_dir: Where to save data.pt and labels.pt
        n_channels: Expected channels (default 16)
        n_samples: Expected samples per window (default 256)
        batch_size: Process this many windows before appending (memory control)

    Returns:
        dict with n_windows, n_seizure, n_background
    """
    df = pd.read_csv(csv_path)
    if df.empty or len(df) == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(0, n_channels, n_samples), output_dir / "data.pt")
        torch.save(torch.zeros(0, dtype=torch.long), output_dir / "labels.pt")
        return {"n_windows": 0, "n_seizure": 0, "n_background": 0}

    n_total = len(df)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    all_labels = []
    raw_cache: Dict = {}
    errors = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_total), desc=f"  Tensorizing", unit="win", miniters=100)
    except ImportError:
        iterator = range(n_total)

    for i in iterator:
        row = df.iloc[i]
        try:
            window = _load_window(
                path=str(row["path"]),
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                n_channels=n_channels,
                n_samples=n_samples,
                raw_cache=raw_cache,
            )
            all_data.append(window)
            all_labels.append(int(row["label"]))
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [WARN] Window {i}: {e}")
            continue

    if not all_data:
        torch.save(torch.zeros(0, n_channels, n_samples), output_dir / "data.pt")
        torch.save(torch.zeros(0, dtype=torch.long), output_dir / "labels.pt")
        return {"n_windows": 0, "n_seizure": 0, "n_background": 0}

    # Stack and save
    data_tensor = torch.from_numpy(np.stack(all_data, axis=0))   # (N, 16, 256)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)   # (N,)

    torch.save(data_tensor, output_dir / "data.pt")
    torch.save(labels_tensor, output_dir / "labels.pt")

    # Save metadata
    metadata = {
        "n_windows": len(all_data),
        "n_seizure": int((labels_tensor == 1).sum()),
        "n_background": int((labels_tensor == 0).sum()),
        "n_channels": n_channels,
        "n_samples": n_samples,
        "errors": errors,
        "subject_ids": df["subject_id"].unique().tolist(),
    }
    torch.save(metadata, output_dir / "metadata.pt")

    if errors > 0:
        print(f"  [WARN] {errors} windows failed to load")

    return metadata


class TensorDataset(Dataset):
    """
    PyTorch Dataset that loads pre-saved tensors.
    Instant — no EDF I/O at training time.
    """

    def __init__(self, tensor_dir: Path):
        self.tensor_dir = Path(tensor_dir)
        data_path = self.tensor_dir / "data.pt"
        labels_path = self.tensor_dir / "labels.pt"

        if not data_path.exists():
            raise FileNotFoundError(f"No data.pt in {tensor_dir}. Run generate() first.")

        self.data = torch.load(data_path, weights_only=True)       # (N, 16, 256)
        self.labels = torch.load(labels_path, weights_only=True)   # (N,)

        meta_path = self.tensor_dir / "metadata.pt"
        self.metadata = torch.load(meta_path, weights_only=True) if meta_path.exists() else {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

    @property
    def n_seizure(self) -> int:
        return int((self.labels == 1).sum())

    @property
    def n_background(self) -> int:
        return int((self.labels == 0).sum())

    @property
    def seizure_ratio(self) -> float:
        n = len(self)
        return self.n_seizure / n if n > 0 else 0.0