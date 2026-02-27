from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import mne


@dataclass(frozen=True)
class WindowRow:
    path: str
    start_sec: float
    end_sec: float
    label: int
    subject_id: Optional[str] = None


def normalize_per_channel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: (C, T)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    return (x - mean) / (std + eps)


class EEGWindowEDFDataset(Dataset):
    """
    Dataset driven by a window_index CSV:
      columns required: path, start_sec, end_sec, label
      subject_id optional

    Returns:
      x: FloatTensor (C, T)
      y: FloatTensor scalar (0/1)
    """
    def __init__(
        self,
        csv_path: str | Path,
        *,
        repo_root: str | Path,
        n_channels: int = 16,
        target_sfreq: Optional[float] = None,  # set if you want resample
        expected_samples: Optional[int] = None,  # optionally enforce fixed length
        preload_raw: bool = False,  # if True, caches full raw per file path (RAM heavy)
        apply_norm: bool = True,
        verbose: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.repo_root = Path(repo_root)
        self.n_channels = int(n_channels)
        self.target_sfreq = target_sfreq
        self.expected_samples = expected_samples
        self.preload_raw = bool(preload_raw)
        self.apply_norm = bool(apply_norm)
        self.verbose = bool(verbose)

        df = pd.read_csv(self.csv_path)
        required = ["path", "start_sec", "end_sec", "label"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"{csv_path} missing required column '{c}'. Found: {list(df.columns)}")

        self.rows = [
            WindowRow(
                path=str(r["path"]),
                start_sec=float(r["start_sec"]),
                end_sec=float(r["end_sec"]),
                label=int(r["label"]),
                subject_id=str(r["subject_id"]) if "subject_id" in df.columns else None,
            )
            for _, r in df.iterrows()
            if float(r["end_sec"]) > float(r["start_sec"])
        ]

        self._raw_cache: Dict[str, mne.io.BaseRaw] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (self.repo_root / pp).resolve()

    def _load_raw(self, abs_path: Path) -> mne.io.BaseRaw:
        key = str(abs_path)
        if self.preload_raw and key in self._raw_cache:
            return self._raw_cache[key]

        raw = mne.io.read_raw_edf(abs_path, preload=self.preload_raw, verbose=False)

        if self.target_sfreq is not None and float(raw.info["sfreq"]) != float(self.target_sfreq):
            raw = raw.copy()
            raw.load_data()
            raw.resample(self.target_sfreq, npad="auto", verbose=False)

        if self.preload_raw:
            self._raw_cache[key] = raw
        return raw

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.rows[idx]
        abs_path = self._resolve(r.path)
        if not abs_path.exists():
            raise FileNotFoundError(f"EDF not found: {abs_path}")

        raw = self._load_raw(abs_path)

        # crop window
        win = raw.copy().crop(tmin=r.start_sec, tmax=r.end_sec, include_tmax=False)
        if not win.preload:
            win.load_data()

        # pick EEG if available, else all
        picks = mne.pick_types(win.info, eeg=True, meg=False, eog=False, ecg=False, stim=False, exclude=[])
        if len(picks) == 0:
            picks = np.arange(len(win.ch_names))

        X = win.get_data(picks=picks)  # (C_all, T) in Volts

        # enforce n_channels (take first n)
        if X.shape[0] >= self.n_channels:
            X = X[: self.n_channels]
        else:
            # pad channels with zeros
            pad = np.zeros((self.n_channels - X.shape[0], X.shape[1]), dtype=X.dtype)
            X = np.concatenate([X, pad], axis=0)

        x = torch.tensor(X, dtype=torch.float32)

        # enforce fixed samples length if requested
        if self.expected_samples is not None:
            T = x.shape[1]
            S = int(self.expected_samples)
            if T > S:
                x = x[:, :S]
            elif T < S:
                x = torch.nn.functional.pad(x, (0, S - T))

        if self.apply_norm:
            x = normalize_per_channel(x)

        y = torch.tensor(float(r.label), dtype=torch.float32)
        return x, y