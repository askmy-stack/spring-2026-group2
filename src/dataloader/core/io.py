from __future__ import annotations

import re
import json
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional

import mne
import numpy as np


SUPPORTED_EXTENSIONS = {".edf", ".bdf", ".vhdr", ".set", ".fif"}

SAMPLE_EDF_URLS = [
    (
        "https://physionet.org/files/eegmmidb/1.0.0/S001/S001R01.edf",
        "S001R01.edf",
    ),
    (
        "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf",
        "chb01_01.edf",
    ),
]


_DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def download_sample_edf(dest_dir: Path = None, force: bool = False) -> Path:
    if dest_dir is None:
        dest_dir = _DEFAULT_DATA_DIR
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for url, filename in SAMPLE_EDF_URLS:
        dest = dest_dir / filename
        if dest.exists() and not force:
            print(f"Sample EDF already exists: {dest}")
            return dest
        try:
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"Saved to {dest}")
            return dest
        except Exception:
            continue

    dest = dest_dir / "sample_synthetic.edf"
    if not dest.exists() or force:
        _write_synthetic_edf(dest)
    return dest


def _write_synthetic_edf(dest: Path):
    sfreq = 256
    duration = 60
    n_channels = 19
    ch_names = [
        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
        "T7", "C3", "Cz", "C4", "T8",
        "P7", "P3", "Pz", "P4", "P8",
        "O1", "O2",
    ]
    t = np.linspace(0, duration, int(sfreq * duration))
    data = np.array([
        1e-5 * np.sin(2 * np.pi * 10 * t + i) + 0.5e-6 * np.random.randn(len(t))
        for i in range(n_channels)
    ])
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    raw.export(str(dest), fmt="edf", overwrite=True)
    print(f"Synthetic EDF written to {dest}")


def read_raw(path: str | Path, preload: bool = False) -> mne.io.BaseRaw:
    p = Path(path)
    ext = p.suffix.lower()
    readers = {
        ".edf": mne.io.read_raw_edf,
        ".bdf": mne.io.read_raw_bdf,
        ".vhdr": mne.io.read_raw_brainvision,
        ".set": mne.io.read_raw_eeglab,
        ".fif": mne.io.read_raw_fif,
    }
    if ext not in readers:
        raise ValueError(f"Unsupported EEG format: {ext} ({p})")
    return readers[ext](str(p), preload=preload, verbose=False)


def extract_metadata_from_companion(path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    meta_cfg = cfg.get("metadata", {})
    defaults = meta_cfg.get("defaults", {})
    meta: Dict[str, Any] = {k: v for k, v in defaults.items()}

    if meta_cfg.get("search_companion_files", True):
        for ext in meta_cfg.get("companion_extensions", [".json"]):
            companion = path.with_suffix(ext)
            if companion.exists():
                _merge_companion(companion, meta)
                break

    if meta_cfg.get("parse_filename", True):
        patterns = meta_cfg.get("filename_patterns", {})
        name = path.stem
        for key, pattern in patterns.items():
            m = re.search(pattern, name, re.IGNORECASE)
            if m:
                meta[key] = m.group(1)

    return meta


def _merge_companion(companion: Path, meta: Dict[str, Any]):
    ext = companion.suffix.lower()
    if ext == ".json":
        with open(companion) as f:
            data = json.load(f)
        meta.update({k: v for k, v in data.items() if v is not None})
    elif ext in (".yaml", ".yml"):
        import yaml
        with open(companion) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            meta.update({k: v for k, v in data.items() if v is not None})


def scan_raw_dir(raw_root: Path, recursive: bool = False) -> list[Path]:
    raw_root = Path(raw_root)
    files: list[Path] = []
    glob_fn = raw_root.rglob if recursive else raw_root.glob
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob_fn(f"*{ext}"))
    return sorted(files)