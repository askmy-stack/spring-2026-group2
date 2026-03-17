"""
Unified Loader — single entry point for the entire data pipeline.

Two functions:
    generate(dataset, subjects)  → runs pipeline, saves CSVs + tensors
    get_dataloaders(dataset)     → returns PyTorch DataLoaders (instant)

Three-level caching:
    1. Tensors exist  → load directly (fastest)
    2. CSVs exist     → tensorize, then load
    3. Nothing exists → full pipeline, then tensorize, then load

All parameters (paths, channels, signal processing, splits) come from
config.yaml. No hardcoded values in this file.
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from dataloader.loader.base import BaseDatasetLoader
from dataloader.loader.tensor_writer import TensorDataset, write_tensors
from dataloader.loader.windowing import build_windows, balance_windows
from dataloader.loader.splits import subject_split

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════
# CONFIG — load from config.yaml
# ═══════════════════════════════════════════════════════════

def _load_full_config(config_path: Optional[str] = None) -> Dict:
    """Load the full config.yaml."""
    path = Path(config_path or "config.yaml")
    if not path.exists():
        path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found. Expected at {path} or in current directory."
        )
    with open(path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════
# REGISTRY — maps dataset name → loader class
# ═══════════════════════════════════════════════════════════

def _get_loader(
    dataset: str,
    force: bool = False,
    config_path: Optional[str] = None,
) -> BaseDatasetLoader:
    """
    Instantiate the right loader class for a dataset.

    Checks config.yaml → datasets for available datasets.
    """
    config = _load_full_config(config_path)
    available = list(config.get("datasets", {}).keys())

    if dataset == "chbmit":
        from dataloader.loader.chbmit_loader import CHBMITLoader
        return CHBMITLoader(force=force, config_path=config_path)
    elif dataset == "siena":
        from dataloader.loader.siena_loader import SienaLoader
        return SienaLoader(force=force, config_path=config_path)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {', '.join(available)}"
        )


# ═══════════════════════════════════════════════════════════
# PATHS — all from config.yaml
# ═══════════════════════════════════════════════════════════

def _csv_dir(dataset: str, config_path: Optional[str] = None) -> Path:
    config = _load_full_config(config_path)
    base = config.get("paths", {}).get("csv_dir", "results/dataloader")
    return Path(base) / dataset


def _tensor_dir(dataset: str, config_path: Optional[str] = None) -> Path:
    config = _load_full_config(config_path)
    base = config.get("paths", {}).get("tensor_dir", "results/tensors")
    return Path(base) / dataset


def _bids_root(dataset: str, config_path: Optional[str] = None) -> Path:
    config = _load_full_config(config_path)
    base = config.get("paths", {}).get("bids_root", "results/bids_dataset")
    return Path(base) / dataset


def _tensors_exist(dataset: str, config_path: Optional[str] = None) -> bool:
    return (_tensor_dir(dataset, config_path) / "train" / "data.pt").exists()


def _csvs_exist(dataset: str, config_path: Optional[str] = None) -> bool:
    return (_csv_dir(dataset, config_path) / "window_index_train.csv").exists()


# ═══════════════════════════════════════════════════════════
# PROCESS — run the pipeline for one EDF
# ═══════════════════════════════════════════════════════════

def _process_edf(
    edf_path: Path,
    subject_id: str,
    seizure_intervals: List[Tuple[float, float]],
    metadata: Dict,
    loader: BaseDatasetLoader,
    bids_root: Path,
    config: Dict,
    max_background_per_file: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Process a single EDF: read → preprocess → standardize → BIDS → window.
    All parameters come from the config dict.
    """
    import mne

    loader_config = loader.get_config()

    # Read signal processing params from config
    signal_cfg = config.get("signal", {})
    channels_cfg = config.get("channels", {})
    windowing_cfg = config.get("windowing", {})

    target_sfreq = signal_cfg.get("target_sfreq", loader_config["target_sfreq"])
    bandpass = signal_cfg.get("bandpass", [1.0, 50.0])
    reference = signal_cfg.get("reference", "average")
    notch_freq = loader_config.get("notch_freq", signal_cfg.get("notch", 60.0))

    target_channels = channels_cfg.get("standard_set", [
        "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "Cz",
        "T7", "T8", "P7", "P8", "C3", "C4", "O1", "O2",
    ])
    min_channels = channels_cfg.get("interpolation", {}).get("min_channels_required", 4)

    window_sec = windowing_cfg.get("window_sec", 1.0)
    stride_sec = windowing_cfg.get("stride_sec", 1.0)

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as e:
        print(f"    [SKIP] {edf_path.name}: {e}")
        return pd.DataFrame()

    sfreq = raw.info["sfreq"]

    # ── Preprocess ────────────────────────────────────────
    try:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) > 0:
            raw.pick(eeg_picks)

        if abs(sfreq - target_sfreq) > 1.0:
            raw.resample(target_sfreq, verbose=False)

        raw.filter(bandpass[0], bandpass[1], verbose=False)
        raw.notch_filter(notch_freq, verbose=False)
        raw.set_eeg_reference(reference, verbose=False)

    except Exception as e:
        print(f"    [SKIP] {edf_path.name} preprocess: {e}")
        return pd.DataFrame()

    # ── Channel standardization ───────────────────────────
    ch_names = raw.ch_names
    name_map = {}
    for ch in ch_names:
        clean = ch.split("-")[0].strip().upper()
        for target in target_channels:
            if clean == target.upper():
                name_map[ch] = target
                break

    matched = [ch for ch in ch_names if ch in name_map]
    if len(matched) < min_channels:
        print(f"    [SKIP] {edf_path.name}: only {len(matched)} channels matched")
        return pd.DataFrame()

    try:
        raw.pick(matched)
        raw.rename_channels(name_map)
    except Exception:
        pass

    # ── Get duration and build windows ────────────────────
    duration_sec = raw.n_times / raw.info["sfreq"]

    run_num = edf_path.stem.split("_")[-1] if "_" in edf_path.stem else "01"
    bids_subj_dir = bids_root / f"sub-{subject_id}" / "ses-001" / "eeg"
    bids_subj_dir.mkdir(parents=True, exist_ok=True)
    bids_edf_path = bids_subj_dir / f"sub-{subject_id}_ses-001_task-eeg_run-{run_num}_eeg.edf"

    # Some legacy EDF headers carry out-of-range meas_date values that break writers.
    # Clearing meas_date keeps export/save stable across MNE versions.
    try:
        raw.set_meas_date(None)
    except Exception:
        pass

    try:
        raw.export(str(bids_edf_path), fmt="edf", overwrite=True, verbose=False)
    except Exception:
        bids_edf_path = bids_edf_path.with_suffix(".fif")
        raw.save(str(bids_edf_path), overwrite=True, verbose=False)

    exclude_near = windowing_cfg.get("exclude_near_seizure_sec", 300.0)

    windows_df = build_windows(
        edf_path=bids_edf_path,
        seizure_intervals=seizure_intervals,
        subject_id=subject_id,
        metadata=metadata,
        duration_sec=duration_sec,
        window_sec=window_sec,
        stride_sec=stride_sec,
        sfreq=target_sfreq,
        exclude_near_seizure_sec=exclude_near,
        max_background_per_file=max_background_per_file,
        seed=seed,
    )

    return windows_df


# ═══════════════════════════════════════════════════════════
# GENERATE — full pipeline: download → process → CSV → tensors
# ═══════════════════════════════════════════════════════════

def generate(
    dataset: str,
    subjects: Optional[List[str]] = None,
    force: bool = False,
    seizure_ratio: Optional[float] = None,
    seed: Optional[int] = None,
    max_background_per_file: Optional[int] = None,
    config_path: Optional[str] = None,
) -> Path:
    """
    Run the full pipeline: download → preprocess → window → balance → split → CSVs → tensors.

    Args:
        dataset: Dataset key from config.yaml → datasets (e.g. "chbmit", "siena")
        subjects: List of subject IDs (None = all available)
        force: Re-download even if cached
        seizure_ratio: Override config balance.seizure_ratio
        seed: Override config balance.seed
        max_background_per_file: Override config windowing.max_background_per_file
        config_path: Path to config.yaml (default: auto-detect)

    Returns:
        Path to tensor directory
    """
    config = _load_full_config(config_path)

    # Read defaults from config, allow overrides from arguments
    balance_cfg = config.get("balance", {})
    windowing_cfg = config.get("windowing", {})
    split_cfg = config.get("split", {})

    if seizure_ratio is None:
        seizure_ratio = balance_cfg.get("seizure_ratio", 0.3)
    if seed is None:
        seed = balance_cfg.get("seed", 42)
    if max_background_per_file is None:
        max_background_per_file = windowing_cfg.get("max_background_per_file", 150)

    loader = _get_loader(dataset, force=force, config_path=config_path)
    csv_dir = _csv_dir(dataset, config_path)
    tensor_dir = _tensor_dir(dataset, config_path)
    bids_root = _bids_root(dataset, config_path)

    print(f"\n{'='*60}")
    print(f"  GENERATE: {dataset.upper()}")
    cap_msg = f"{max_background_per_file} per file" if max_background_per_file > 0 else "OFF (all windows)"
    print(f"  Background cap: {cap_msg}")
    print(f"{'='*60}")

    # ── Step 1: Download ──────────────────────────────────
    print(f"\n[1/5] Downloading...")
    loader.download(subjects=subjects)

    available = loader.list_subjects()
    target_subjects = subjects or available
    target_subjects = [s for s in target_subjects if s in available]
    print(f"  Subjects: {len(target_subjects)}")

    # ── Step 2: Process each EDF ──────────────────────────
    print(f"\n[2/5] Processing EDFs...")
    all_windows = []

    for sid in target_subjects:
        edfs = loader.get_edf_paths(sid)
        metadata = loader.get_metadata(sid)
        print(f"\n  [{sid}] {len(edfs)} EDFs | age={metadata.get('age','NA')} sex={metadata.get('sex','NA')}")

        for edf in edfs:
            seizures = loader.get_seizure_intervals(edf)
            print(f"    {edf.name}", end="")

            windows_df = _process_edf(
                edf_path=edf,
                subject_id=sid,
                seizure_intervals=seizures,
                metadata=metadata,
                loader=loader,
                bids_root=bids_root,
                config=config,
                max_background_per_file=max_background_per_file,
                seed=seed,
            )

            n_total = len(windows_df)
            n_sz = (windows_df["label"] == 1).sum() if not windows_df.empty else 0

            if seizures:
                print(f" → {n_total} windows, {n_sz} seizure")
            else:
                print(f" → {n_total} windows")

            all_windows.append(windows_df)

    # Combine
    full_df = pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame()

    if full_df.empty:
        print("\n  [ERROR] No windows generated. Check data.")
        return tensor_dir

    n_total = len(full_df)
    n_seizure = (full_df["label"] == 1).sum()
    print(f"\n  Total: {n_total:,} windows, {n_seizure:,} seizure ({n_seizure/n_total:.1%})")

    # ── Step 3: Subject-independent split ─────────────────
    print(f"\n[3/5] Splitting by subject...")
    train_ratio = split_cfg.get("train", 0.70)
    val_ratio = split_cfg.get("val", 0.15)
    test_ratio = split_cfg.get("test", 0.15)

    train_df, val_df, test_df = subject_split(
        full_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # ── Step 4: Balance each split ────────────────────────
    print(f"\n[4/5] Balancing to {seizure_ratio:.0%} seizure ratio...")
    before_train_sz = (train_df["label"] == 1).sum() if not train_df.empty else 0
    before_val_sz = (val_df["label"] == 1).sum() if not val_df.empty else 0
    before_test_sz = (test_df["label"] == 1).sum() if not test_df.empty else 0

    train_df = balance_windows(train_df, seizure_ratio=seizure_ratio, seed=seed)
    val_df = balance_windows(val_df, seizure_ratio=seizure_ratio, seed=seed + 1)
    test_df = balance_windows(test_df, seizure_ratio=seizure_ratio, seed=seed + 2)

    csv_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(csv_dir / "window_index_train.csv", index=False)
    val_df.to_csv(csv_dir / "window_index_val.csv", index=False)
    test_df.to_csv(csv_dir / "window_index_test.csv", index=False)

    for name, df, before_sz in [("train", train_df, before_train_sz),
                                  ("val", val_df, before_val_sz),
                                  ("test", test_df, before_test_sz)]:
        n = len(df)
        n_sz = (df["label"] == 1).sum() if not df.empty else 0
        print(f"  {name:5s}: {n:,} windows, {n_sz:,} seizure (before balance: {before_sz:,})")

    print(f"  CSVs → {csv_dir}/")

    # ── Step 5: Convert to tensors ────────────────────────
    print(f"\n[5/5] Converting to tensors...")
    channels_cfg = config.get("channels", {})
    signal_cfg = config.get("signal", {})
    n_channels = channels_cfg.get("target_count", 16)
    n_samples = signal_cfg.get("target_sfreq", 256)

    for split_name, split_csv in [("train", "window_index_train.csv"),
                                    ("val", "window_index_val.csv"),
                                    ("test", "window_index_test.csv")]:
        csv_path = csv_dir / split_csv
        out_dir = tensor_dir / split_name
        print(f"  {split_name}:")
        stats = write_tensors(csv_path, out_dir, n_channels=n_channels, n_samples=n_samples)
        print(f"    → {stats.get('n_windows', 0):,} tensors saved")

    print(f"\n{'='*60}")
    print(f"  DONE — tensors at {tensor_dir}/")
    print(f"{'='*60}\n")

    return tensor_dir


# ═══════════════════════════════════════════════════════════
# GET DATALOADERS — instant if tensors exist
# ═══════════════════════════════════════════════════════════

def get_dataloaders(
    dataset: str,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    shuffle_train: Optional[bool] = None,
    config_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get PyTorch DataLoaders for train/val/test.

    If tensors exist: loads instantly.
    If CSVs exist but no tensors: tensorizes first.
    If nothing exists: raises error — run generate() first.

    Args read from config.yaml → pytorch, overridable by arguments.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    config = _load_full_config(config_path)
    pytorch_cfg = config.get("pytorch", {})

    # Read from config, allow overrides
    if batch_size is None:
        batch_size = pytorch_cfg.get("batch_size", 64)
    if num_workers is None:
        num_workers = pytorch_cfg.get("num_workers", 0)
    if pin_memory is None:
        pin_memory = pytorch_cfg.get("pin_memory", True)
    if shuffle_train is None:
        shuffle_train = pytorch_cfg.get("shuffle_train", True)

    tensor_base = _tensor_dir(dataset, config_path)
    csv_base = _csv_dir(dataset, config_path)

    if _tensors_exist(dataset, config_path):
        print(f"[{dataset}] Loading pre-saved tensors from {tensor_base}/")

    elif _csvs_exist(dataset, config_path):
        print(f"[{dataset}] CSVs found, converting to tensors...")
        channels_cfg = config.get("channels", {})
        signal_cfg = config.get("signal", {})
        n_channels = channels_cfg.get("target_count", 16)
        n_samples = signal_cfg.get("target_sfreq", 256)

        for split in ["train", "val", "test"]:
            csv_path = csv_base / f"window_index_{split}.csv"
            out_dir = tensor_base / split
            write_tensors(csv_path, out_dir, n_channels=n_channels, n_samples=n_samples)
        print(f"[{dataset}] Tensors saved to {tensor_base}/")

    else:
        raise FileNotFoundError(
            f"No data found for '{dataset}'. Run generate() first:\n"
            f"  python main_loader.py --dataset {dataset} --generate"
        )

    train_ds = TensorDataset(tensor_base / "train")
    val_ds = TensorDataset(tensor_base / "val")
    test_ds = TensorDataset(tensor_base / "test")

    print(f"  Train: {len(train_ds):,} windows ({train_ds.seizure_ratio:.1%} seizure)")
    print(f"  Val  : {len(val_ds):,} windows ({val_ds.seizure_ratio:.1%} seizure)")
    print(f"  Test : {len(test_ds):,} windows ({test_ds.seizure_ratio:.1%} seizure)")

    device_is_gpu = torch.cuda.is_available() or (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, pin_memory=pin_memory and device_is_gpu, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory and device_is_gpu,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory and device_is_gpu,
    )

    return train_dl, val_dl, test_dl
