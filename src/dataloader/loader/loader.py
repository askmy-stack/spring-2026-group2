"""
Unified Loader — single entry point for the entire data pipeline.

Two functions:
    generate(dataset, subjects)  → runs pipeline, saves CSVs + tensors
    get_dataloaders(dataset)     → returns PyTorch DataLoaders (instant)

Three-level caching:
    1. Tensors exist  → load directly (fastest)
    2. CSVs exist     → tensorize, then load
    3. Nothing exists → full pipeline, then tensorize, then load

Uses OOP loader classes (CHBMITLoader, SienaLoader) that inherit from
BaseDatasetLoader. No hardcoded metadata — everything parsed from source.
"""

from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataloader.loader.base import BaseDatasetLoader
from dataloader.loader.tensor_writer import TensorDataset, write_tensors
from dataloader.loader.windowing import build_windows, balance_windows
from dataloader.loader.splits import subject_split

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════
# REGISTRY — maps dataset name → loader class
# ═══════════════════════════════════════════════════════════

def _get_loader(dataset: str, force: bool = False) -> BaseDatasetLoader:
    """
    Instantiate the right loader class for a dataset.

    Returns:
        BaseDatasetLoader subclass instance
    """
    if dataset == "chbmit":
        from dataloader.loader.chbmit_loader import CHBMITLoader
        return CHBMITLoader(force=force)
    elif dataset == "siena":
        from dataloader.loader.siena_loader import SienaLoader
        return SienaLoader(force=force)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: chbmit, siena"
        )


# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════

def _csv_dir(dataset: str) -> Path:
    return Path(f"results/dataloader/{dataset}")


def _tensor_dir(dataset: str) -> Path:
    return Path(f"results/tensors/{dataset}")


def _tensors_exist(dataset: str) -> bool:
    return (_tensor_dir(dataset) / "train" / "data.pt").exists()


def _csvs_exist(dataset: str) -> bool:
    return (_csv_dir(dataset) / "window_index_train.csv").exists()


# ═══════════════════════════════════════════════════════════
# PROCESS — run the pipeline for one EDF
# ═══════════════════════════════════════════════════════════

TARGET_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "Cz",
    "T7", "T8", "P7", "P8", "C3", "C4", "O1", "O2",
]


def _process_edf(
    edf_path: Path,
    subject_id: str,
    seizure_intervals: List[Tuple[float, float]],
    metadata: Dict,
    loader: BaseDatasetLoader,
    bids_root: Path,
    max_background_per_file: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Process a single EDF: read → preprocess → standardize → BIDS → window.
    Uses loader.get_config() for dataset-specific parameters.
    """
    import mne

    config = loader.get_config()

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    except Exception as e:
        print(f"    [SKIP] {edf_path.name}: {e}")
        return pd.DataFrame()

    sfreq = raw.info["sfreq"]
    target_sfreq = config["target_sfreq"]

    # ── Preprocess ────────────────────────────────────────
    try:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) > 0:
            raw.pick(eeg_picks)

        if abs(sfreq - target_sfreq) > 1.0:
            raw.resample(target_sfreq, verbose=False)

        raw.filter(1.0, 50.0, verbose=False)
        raw.notch_filter(config["notch_freq"], verbose=False)
        raw.set_eeg_reference("average", verbose=False)

    except Exception as e:
        print(f"    [SKIP] {edf_path.name} preprocess: {e}")
        return pd.DataFrame()

    # ── Channel standardization ───────────────────────────
    ch_names = raw.ch_names
    name_map = {}
    for ch in ch_names:
        clean = ch.split("-")[0].strip().upper()
        for target in TARGET_CHANNELS:
            if clean == target.upper():
                name_map[ch] = target
                break

    matched = [ch for ch in ch_names if ch in name_map]
    if len(matched) < 4:
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

    try:
        raw.export(str(bids_edf_path), fmt="edf", overwrite=True, verbose=False)
    except Exception:
        bids_edf_path = bids_edf_path.with_suffix(".fif")
        raw.save(str(bids_edf_path), overwrite=True, verbose=False)

    windows_df = build_windows(
        edf_path=bids_edf_path,
        seizure_intervals=seizure_intervals,
        subject_id=subject_id,
        metadata=metadata,
        duration_sec=duration_sec,
        window_sec=1.0,
        stride_sec=1.0,
        sfreq=target_sfreq,
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
    seizure_ratio: float = 0.3,
    seed: int = 42,
    max_background_per_file: int = 150,
) -> Path:
    """
    Run the full pipeline: download → preprocess → window → balance → split → CSVs → tensors.

    Args:
        dataset: "chbmit" or "siena"
        subjects: List of subject IDs (None = all available)
        force: Re-download even if cached
        seizure_ratio: Target seizure window ratio (default 0.3)
        seed: Random seed
        max_background_per_file: Cap background windows per EDF file (default 150).
            Seizure windows are NEVER capped. Set 0 to keep all windows.

    Returns:
        Path to tensor directory
    """
    loader = _get_loader(dataset, force=force)
    csv_dir = _csv_dir(dataset)
    tensor_dir = _tensor_dir(dataset)
    bids_root = Path(f"results/bids_dataset/{dataset}")

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
    train_df, val_df, test_df = subject_split(full_df, seed=seed)

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
    for split_name, split_csv in [("train", "window_index_train.csv"),
                                    ("val", "window_index_val.csv"),
                                    ("test", "window_index_test.csv")]:
        csv_path = csv_dir / split_csv
        out_dir = tensor_dir / split_name
        print(f"  {split_name}:")
        stats = write_tensors(csv_path, out_dir)
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
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get PyTorch DataLoaders for train/val/test.

    If tensors exist: loads instantly.
    If CSVs exist but no tensors: tensorizes first.
    If nothing exists: raises error — run generate() first.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    tensor_base = _tensor_dir(dataset)
    csv_base = _csv_dir(dataset)

    if _tensors_exist(dataset):
        print(f"[{dataset}] Loading pre-saved tensors from {tensor_base}/")

    elif _csvs_exist(dataset):
        print(f"[{dataset}] CSVs found, converting to tensors...")
        for split in ["train", "val", "test"]:
            csv_path = csv_base / f"window_index_{split}.csv"
            out_dir = tensor_base / split
            write_tensors(csv_path, out_dir)
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