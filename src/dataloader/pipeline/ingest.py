from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.io import read_raw, scan_raw_dir, extract_metadata_from_companion, download_sample_edf
from core.download import download_chbmit, get_chbmit_data_dir, get_chbmit_subject_meta, print_chbmit_info
from core.signal import preprocess
from core.channels import standardize_channels
from core.bids import convert_to_bids, load_participants
from core.labels import extract_seizure_intervals, build_window_index, balance_index
from core.stratify import stratify_subjects, assign_split_column


def load_cfg(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str = "config.yaml", download_sample: bool = False):
    cfg = load_cfg(config_path)

    ds = cfg.get("dataset", {})
    raw_root = Path(ds.get("raw_root", "../data/raw"))
    bids_root = Path(ds.get("bids_root", "results/bids_dataset"))
    results_root = Path(ds.get("results_root", "results"))
    output_dir = results_root / "dataloader"
    output_dir.mkdir(parents=True, exist_ok=True)

    if download_sample or not any(raw_root.rglob("*.edf")):
        sample_dir = Path(ds.get("sample_data_dir", str(raw_root)))
        download_sample_edf(sample_dir)
        raw_root = sample_dir

    raw_files = scan_raw_dir(raw_root)
    if not raw_files:
        print(f"No EEG files found in {raw_root}")
        return

    print(f"Found {len(raw_files)} raw file(s)")

    subjects_meta: List[Dict[str, Any]] = []
    bids_records: List[Dict[str, Any]] = []

    for i, fpath in enumerate(raw_files):
        subject_id = f"{i + 1:04d}"
        print(f"  [{i + 1}/{len(raw_files)}] {fpath.name}")

        meta = extract_metadata_from_companion(fpath, cfg)
        meta.setdefault("subject_id", subject_id)

        raw = read_raw(fpath, preload=True)
        raw = preprocess(raw, cfg)
        raw = standardize_channels(raw, cfg)

        seizure_kw = cfg.get("labeling", {}).get("seizure_keywords", ["seizure", "sz"])
        seizure_intervals = _get_seizure_intervals(fpath, seizure_kw)

        events = _seizure_intervals_to_events(seizure_intervals, raw.info["sfreq"])

        bids_path = convert_to_bids(
            raw=raw,
            bids_root=bids_root,
            subject_id=subject_id,
            metadata=meta,
            cfg=cfg,
            run=f"{i + 1:02d}",
            events=events,
        )

        duration = raw.times[-1] if len(raw.times) > 0 else 0.0
        win_df = build_window_index(
            eeg_path=str(bids_path),
            duration_sec=duration,
            subject_id=subject_id,
            seizure_intervals=seizure_intervals,
            cfg=cfg,
            metadata=meta,
        )
        bids_records.append({
            "subject_id": subject_id,
            "bids_path": str(bids_path),
            "windows": win_df,
            "meta": meta,
        })

        has_seizure = (win_df["label"] == 1).any() if not win_df.empty else False
        subjects_meta.append({
            "subject_id": subject_id,
            "age": meta.get("age", "NA"),
            "sex": meta.get("sex", "NA"),
            "has_seizure": has_seizure,
        })

    subjects_df = pd.DataFrame(subjects_meta)
    train_ids, val_ids, test_ids = _split_subjects(subjects_df, cfg)

    _save_split_indices(bids_records, train_ids, val_ids, test_ids, cfg, output_dir)

    print(f"\nPipeline complete. Indices saved to {output_dir}")
    return output_dir


def _split_subjects(subjects_df: pd.DataFrame, cfg: Dict[str, Any]):
    """Shared logic: stratify or fallback-split subjects into train/val/test."""

    # ========== FIX: Handle small datasets ==========
    n_subjects = len(subjects_df)

    # For small datasets (< 6 subjects), use simple splitting to ensure all splits have data
    if n_subjects < 6:
        print(f"    Small dataset ({n_subjects} subjects), using simple split")
        ids = subjects_df["subject_id"].tolist()

        if n_subjects >= 3:
            # Ensure at least 1 subject in val and test
            n_val = 1
            n_test = 1
            n_train = n_subjects - 2
        elif n_subjects == 2:
            n_train = 1
            n_val = 1
            n_test = 0
        else:
            n_train = n_subjects
            n_val = 0
            n_test = 0

        return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]
    # ================================================

    if cfg.get("stratification", {}).get("enable", True) and len(subjects_df) >= 3:
        train_df, val_df, test_df = stratify_subjects(subjects_df, cfg)

        # ========== FIX: Ensure no empty splits ==========
        train_ids = train_df["subject_id"].tolist()
        val_ids = val_df["subject_id"].tolist()
        test_ids = test_df["subject_id"].tolist()

        # If val is empty, take 1 from train
        if len(val_ids) == 0 and len(train_ids) > 1:
            print("  ⚠️  Val split empty, moving 1 subject from train")
            val_ids = [train_ids.pop()]

        # If test is empty, take 1 from train
        if len(test_ids) == 0 and len(train_ids) > 1:
            print("  ⚠️  Test split empty, moving 1 subject from train")
            test_ids = [train_ids.pop()]

        return train_ids, val_ids, test_ids
        # =================================================

    ids = subjects_df["subject_id"].tolist()
    n = len(ids)
    split_cfg = cfg.get("split", {})
    if n >= 3:
        n_test = max(1, int(n * split_cfg.get("test", 0.15)))
        n_val = max(1, int(n * split_cfg.get("val", 0.15)))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            leftover = n - 1
            n_val = max(1, leftover // 2)
            n_test = leftover - n_val
    else:
        n_train, n_val, n_test = n, 0, 0
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def _save_split_indices(
        bids_records: List[Dict[str, Any]],
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
        cfg: Dict[str, Any],
        output_dir: Path,
):
    """Shared logic: assign splits, balance ALL splits, and write CSVs."""
    all_windows = pd.concat([r["windows"] for r in bids_records], ignore_index=True)
    all_windows = assign_split_column(all_windows, train_ids, val_ids, test_ids)

    # ========== FIX: Balance each split separately ==========
    for split in ["train", "val", "test"]:
        split_df = all_windows[all_windows["split"] == split].drop(columns=["split"], errors="ignore")

        # Skip if split is empty
        if len(split_df) == 0:
            print(f"  {split}: 0 windows (no subjects assigned)")
            out_path = output_dir / f"window_index_{split}.csv"
            split_df.to_csv(out_path, index=False)
            continue

        # Count before balancing
        n_before = (split_df["label"] == 1).sum()

        # Balance this split
        split_df = balance_index(split_df, cfg)

        # Count after balancing
        n_after = (split_df["label"] == 1).sum()

        # Save
        out_path = output_dir / f"window_index_{split}.csv"
        split_df.to_csv(out_path, index=False)

        print(f"  {split}: {len(split_df)} windows, {n_after} seizure (before balance: {n_before})")
    # ========================================================


def _get_seizure_intervals(fpath: Path, seizure_kw: List[str]):
    candidate = fpath.parent / (fpath.stem + "_events.tsv")
    if candidate.exists():
        return extract_seizure_intervals(str(candidate), seizure_kw)
    return []


def _seizure_intervals_to_events(
        intervals: List,
        sfreq: float,
) -> List[Dict[str, Any]]:
    events = []
    for s, e in intervals:
        events.append({
            "onset": s,
            "duration": e - s,
            "trial_type": "seizure",
            "value": 1,
            "sample": int(s * sfreq),
        })
    return events


def run_chbmit_pipeline(config_path: str = "config.yaml"):
    print_chbmit_info()
    cfg = load_cfg(config_path)

    chbmit_dir = get_chbmit_data_dir()
    print(f"\nDownloading CHB-MIT subjects to {chbmit_dir} ...")
    subjects_edfs = download_chbmit(dest_dir=chbmit_dir)

    if not subjects_edfs:
        print("Download failed. Check internet connection.")
        return

    bids_root = Path(cfg.get("dataset", {}).get("bids_root", "results/bids_dataset"))
    results_root = Path(cfg.get("dataset", {}).get("results_root", "results"))
    output_dir = results_root / "dataloader"
    output_dir.mkdir(parents=True, exist_ok=True)

    seizure_kw = cfg.get("labeling", {}).get("seizure_keywords", ["seizure", "sz"])
    subjects_meta: List[Dict[str, Any]] = []
    bids_records: List[Dict[str, Any]] = []

    for subject_id, edf_paths in subjects_edfs.items():
        base_meta = get_chbmit_subject_meta(subject_id)

        for i, fpath in enumerate(edf_paths):
            print(f"\n  [{subject_id}] {fpath.name}")

            meta = {**base_meta, "subject_id": subject_id}

            raw = read_raw(fpath, preload=True)
            raw = preprocess(raw, cfg)
            raw = standardize_channels(raw, cfg)

            seizure_intervals = _get_seizure_intervals(fpath, seizure_kw)
            print(f"    Seizure intervals: {seizure_intervals}")

            events = _seizure_intervals_to_events(seizure_intervals, raw.info["sfreq"])

            bids_path = convert_to_bids(
                raw=raw,
                bids_root=bids_root,
                subject_id=subject_id,
                metadata=meta,
                cfg=cfg,
                run=f"{i + 1:02d}",
                events=events,
            )

            duration = raw.times[-1] if len(raw.times) > 0 else 0.0
            win_df = build_window_index(
                eeg_path=str(bids_path),
                duration_sec=duration,
                subject_id=subject_id,
                seizure_intervals=seizure_intervals,
                cfg=cfg,
                metadata=meta,
            )
            n_pos = (win_df["label"] == 1).sum() if not win_df.empty else 0
            print(f"    Windows: {len(win_df)} total, {n_pos} seizure")

            bids_records.append({
                "subject_id": subject_id,
                "bids_path": str(bids_path),
                "windows": win_df,
                "meta": meta,
            })

        has_seizure = any(
            (r["windows"]["label"] == 1).any()
            for r in bids_records
            if r["subject_id"] == subject_id and not r["windows"].empty
        )
        subjects_meta.append({
            "subject_id": subject_id,
            "age": base_meta.get("age", "NA"),
            "sex": base_meta.get("sex", "NA"),
            "has_seizure": has_seizure,
        })

    subjects_df = pd.DataFrame(subjects_meta)
    print(f"\nSubjects: {len(subjects_df)}")

    train_ids, val_ids, test_ids = _split_subjects(subjects_df, cfg)

    print(f"  Train subjects: {train_ids}")
    print(f"  Val subjects  : {val_ids}")
    print(f"  Test subjects : {test_ids}")

    _save_split_indices(bids_records, train_ids, val_ids, test_ids, cfg, output_dir)

    print(f"\nCHB-MIT pipeline complete. Indices saved to {output_dir}")
    return output_dir