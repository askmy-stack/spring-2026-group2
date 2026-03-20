"""
Pre-Cache All Tensors to Disk
==============================
Reads every unique window from window_index_{train,val,test}.csv,
loads the EDF, extracts + normalizes the signal, and saves the tensor
as a pickle file that CachedEEGLoader can find directly.

Cache layout (matches CachedEEGLoader exactly):
    results/cache/pickle/tensor/{md5_key}.pkl

Each file in train is only processed once even if oversampled (duplicates skipped).

Run on server:
    cd /home/amir/Desktop/GWU/Research/EEG/src/data_loader
    python3 precache.py [--workers N] [--splits train val test]

Default: all 3 splits, 4 parallel workers (one EDF file read per worker).
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import yaml

# ── Paths ───────────────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).parent                               # src/data_loader
RESULTS_ROOT = Path("/home/amir/Desktop/GWU/Research/EEG/results")
CACHE_DIR    = RESULTS_ROOT / "cache" / "pickle" / "tensor"
DATALOADER   = RESULTS_ROOT / "dataloader"
CONFIG_PATH  = THIS_DIR / "config.yaml"

sys.path.insert(0, str(THIS_DIR))
from core.signal import normalize_signal


# ── Helpers (must be top-level for multiprocessing) ─────────────────────────

def make_cache_key(*parts) -> str:
    combined = ":".join(str(p) for p in parts)
    return hashlib.md5(combined.encode()).hexdigest()


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return (THIS_DIR / p).resolve() if not p.is_absolute() else p


def pad_or_trim(data: np.ndarray, n_channels: int, n_samples: int) -> np.ndarray:
    c, t = data.shape
    if c < n_channels:
        data = np.vstack([data, np.zeros((n_channels - c, t), dtype=data.dtype)])
    elif c > n_channels:
        data = data[:n_channels]
    if t < n_samples:
        data = np.hstack([data, np.zeros((n_channels, n_samples - t), dtype=data.dtype)])
    elif t > n_samples:
        data = data[:, :n_samples]
    return data


def process_edf_group(args: Tuple) -> Dict:
    """
    Worker function: given one EDF path and all windows from it,
    read the EDF once and save every window tensor to disk.
    Returns a dict with counts of saved/skipped/errors.
    """
    edf_path_str, windows, n_channels, window_samples, norm_method, cache_dir_str, split = args
    cache_dir = Path(cache_dir_str)

    saved = skipped = errors = 0

    # Check which keys are already cached before loading the EDF
    keys_needed = []
    for mode, path_str, start_sec, end_sec, label in windows:
        key = make_cache_key(mode, path_str, start_sec, end_sec)
        pkl = cache_dir / f"{key}.pkl"
        if pkl.exists():
            skipped += 1
        else:
            keys_needed.append((key, mode, path_str, start_sec, end_sec, label, pkl))

    if not keys_needed:
        return {"saved": 0, "skipped": skipped, "errors": 0, "edf": edf_path_str}

    # Load EDF once
    try:
        import mne
        edf_path = resolve_path(edf_path_str)
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        sfreq = raw.info["sfreq"]
        raw_data = raw.get_data()   # shape: (n_ch, n_times)
    except Exception as e:
        return {"saved": 0, "skipped": skipped, "errors": len(keys_needed),
                "edf": edf_path_str, "error_msg": str(e)}

    for key, mode, path_str, start_sec, end_sec, label, pkl in keys_needed:
        try:
            s = int(float(start_sec) * sfreq)
            e = min(int(float(end_sec) * sfreq), raw_data.shape[1])
            data = raw_data[:, s:e].copy()
            data = pad_or_trim(data, n_channels, window_samples)
            if norm_method:
                data = normalize_signal(data, method=norm_method)
            data = data.astype(np.float32)

            import torch
            tensor = torch.tensor(data, dtype=torch.float32)
            label_t = torch.tensor(int(label), dtype=torch.long)

            with open(pkl, "wb") as f:
                pickle.dump((tensor, label_t), f, protocol=pickle.HIGHEST_PROTOCOL)
            saved += 1
        except Exception as e:
            errors += 1

    return {"saved": saved, "skipped": skipped, "errors": errors, "edf": edf_path_str}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-cache EEG tensors to disk")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    # Load config
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    sfreq          = int(cfg["signal"]["target_sfreq"])
    window_sec     = float(cfg["windowing"]["window_sec"])
    n_channels     = int(cfg["channels"]["target_count"])
    window_samples = int(window_sec * sfreq)
    norm_method    = cfg.get("signal", {}).get("normalize", None)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Pre-Cache Tensor Builder")
    print(f"  Cache dir     : {CACHE_DIR}")
    print(f"  Shape         : ({n_channels}, {window_samples})")
    print(f"  Normalization : {norm_method}")
    print(f"  Workers       : {args.workers}")
    print(f"  Splits        : {args.splits}")
    print("=" * 65)

    for split in args.splits:
        csv_path = DATALOADER / f"window_index_{split}.csv"
        if not csv_path.exists():
            print(f"\n[SKIP] {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        # Drop duplicate windows (oversampled rows in train share the same EDF slice)
        df_unique = df.drop_duplicates(subset=["path", "start_sec", "end_sec"]).copy()
        n_total = len(df_unique)

        print(f"\n[{split.upper()}] {n_total:,} unique windows across "
              f"{df_unique['path'].nunique()} EDF files")

        # Group by EDF path so we read each file only once
        groups = []
        for edf_path_str, grp in df_unique.groupby("path"):
            windows = [
                (split, row["path"], row["start_sec"], row["end_sec"], row["label"])
                for _, row in grp.iterrows()
            ]
            groups.append((
                edf_path_str, windows,
                n_channels, window_samples, norm_method,
                str(CACHE_DIR), split,
            ))

        total_saved = total_skipped = total_errors = 0
        completed = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_edf_group, g): g[0] for g in groups}
            for future in as_completed(futures):
                result = future.result()
                total_saved   += result["saved"]
                total_skipped += result["skipped"]
                total_errors  += result["errors"]
                completed += 1

                if result.get("error_msg"):
                    print(f"  ERROR reading {result['edf']}: {result['error_msg']}")

                # Progress every 10 files
                if completed % 10 == 0 or completed == len(groups):
                    pct = 100 * completed / len(groups)
                    done = total_saved + total_skipped
                    print(f"  [{pct:5.1f}%] files={completed}/{len(groups)}  "
                          f"saved={total_saved:,}  skipped={total_skipped:,}  "
                          f"errors={total_errors}  done={done:,}/{n_total:,}")

        status = "DONE" if total_errors == 0 else f"DONE WITH {total_errors} ERRORS"
        print(f"\n  [{status}] {split}: saved={total_saved:,}  "
              f"already_cached={total_skipped:,}  errors={total_errors}")

    # Final disk usage
    pkl_files = list(CACHE_DIR.glob("*.pkl"))
    size_gb = sum(p.stat().st_size for p in pkl_files) / (1024 ** 3)
    print(f"\n{'='*65}")
    print(f"Cache complete: {len(pkl_files):,} files  |  {size_gb:.2f} GB")
    print(f"Location: {CACHE_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()