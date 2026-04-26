#!/usr/bin/env python3
"""
precache_fixed.py — Portable pre-cache script that reads all paths from config.yaml.

No hardcoded usernames or machine paths. Uses the existing data_loader/config.yaml
with its relative path structure (../../data/raw_data, ../../results, etc.).

Usage:
    cd ~/spring-2026-group2/src/data_loader
    python3 precache_fixed.py [--workers 4] [--splits train val test]
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

THIS_DIR = Path(__file__).parent.resolve()   # src/data_loader/


# ── resolve relative paths from config ────────────────────────────────────────

def resolve_from_config(rel_or_abs: str, base: Path) -> Path:
    """Resolve a path that may be relative (to base) or absolute."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def load_paths(cfg: dict, config_path: Path) -> dict:
    """Extract and resolve all relevant paths from config.yaml."""
    base         = config_path.parent                    # src/data_loader/
    raw_root     = resolve_from_config(cfg["dataset"]["raw_root"],     base)
    results_root = resolve_from_config(cfg["dataset"]["results_root"], base)
    cache_base   = resolve_from_config(cfg["caching"]["disk_cache_dir"], base)
    cache_dir    = cache_base / "pickle" / "tensor"
    dataloader_dir = results_root / "dataloader"
    return dict(
        raw_root=raw_root,
        results_root=results_root,
        cache_dir=cache_dir,
        dataloader_dir=dataloader_dir,
    )


# ── worker helpers (top-level for multiprocessing) ───────────────────────────

def make_cache_key(*parts) -> str:
    combined = ":".join(str(p) for p in parts)
    return hashlib.md5(combined.encode()).hexdigest()


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
    Worker: read one EDF file once, write all requested windows as .pkl files.
    Returns counts of saved / skipped / errors.
    """
    (edf_path_str, windows,
     n_channels, window_samples, norm_method,
     cache_dir_str) = args

    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    cache_dir = Path(cache_dir_str)
    skipped = saved = errors = 0

    # Check which windows are already cached
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
        raw      = mne.io.read_raw_edf(edf_path_str, preload=True, verbose=False)
        sfreq    = raw.info["sfreq"]
        raw_data = raw.get_data()   # (n_ch, n_times)
    except Exception as e:
        return {
            "saved": 0, "skipped": skipped,
            "errors": len(keys_needed),
            "edf": edf_path_str, "error_msg": str(e),
        }

    # Optional normalization
    normalize_fn = None
    if norm_method:
        try:
            from core.signal import normalize_signal as _norm
            normalize_fn = lambda d: _norm(d, method=norm_method)
        except Exception:
            pass

    for key, mode, path_str, start_sec, end_sec, label, pkl in keys_needed:
        try:
            s    = int(float(start_sec) * sfreq)
            e    = min(int(float(end_sec)   * sfreq), raw_data.shape[1])
            data = raw_data[:, s:e].copy()
            data = pad_or_trim(data, n_channels, window_samples)
            if normalize_fn is not None:
                data = normalize_fn(data)
            data = data.astype(np.float32)

            import torch
            tensor  = torch.tensor(data,        dtype=torch.float32)
            label_t = torch.tensor(int(label),  dtype=torch.long)

            pkl.parent.mkdir(parents=True, exist_ok=True)
            with open(pkl, "wb") as f:
                pickle.dump((tensor, label_t), f, protocol=pickle.HIGHEST_PROTOCOL)
            saved += 1
        except Exception:
            errors += 1

    return {"saved": saved, "skipped": skipped, "errors": errors, "edf": edf_path_str}


# ── EDF path resolution ───────────────────────────────────────────────────────

def resolve_edf_path(raw_path_str: str, raw_root: Path) -> str:
    """
    Try to resolve an EDF path from the window index CSV.
    Order tried:
      1. As-is (absolute or CWD-relative)
      2. Relative to raw_root
      3. Filename only under raw_root (recursive search)
    """
    p = Path(raw_path_str).expanduser()
    if p.exists():
        return str(p.resolve())

    candidate = (raw_root / p).resolve()
    if candidate.exists():
        return str(candidate)

    # Last resort: search by filename under raw_root
    fname = p.name
    matches = list(raw_root.rglob(fname))
    if matches:
        return str(matches[0])

    return str(p)   # return original; worker will report the missing file


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-cache EEG tensors using paths from config.yaml"
    )
    parser.add_argument(
        "--config",
        default=str(THIS_DIR / "config.yaml"),
        help="Path to config.yaml (default: config.yaml next to this script)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--splits", nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Signal / windowing params
    sfreq          = int(cfg["signal"]["target_sfreq"])
    window_sec     = float(cfg["windowing"]["window_sec"])
    n_channels     = int(cfg["channels"]["target_count"])
    window_samples = int(window_sec * sfreq)
    norm_method    = cfg.get("signal", {}).get("normalize", None)

    # Resolved paths
    paths = load_paths(cfg, config_path)
    raw_root       = paths["raw_root"]
    cache_dir      = paths["cache_dir"]
    dataloader_dir = paths["dataloader_dir"]

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Pre-Cache Tensor Builder")
    print(f"  Config file   : {config_path}")
    print(f"  Raw EDF root  : {raw_root}")
    print(f"  Window index  : {dataloader_dir}")
    print(f"  Cache output  : {cache_dir}")
    print(f"  Shape         : ({n_channels}, {window_samples})")
    print(f"  Normalization : {norm_method}")
    print(f"  Workers       : {args.workers}")
    print(f"  Splits        : {args.splits}")
    print("=" * 65)

    # Quick sanity check
    if not raw_root.exists():
        print(f"\nWARNING: raw_root does not exist: {raw_root}")
        print("         EDF files may not be found. Check dataset.raw_root in config.yaml\n")

    for split in args.splits:
        csv_path = dataloader_dir / f"window_index_{split}.csv"
        if not csv_path.exists():
            print(f"\n[SKIP] {csv_path} not found")
            continue

        df        = pd.read_csv(csv_path)
        df_unique = df.drop_duplicates(subset=["path", "start_sec", "end_sec"]).copy()
        n_total   = len(df_unique)

        print(f"\n[{split.upper()}] {n_total:,} unique windows  |  "
              f"{df_unique['path'].nunique()} EDF files")

        # Build worker groups (one per EDF file)
        groups = []
        for raw_path_str, grp in df_unique.groupby("path"):
            edf_str = resolve_edf_path(raw_path_str, raw_root)
            windows = [
                (split, row["path"], row["start_sec"], row["end_sec"], row["label"])
                for _, row in grp.iterrows()
            ]
            groups.append((
                edf_str, windows,
                n_channels, window_samples, norm_method,
                str(cache_dir),
            ))

        total_saved = total_skipped = total_errors = 0
        completed   = 0

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_edf_group, g): g[0] for g in groups}
            for future in as_completed(futures):
                result = future.result()
                total_saved   += result["saved"]
                total_skipped += result["skipped"]
                total_errors  += result["errors"]
                completed     += 1

                if result.get("error_msg"):
                    print(f"  ERROR {Path(result['edf']).name}: {result['error_msg']}")

                if completed % 10 == 0 or completed == len(groups):
                    pct  = 100 * completed / len(groups)
                    done = total_saved + total_skipped
                    print(f"  [{pct:5.1f}%] files={completed}/{len(groups)}  "
                          f"saved={total_saved:,}  skipped={total_skipped:,}  "
                          f"errors={total_errors}  done={done:,}/{n_total:,}")

        status = "DONE" if total_errors == 0 else f"DONE WITH {total_errors} ERRORS"
        print(f"\n  [{status}] {split}: saved={total_saved:,}  "
              f"cached={total_skipped:,}  errors={total_errors}")

    pkl_files = list(cache_dir.glob("*.pkl"))
    size_gb   = sum(p.stat().st_size for p in pkl_files) / (1024 ** 3)
    print(f"\n{'='*65}")
    print(f"Cache: {len(pkl_files):,} files  |  {size_gb:.2f} GB  |  {cache_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
