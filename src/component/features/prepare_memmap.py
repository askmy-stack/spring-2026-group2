#!/usr/bin/env python3
"""
Converts feature CSVs into memory-mapped NumPy arrays for TabNet training.
Reads source CSV paths from the same YAML config used by train_tabnet.py
and optuna_tabnet.py, so there is a single source of truth for all paths.

Skips conversion if memmap files already exist (use --force to overwrite).

Usage:
    python tabnet/prepare_memmap.py --config configs/tabnet_baseline.yaml
    python tabnet/prepare_memmap.py --config configs/tabnet_baseline.yaml --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.config_utils import load_config
from utils.io_utils import ensure_dir

DEFAULT_META_COLS = {
    "path", "start_sec", "end_sec", "label",
    "recording_path", "subject_id", "age", "sex",
    "subject", "window_idx", "start", "end",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def count_rows(csv_path: Path) -> int:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)


def detect_feature_columns(csv_path: Path, target_col: str) -> list[str]:
    sample = pd.read_csv(csv_path, nrows=2000, low_memory=False)
    if target_col not in sample.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")

    drop = DEFAULT_META_COLS | {target_col}
    candidates = [c for c in sample.columns if c not in drop]
    numeric = (
        sample[candidates]
        .select_dtypes(include=[np.number, "bool"])
        .columns.tolist()
    )
    if not numeric:
        raise ValueError(f"No numeric feature columns found in {csv_path}")
    return numeric


def build_label_mapping(csv_path: Path, target_col: str,
                        chunksize: int) -> dict:
    vals: set = set()
    for chunk in pd.read_csv(csv_path, usecols=[target_col],
                             chunksize=chunksize):
        vals.update(chunk[target_col].dropna().unique().tolist())
    vals = sorted(vals)
    if len(vals) != 2:
        raise ValueError(f"Expected binary labels, found: {vals}")
    if set(vals) == {0, 1}:
        return {0: 0, 1: 1}
    return {vals[0]: 0, vals[1]: 1}


def already_exists(out_dir: Path, split: str) -> bool:
    return (
        (out_dir / f"X_{split}.dat").exists()
        and (out_dir / f"y_{split}.dat").exists()
        and (out_dir / f"{split}_meta.json").exists()
    )


def write_split(
    csv_path: Path,
    split: str,
    feature_cols: list[str],
    label_mapping: dict,
    out_dir: Path,
    target_col: str,
    chunksize: int,
) -> None:
    n_rows     = count_rows(csv_path)
    n_features = len(feature_cols)

    x_path   = out_dir / f"X_{split}.dat"
    y_path   = out_dir / f"y_{split}.dat"
    meta_path = out_dir / f"{split}_meta.json"

    X = np.memmap(x_path, dtype="float32", mode="w+",
                  shape=(n_rows, n_features))
    y = np.memmap(y_path, dtype="int64",   mode="w+",
                  shape=(n_rows,))

    usecols = list(feature_cols) + [target_col]
    row_ptr = 0

    for i, chunk in enumerate(
        pd.read_csv(csv_path, usecols=usecols,
                    chunksize=chunksize, low_memory=False)
    ):
        X_chunk = (
            chunk[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype("float32")
            .to_numpy()
        )
        y_chunk = (
            chunk[target_col]
            .map(label_mapping)
            .astype("int64")
            .to_numpy()
        )
        n = len(chunk)
        X[row_ptr:row_ptr + n] = X_chunk
        y[row_ptr:row_ptr + n] = y_chunk
        row_ptr += n

        if (i + 1) % 10 == 0:
            logging.info("[%s] %d / %d rows written", split, row_ptr, n_rows)

    X.flush()
    y.flush()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "split":              split,
            "rows":               n_rows,
            "features":           n_features,
            "x_path":             str(x_path),
            "y_path":             str(y_path),
            "feature_names_path": str(out_dir / "feature_names.json"),
        }, f, indent=2)

    logging.info("[%s] done: %d rows, %d features", split, n_rows, n_features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to tabnet YAML config")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing memmap files")
    parser.add_argument("--chunksize", type=int, default=50_000,
                        help="CSV chunk size for streaming read")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    target_col = cfg["target_col"]
    paths      = cfg["paths"]
    train_csv  = Path(paths["train_csv"]).expanduser().resolve()
    val_csv    = Path(paths["val_csv"]).expanduser().resolve()
    test_csv   = Path(paths["test_csv"]).expanduser().resolve()
    out_dir    = ensure_dir(Path(paths["memmap_dir"]).expanduser().resolve())

    # ── detect features and label mapping from train CSV ────────────────────
    logging.info("Detecting feature columns from %s", train_csv)
    feature_cols   = detect_feature_columns(train_csv, target_col)
    label_mapping  = build_label_mapping(train_csv, target_col, args.chunksize)

    logging.info("Features: %d  |  Label mapping: %s",
                 len(feature_cols), label_mapping)

    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    with open(out_dir / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)

    # ── write each split ─────────────────────────────────────────────────────
    for split, csv_path in [("train", train_csv),
                             ("val",   val_csv),
                             ("test",  test_csv)]:
        if not args.force and already_exists(out_dir, split):
            logging.info("[%s] memmap already exists — skipping "
                         "(use --force to overwrite)", split)
            continue
        logging.info("[%s] writing memmap from %s", split, csv_path)
        write_split(csv_path, split, feature_cols, label_mapping,
                    out_dir, target_col, args.chunksize)

    logging.info("All splits ready in %s", out_dir)


if __name__ == "__main__":
    main()
