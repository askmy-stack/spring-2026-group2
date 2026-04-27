from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def load_split(
    csv_path: str | Path,
    target_col: str,
    meta_cols: Sequence[str],
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    csv_path = Path(csv_path).expanduser().resolve()
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")

    y = df[target_col].astype(np.int32).to_numpy()

    feature_cols = [c for c in df.columns if c not in set(meta_cols)]
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    if dtype == "float32":
        X = df[feature_cols].to_numpy(dtype=np.float32)
    elif dtype == "float64":
        X = df[feature_cols].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return X, y, feature_cols, df


def validate_feature_columns(
    train_cols: list[str],
    val_cols: list[str],
    test_cols: list[str],
) -> None:
    if train_cols != val_cols or train_cols != test_cols:
        raise ValueError("Feature columns do not match across train/val/test.")
