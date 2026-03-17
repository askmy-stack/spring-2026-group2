"""
Subject-independent train/val/test splitting.

Ensures NO subject appears in more than one split.
Uses seizure ratio stratification when possible.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd


def subject_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split window index by subject — subject-level integrity guaranteed.

    Args:
        df: Full window index with 'subject_id' and 'label' columns
        train_ratio: Fraction of subjects for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        (train_df, val_df, test_df) — non-overlapping by subject
    """
    if df.empty:
        empty = pd.DataFrame(columns=df.columns)
        return empty, empty.copy(), empty.copy()

    subjects = sorted(df["subject_id"].unique())
    n = len(subjects)

    if n < 3:
        print(f"  [WARN] Only {n} subject(s) — all go to train. Need 3+ for val/test.")
        return df, pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    rng = np.random.RandomState(seed)

    # Compute seizure ratio per subject for stratified splitting
    subj_info = []
    for sid in subjects:
        mask = df["subject_id"] == sid
        n_total = mask.sum()
        n_seizure = (df.loc[mask, "label"] == 1).sum()
        subj_info.append({
            "subject_id": sid,
            "n_windows": n_total,
            "n_seizure": n_seizure,
            "has_seizure": n_seizure > 0,
        })
    subj_df = pd.DataFrame(subj_info)

    # Separate subjects with/without seizures for stratification
    with_sz = subj_df[subj_df["has_seizure"]]["subject_id"].tolist()
    no_sz = subj_df[~subj_df["has_seizure"]]["subject_id"].tolist()

    rng.shuffle(with_sz)
    rng.shuffle(no_sz)

    # Allocate seizure subjects proportionally
    n_sz = len(with_sz)
    n_train_sz = max(1, int(n_sz * train_ratio))
    n_val_sz = max(1, int(n_sz * val_ratio)) if n_sz > 2 else 0
    n_test_sz = n_sz - n_train_sz - n_val_sz

    train_subjs = with_sz[:n_train_sz]
    val_subjs = with_sz[n_train_sz:n_train_sz + n_val_sz]
    test_subjs = with_sz[n_train_sz + n_val_sz:]

    # Distribute non-seizure subjects
    n_no = len(no_sz)
    n_train_no = max(0, int(n_no * train_ratio))
    n_val_no = max(0, int(n_no * val_ratio))

    train_subjs += no_sz[:n_train_no]
    val_subjs += no_sz[n_train_no:n_train_no + n_val_no]
    test_subjs += no_sz[n_train_no + n_val_no:]

    train_df = df[df["subject_id"].isin(train_subjs)].reset_index(drop=True)
    val_df = df[df["subject_id"].isin(val_subjs)].reset_index(drop=True)
    test_df = df[df["subject_id"].isin(test_subjs)].reset_index(drop=True)

    print(f"  Train: {sorted(train_subjs)}")
    print(f"  Val  : {sorted(val_subjs)}")
    print(f"  Test : {sorted(test_subjs)}")

    return train_df, val_df, test_df