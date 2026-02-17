from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


def stratify_subjects(
    subjects_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strat_cfg = cfg.get("stratification", {})
    split_cfg = cfg.get("split", {})

    age_bins = strat_cfg.get("age_bins", [0, 10, 20, 30, 100])
    age_labels = strat_cfg.get("age_bin_labels", None)
    train_r = float(split_cfg.get("train", 0.70))
    val_r = float(split_cfg.get("val", 0.15))
    test_r = float(split_cfg.get("test", 0.15))
    seed = int(split_cfg.get("seed", 42))

    df = subjects_df.copy()
    df = _add_age_bin(df, age_bins, age_labels)
    df = _normalize_sex(df)

    df["stratum"] = df["age_bin"].astype(str) + "_" + df["sex_norm"].astype(str)

    N = len(df)
    if N < 3:
        return df.copy(), df.iloc[0:0].copy(), df.iloc[0:0].copy()

    rng = np.random.default_rng(seed)

    records = (
        df[["subject_id", "stratum"]]
        .sort_values(["stratum", "subject_id"])
        .to_dict("records")
    )
    rng.shuffle(records)

    n_test = max(1, round(N * test_r))
    n_val = max(1, round(N * val_r))
    n_train = N - n_val - n_test
    if n_train < 1:
        n_train = 1
        leftover = N - 1
        n_val = max(1, leftover // 2)
        n_test = leftover - n_val

    targets = {"train": n_train, "val": n_val, "test": n_test}
    counts = {"train": 0, "val": 0, "test": 0}
    stratum_counts: Dict[str, Dict[str, int]] = {}
    assignments: Dict[str, str] = {}

    for rec in records:
        sid = rec["subject_id"]
        strat = rec["stratum"]

        if strat not in stratum_counts:
            stratum_counts[strat] = {"train": 0, "val": 0, "test": 0}

        candidates = [s for s in ["train", "val", "test"] if counts[s] < targets[s]]
        if not candidates:
            candidates = ["train"]

        best = min(
            candidates,
            key=lambda s: (stratum_counts[strat][s], counts[s] - targets[s]),
        )

        assignments[sid] = best
        counts[best] += 1
        stratum_counts[strat][best] += 1

    train_ids = [sid for sid, sp in assignments.items() if sp == "train"]
    val_ids = [sid for sid, sp in assignments.items() if sp == "val"]
    test_ids = [sid for sid, sp in assignments.items() if sp == "test"]

    train_df = df[df["subject_id"].isin(set(train_ids))].copy()
    val_df = df[df["subject_id"].isin(set(val_ids))].copy()
    test_df = df[df["subject_id"].isin(set(test_ids))].copy()

    return train_df, val_df, test_df


def _add_age_bin(df: pd.DataFrame, bins: List, labels: List = None) -> pd.DataFrame:
    df = df.copy()
    ages = pd.to_numeric(df.get("age", pd.Series(["NA"] * len(df))), errors="coerce")

    if labels is None:
        labels = [f"bin{i}" for i in range(len(bins) - 1)]

    binned = pd.cut(ages, bins=bins, labels=labels, right=False)
    df["age_bin"] = binned.astype(str).replace("nan", "unknown")
    return df


def _normalize_sex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sex_col = df.get("sex", pd.Series(["NA"] * len(df)))
    normed = []
    for v in sex_col:
        s = str(v).strip().upper()
        if s in ("M", "MALE"):
            normed.append("M")
        elif s in ("F", "FEMALE"):
            normed.append("F")
        else:
            normed.append("unknown")
    df["sex_norm"] = normed
    return df


def get_stratum_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stratum, group in df.groupby("stratum"):
        rows.append({
            "stratum": stratum,
            "n_subjects": len(group),
            "seizure_subjects": (group.get("has_seizure", pd.Series([False] * len(group)))).sum(),
        })
    return pd.DataFrame(rows)


def assign_split_column(
    windows_df: pd.DataFrame,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
) -> pd.DataFrame:
    df = windows_df.copy()
    id_to_split = {}
    for sid in train_ids:
        id_to_split[sid] = "train"
    for sid in val_ids:
        id_to_split[sid] = "val"
    for sid in test_ids:
        id_to_split[sid] = "test"
    mapped = df["subject_id"].map(id_to_split)
    unmapped = mapped.isna()
    if unmapped.any():
        unmapped_ids = df.loc[unmapped, "subject_id"].unique().tolist()
        raise ValueError(
            f"assign_split_column: {len(unmapped_ids)} subject(s) not found in any "
            f"split list: {unmapped_ids}. This indicates a data integrity issue."
        )
    df["split"] = mapped
    return df