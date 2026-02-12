from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, Any


COMMON_SUBFOLDERS = ["", "tables", "outputs", "outputs/tables"]

def find_file(results_root: Path, filename: str) -> Path:
    for sub in COMMON_SUBFOLDERS:
        p = results_root / sub / filename
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {filename} under {results_root} (checked {COMMON_SUBFOLDERS}).")

def read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Handles the case where file is read with columns like C1/C2 and the first row is the real header.
    """
    df = pd.read_csv(path)
    cols = [str(c).lower() for c in df.columns]
    if cols and all(c.startswith("c") for c in cols) and len(df) > 0:
        first = df.iloc[0].astype(str).str.lower().tolist()
        if "subject_id" in first and ("edf_path" in first or "path" in first):
            df2 = pd.read_csv(path, header=None)
            header = df2.iloc[0].tolist()
            df2 = df2[1:].copy()
            df2.columns = header
            return df2.reset_index(drop=True)
    return df

def load_tables(results_root: Path):
    recs = read_csv_flexible(find_file(results_root, "recordings.csv"))
    subs = read_csv_flexible(find_file(results_root, "subjects.csv"))
    evts = read_csv_flexible(find_file(results_root, "seizure_events.csv"))

    if "edf_path" not in recs.columns and "path" in recs.columns:
        recs = recs.rename(columns={"path": "edf_path"})

    # Clean header rows inside subjects.csv
    if "subject_id" in subs.columns:
        subs = subs[subs["subject_id"].astype(str).str.lower() != "case"]
    if "sex" in subs.columns:
        subs = subs[subs["sex"].astype(str).str.lower() != "gender"]

    if "age" in subs.columns:
        subs["age"] = pd.to_numeric(subs["age"], errors="coerce")

    if "subject_id" in recs.columns and "subject_id" in subs.columns:
        recs = recs.merge(subs, on="subject_id", how="left")

    needed = {"subject_id", "edf_file", "seizure_index", "seizure_start_sec", "seizure_end_sec"}
    if not needed.issubset(set(evts.columns)):
        raise ValueError(f"seizure_events.csv missing required cols: {needed}. Got: {set(evts.columns)}")

    return recs, subs, evts

def load_yaml_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

