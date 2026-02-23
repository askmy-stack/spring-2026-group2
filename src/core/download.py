#%%
from __future__ import annotations

from collections import defaultdict
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

_DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

CHBMIT_BASE = "https://physionet.org/files/chbmit/1.0.0"
CHBMIT_SUBJECTS_BASE = "https://physionet.org/files/chbmit/1.0.0/SUBJECT-INFO"
CHBMIT_EDF_FILES_BASE = "https://physionet.org/files/chbmit/1.0.0/RECORDS"
CHBMIT_SEIZURES_BASE = "https://physionet.org/files/chbmit/1.0.0/RECORDS-WITH-SEIZURES"

# Module-level cache for loaded data
_CHBMIT_SUBJECTS_CACHE: List[Dict[str, Any]] | None = None
_CHBMIT_EDF_FILES_LIST_CACHE: List[str] | None = None
_CHBMIT_SEIZURES_LIST_CACHE: List[str] | None = None
_CHBMIT_EDF_FILES_BY_SUBJECT_CACHE: Dict[str, List[str]] | None = None
_CHBMIT_SEIZURES_BY_SUBJECT_CACHE: Dict[str, List[str]] | None = None


def load_chbmit_subjects() -> List[Dict[str, Any]]:
    """Load CHBMIT subject metadata from PhysioNet."""
    global _CHBMIT_SUBJECTS_CACHE
    if _CHBMIT_SUBJECTS_CACHE is None:
        _CHBMIT_SUBJECTS_CACHE = (
            pd.read_csv(CHBMIT_SUBJECTS_BASE, sep="\t")
            .rename(columns={"Case": "subject_id", "Age (years)": "age", "Gender": "sex"})
            .to_dict(orient="records")
        )
    return _CHBMIT_SUBJECTS_CACHE


def load_chbmit_edf_files_list() -> List[str]:
    """Load list of all CHBMIT EDF file paths from PhysioNet."""
    global _CHBMIT_EDF_FILES_LIST_CACHE
    if _CHBMIT_EDF_FILES_LIST_CACHE is None:
        _CHBMIT_EDF_FILES_LIST_CACHE = (
            pd.read_csv(CHBMIT_EDF_FILES_BASE, sep="\t", header=None)[0]
            .dropna()
            .astype(str)
            .tolist()
        )
    return _CHBMIT_EDF_FILES_LIST_CACHE


def load_chbmit_seizures_list() -> List[str]:
    """Load list of CHBMIT EDF file paths that contain seizures from PhysioNet."""
    global _CHBMIT_SEIZURES_LIST_CACHE
    if _CHBMIT_SEIZURES_LIST_CACHE is None:
        _CHBMIT_SEIZURES_LIST_CACHE = (
            pd.read_csv(CHBMIT_SEIZURES_BASE, sep="\t", header=None)[0]
            .dropna()
            .astype(str)
            .tolist()
        )
    return _CHBMIT_SEIZURES_LIST_CACHE


def group_edf_files_by_subject(edf_files_list: List[str] | None = None) -> Dict[str, List[str]]:
    """Group EDF file paths by subject_id.
    
    Args:
        edf_files_list: List of EDF file paths. If None, loads from PhysioNet.
    
    Returns:
        Dictionary mapping subject_id to sorted list of EDF file paths.
    """
    global _CHBMIT_EDF_FILES_BY_SUBJECT_CACHE
    if _CHBMIT_EDF_FILES_BY_SUBJECT_CACHE is None:
        if edf_files_list is None:
            edf_files_list = load_chbmit_edf_files_list()
        
        edf_by_subject: Dict[str, List[str]] = defaultdict(list)
        for rel_path in edf_files_list:
            rel_path = rel_path.strip()
            if not rel_path or not rel_path.endswith(".edf"):
                continue
            subject_id = rel_path.split("/", 1)[0]
            edf_by_subject[subject_id].append(rel_path)
        
        _CHBMIT_EDF_FILES_BY_SUBJECT_CACHE = {
            subject_id: sorted(paths) for subject_id, paths in edf_by_subject.items()
        }
    return _CHBMIT_EDF_FILES_BY_SUBJECT_CACHE


def group_seizures_by_subject(seizures_list: List[str] | None = None) -> Dict[str, List[str]]:
    """Group seizure EDF file paths by subject_id.
    
    Args:
        seizures_list: List of seizure EDF file paths. If None, loads from PhysioNet.
    
    Returns:
        Dictionary mapping subject_id to sorted list of seizure EDF file paths.
    """
    global _CHBMIT_SEIZURES_BY_SUBJECT_CACHE
    if _CHBMIT_SEIZURES_BY_SUBJECT_CACHE is None:
        if seizures_list is None:
            seizures_list = load_chbmit_seizures_list()
        
        seizures_by_subject: Dict[str, List[str]] = defaultdict(list)
        for rel_path in seizures_list:
            rel_path = rel_path.strip()
            if not rel_path or not rel_path.endswith(".edf"):
                continue
            subject_id = rel_path.split("/", 1)[0]
            seizures_by_subject[subject_id].append(rel_path)
        
        _CHBMIT_SEIZURES_BY_SUBJECT_CACHE = {
            subject_id: sorted(paths) for subject_id, paths in seizures_by_subject.items()
        }
    return _CHBMIT_SEIZURES_BY_SUBJECT_CACHE


# Module-level constants (initialized lazily via functions)
# These are populated on first access to maintain backward compatibility
CHBMIT_SUBJECTS: List[Dict[str, Any]] = load_chbmit_subjects()
CHBMIT_EDF_FILES_LIST: List[str] = load_chbmit_edf_files_list()
CHBMIT_SEIZURES_LIST: List[str] = load_chbmit_seizures_list()
CHBMIT_EDF_FILES_BY_SUBJECT: Dict[str, List[str]] = group_edf_files_by_subject()
CHBMIT_SEIZURES_BY_SUBJECT: Dict[str, List[str]] = group_seizures_by_subject()

def download_chbmit(
    dest_dir: Path = None,
    force: bool = False,
    subjects: List[str] = None,
) -> Dict[str, List[Path]]:
    if dest_dir is None:
        dest_dir = _DEFAULT_DATA_DIR / "chbmit"
    dest_dir = Path(dest_dir)

    if subjects is None:
        subjects = [s["subject_id"] for s in CHBMIT_SUBJECTS]

    result: Dict[str, List[Path]] = {}

    for subject_id in subjects:
        edf_paths = CHBMIT_EDF_FILES_BY_SUBJECT.get(subject_id, [])
        if not edf_paths:
            continue

        subj_dir = dest_dir / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        summary_path = subj_dir / f"{subject_id}-summary.txt"
        if not summary_path.exists():
            url = f"{CHBMIT_BASE}/{subject_id}/{subject_id}-summary.txt"
            _download_with_retry(url, summary_path)

        downloaded_paths: List[Path] = []
        for rel_path in edf_paths:
            # Extract filename from path like "chb01/chb01_12.edf" -> "chb01_12.edf"
            edf_name = rel_path.split("/", 1)[1] if "/" in rel_path else rel_path
            edf_path = subj_dir / edf_name

            if edf_path.exists() and not force:
                print(f"  [{subject_id}] Already exists: {edf_name}")
            else:
                url = f"{CHBMIT_BASE}/{rel_path}"
                print(f"  [{subject_id}] Downloading {edf_name} ...")
                if not _download_with_retry(url, edf_path):
                    print(f"  [{subject_id}] Failed: {edf_name}")
                    continue
                print(f"  [{subject_id}] Saved: {edf_path}")

            _write_events_tsv(subj_dir, edf_name)
            downloaded_paths.append(edf_path)

        _write_subject_meta_json(dest_dir, subject_id)
        result[subject_id] = downloaded_paths

    return result


def _download_with_retry(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return False


def _write_events_tsv(subj_dir: Path, edf_name: str):
    # Check if this EDF file has seizures according to CHBMIT_SEIZURES_BY_SUBJECT
    subject_id = subj_dir.name
    seizure_paths = CHBMIT_SEIZURES_BY_SUBJECT.get(subject_id, [])
    
    # Reconstruct the relative path to check if this file is in the seizure list
    rel_path = f"{subject_id}/{edf_name}"
    if rel_path not in seizure_paths:
        return
    
    # File has seizures, but we don't have interval data from ground truth
    # Events TSV will need to be populated from summary files or other sources
    tsv_path = subj_dir / edf_name.replace(".edf", "_events.tsv")
    if tsv_path.exists():
        return
    # Create empty events file - intervals should be populated from summary.txt parsing
    pd.DataFrame(columns=["onset", "duration", "trial_type", "value", "sample"]).to_csv(tsv_path, sep="\t", index=False)


def _write_subject_meta_json(chbmit_root: Path, subject_id: str):
    meta = next((s for s in CHBMIT_SUBJECTS if s["subject_id"] == subject_id), None)
    if not meta:
        return
    json_path = chbmit_root / subject_id / f"{subject_id}_meta.json"
    if json_path.exists():
        return
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)


def get_chbmit_data_dir(dest_dir: Path = None) -> Path:
    return Path(dest_dir) if dest_dir else _DEFAULT_DATA_DIR / "chbmit"


def get_chbmit_subject_meta(subject_id: str) -> Dict[str, Any]:
    return next((s for s in CHBMIT_SUBJECTS if s["subject_id"] == subject_id), {})


def print_chbmit_info():
    print("=" * 60)
    print("CHB-MIT Scalp EEG Dataset — Multi-Subject")
    print("=" * 60)
    print("Source  : PhysioNet (https://physionet.org/content/chbmit/)")
    print("License : ODC-By 1.0 (open, no login required)")
    print(f"Subjects: {len(CHBMIT_SUBJECTS)}")
    for s in CHBMIT_SUBJECTS:
        subject_id = s["subject_id"]
        seizure_paths = CHBMIT_SEIZURES_BY_SUBJECT.get(subject_id, [])
        
        print(f"  {subject_id}: age={s['age']} sex={s['sex']}  "
              f"{len(seizure_paths)} seizure file(s)")
    print("=" * 60)
# %%
