from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

_DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

CHBMIT_BASE = "https://physionet.org/files/chbmit/1.0.0"

# 6 subjects with diverse age/sex for stratification testing
# Age/sex sourced from CHB-MIT dataset documentation
# Each entry: (subject_id, edf_filename, age, sex)
CHBMIT_SUBJECTS: List[Dict[str, Any]] = [
    {"subject_id": "chb01", "age": 11,  "sex": "F", "group": "pediatric"},
    {"subject_id": "chb02", "age": 11,  "sex": "M", "group": "pediatric"},
    {"subject_id": "chb03", "age": 14,  "sex": "F", "group": "pediatric"},
    {"subject_id": "chb05", "age": 7,   "sex": "F", "group": "pediatric"},
    {"subject_id": "chb08", "age": 3,   "sex": "M", "group": "pediatric"},
    {"subject_id": "chb10", "age": 3,   "sex": "M", "group": "pediatric"},
]

# One representative EDF per subject that contains confirmed seizures
# Annotations extracted from each subject's summary.txt on PhysioNet
CHBMIT_EDF_FILES: Dict[str, str] = {
    "chb01": "chb01_03.edf",
    "chb02": "chb02_16.edf",
    "chb03": "chb03_01.edf",
    "chb05": "chb05_06.edf",
    "chb08": "chb08_02.edf",
    "chb10": "chb10_12.edf",
}

# Ground-truth seizure intervals (onset_sec, offset_sec) from PhysioNet summary files
CHBMIT_SEIZURES: Dict[str, List[Tuple[float, float]]] = {
    "chb01_03.edf": [(2996.0, 3036.0), (3369.0, 3378.0)],
    "chb02_16.edf": [(130.0,  212.0)],
    "chb03_01.edf": [(362.0,  414.0)],
    "chb05_06.edf": [(417.0,  532.0)],
    "chb08_02.edf": [(2972.0, 3053.0)],
    "chb10_12.edf": [(6.0,    100.0)],
}


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
        edf_name = CHBMIT_EDF_FILES.get(subject_id)
        if not edf_name:
            continue

        subj_dir = dest_dir / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        edf_path = subj_dir / edf_name
        summary_path = subj_dir / f"{subject_id}-summary.txt"

        if edf_path.exists() and not force:
            print(f"  [{subject_id}] Already exists: {edf_name}")
        else:
            url = f"{CHBMIT_BASE}/{subject_id}/{edf_name}"
            print(f"  [{subject_id}] Downloading {edf_name} ...")
            if not _download_with_retry(url, edf_path):
                print(f"  [{subject_id}] Failed: {edf_name}")
                continue
            print(f"  [{subject_id}] Saved: {edf_path}")

        if not summary_path.exists():
            url = f"{CHBMIT_BASE}/{subject_id}/{subject_id}-summary.txt"
            _download_with_retry(url, summary_path)

        _write_events_tsv(subj_dir, edf_name)
        _write_subject_meta_json(dest_dir, subject_id)

        result[subject_id] = [edf_path]

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
    intervals = CHBMIT_SEIZURES.get(edf_name, [])
    if not intervals:
        return
    tsv_path = subj_dir / edf_name.replace(".edf", "_events.tsv")
    if tsv_path.exists():
        return
    rows = [
        {"onset": s, "duration": round(e - s, 3), "trial_type": "seizure", "value": 1, "sample": "NA"}
        for s, e in intervals
    ]
    pd.DataFrame(rows).to_csv(tsv_path, sep="\t", index=False)


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
        edf = CHBMIT_EDF_FILES[s["subject_id"]]
        intervals = CHBMIT_SEIZURES.get(edf, [])
        total_sz = sum(e - o for o, e in intervals)
        print(f"  {s['subject_id']}: age={s['age']:2d} sex={s['sex']}  "
              f"{edf}  {len(intervals)} seizure(s) {total_sz:.0f}s")
    print("=" * 60)