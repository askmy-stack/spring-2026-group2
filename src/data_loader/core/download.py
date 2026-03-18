from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

# Path from /EEG/src/data_loader/core/download.py -> /EEG/data/raw_data
# Go up: core -> data_loader -> src -> EEG, then down to data/raw_data
_DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw_data"

CHBMIT_BASE = "https://physionet.org/files/chbmit/1.0.0"
CHBMIT_RECORDS_URL = f"{CHBMIT_BASE}/RECORDS"

# Complete CHB-MIT dataset metadata: 23 cases from 22 subjects
# Age/sex sourced from CHB-MIT dataset documentation
# Note: chb21 is the same subject as chb01, recorded 1.5 years later
CHBMIT_SUBJECTS: List[Dict[str, Any]] = [
    {"subject_id": "chb01", "age": 11, "sex": "F", "group": "pediatric"},
    {"subject_id": "chb02", "age": 11, "sex": "M", "group": "pediatric"},
    {"subject_id": "chb03", "age": 14, "sex": "F", "group": "pediatric"},
    {"subject_id": "chb04", "age": 22, "sex": "M", "group": "adult"},
    {"subject_id": "chb05", "age": 7,  "sex": "F", "group": "pediatric"},
    {"subject_id": "chb06", "age": 1.5, "sex": "F", "group": "pediatric"},
    {"subject_id": "chb07", "age": 14.5, "sex": "F", "group": "pediatric"},
    {"subject_id": "chb08", "age": 3.5, "sex": "M", "group": "pediatric"},
    {"subject_id": "chb09", "age": 10, "sex": "F", "group": "pediatric"},
    {"subject_id": "chb10", "age": 3,  "sex": "M", "group": "pediatric"},
    {"subject_id": "chb11", "age": 12, "sex": "F", "group": "pediatric"},
    {"subject_id": "chb12", "age": 2,  "sex": "F", "group": "pediatric"},
    {"subject_id": "chb13", "age": 3,  "sex": "F", "group": "pediatric"},
    {"subject_id": "chb14", "age": 9,  "sex": "F", "group": "pediatric"},
    {"subject_id": "chb15", "age": 16, "sex": "F", "group": "pediatric"},
    {"subject_id": "chb16", "age": 7,  "sex": "F", "group": "pediatric"},
    {"subject_id": "chb17", "age": 12, "sex": "F", "group": "pediatric"},
    {"subject_id": "chb18", "age": 18, "sex": "F", "group": "adult"},
    {"subject_id": "chb19", "age": 19, "sex": "F", "group": "adult"},
    {"subject_id": "chb20", "age": 6,  "sex": "F", "group": "pediatric"},
    {"subject_id": "chb21", "age": 13, "sex": "F", "group": "pediatric"},  # Same as chb01, 1.5yr later
    {"subject_id": "chb22", "age": 9,  "sex": "F", "group": "pediatric"},
    {"subject_id": "chb23", "age": 6,  "sex": "F", "group": "pediatric"},
    {"subject_id": "chb24", "age": None, "sex": None, "group": "unknown"},
]


def download_chbmit(
    dest_dir: Path = None,
    force: bool = False,
    subjects: List[str] = None,
    download_all_files: bool = True,
) -> Dict[str, List[Path]]:
    """
    Download CHB-MIT dataset from PhysioNet.

    Args:
        dest_dir: Destination directory (default: ../data/raw/chbmit)
        force: Force re-download even if files exist
        subjects: List of subject IDs to download (default: all 23 subjects)
        download_all_files: If True, download all EDF files per subject (664 total files)
                           If False, only download summary files for testing

    Returns:
        Dictionary mapping subject_id to list of downloaded EDF file paths
    """
    if dest_dir is None:
        dest_dir = _DEFAULT_DATA_DIR / "chbmit"
    dest_dir = Path(dest_dir)

    if subjects is None:
        subjects = [s["subject_id"] for s in CHBMIT_SUBJECTS]

    # Fetch the complete file list from PhysioNet
    if download_all_files:
        print(f"Fetching file list from {CHBMIT_RECORDS_URL}...")
        file_list = _fetch_records_list()
    else:
        file_list = []

    result: Dict[str, List[Path]] = {}

    for subject_id in subjects:
        print(f"\n{'='*60}")
        print(f"Processing subject: {subject_id}")
        print(f"{'='*60}")

        subj_dir = dest_dir / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Download summary file
        summary_path = subj_dir / f"{subject_id}-summary.txt"
        if not summary_path.exists() or force:
            url = f"{CHBMIT_BASE}/{subject_id}/{subject_id}-summary.txt"
            print(f"  Downloading summary file...")
            _download_with_retry(url, summary_path)

        # Parse summary to extract seizure annotations
        seizure_files = _parse_summary_file(summary_path) if summary_path.exists() else {}

        # Download all EDF files for this subject
        edf_files = [f for f in file_list if f.startswith(f"{subject_id}/")]

        if not edf_files and download_all_files:
            print(f"  Warning: No EDF files found for {subject_id}")
            continue

        downloaded_paths = []

        for file_path in edf_files:
            edf_name = Path(file_path).name
            local_path = subj_dir / edf_name

            if local_path.exists() and not force:
                print(f"  ✓ Already exists: {edf_name}")
                downloaded_paths.append(local_path)
            else:
                url = f"{CHBMIT_BASE}/{file_path}"
                print(f"  ⬇ Downloading: {edf_name}...")
                if _download_with_retry(url, local_path):
                    print(f"    ✓ Saved: {edf_name}")
                    downloaded_paths.append(local_path)
                else:
                    print(f"    ✗ Failed: {edf_name}")

            # Write event annotations if this file has seizures
            if edf_name in seizure_files:
                _write_events_tsv_from_summary(subj_dir, edf_name, seizure_files[edf_name])

        # Write subject metadata
        _write_subject_meta_json(dest_dir, subject_id)

        result[subject_id] = downloaded_paths
        print(f"  Total files downloaded: {len(downloaded_paths)}")

    return result


def _fetch_records_list() -> List[str]:
    """Fetch the RECORDS file from PhysioNet listing all EDF files."""
    try:
        with urllib.request.urlopen(CHBMIT_RECORDS_URL) as response:
            content = response.read().decode('utf-8')
            files = [line.strip() for line in content.split('\n') if line.strip().endswith('.edf')]
            return files
    except Exception as e:
        print(f"Error fetching RECORDS: {e}")
        return []


def _parse_summary_file(summary_path: Path) -> Dict[str, List[Dict[str, float]]]:
    """
    Parse subject summary file to extract seizure annotations.

    Returns:
        Dict mapping EDF filename to list of seizure events with 'onset' and 'duration'
    """
    seizure_data = {}

    if not summary_path.exists():
        return seizure_data

    with open(summary_path, 'r') as f:
        lines = f.readlines()

    current_file = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for file name
        if line.startswith("File Name:"):
            current_file = line.split(":")[-1].strip()
            if not current_file.endswith('.edf'):
                current_file += '.edf'

        # Look for number of seizures
        elif line.startswith("Number of Seizures in File:"):
            if current_file:
                num_seizures = int(line.split(":")[-1].strip())

                if num_seizures > 0:
                    seizures = []

                    # Parse each seizure
                    for _ in range(num_seizures):
                        i += 1
                        if i >= len(lines):
                            break

                        # Look for onset time
                        onset_line = lines[i].strip()
                        if "Seizure Start Time:" in onset_line or "Seizure" in onset_line and "Start" in onset_line:
                            onset = float(onset_line.split(":")[-1].strip().split()[0])

                            i += 1
                            if i >= len(lines):
                                break

                            # Look for end time
                            end_line = lines[i].strip()
                            if "Seizure End Time:" in end_line or "Seizure" in end_line and "End" in end_line:
                                end = float(end_line.split(":")[-1].strip().split()[0])

                                seizures.append({
                                    "onset": onset,
                                    "duration": end - onset
                                })

                    if seizures:
                        seizure_data[current_file] = seizures

        i += 1

    return seizure_data


def _download_with_retry(url: str, dest: Path, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return False


def _write_events_tsv_from_summary(
    subj_dir: Path,
    edf_name: str,
    seizures: List[Dict[str, float]]
):
    """Write event TSV file from parsed summary data."""
    if not seizures:
        return

    tsv_path = subj_dir / edf_name.replace(".edf", "_events.tsv")
    if tsv_path.exists():
        return

    rows = [
        {
            "onset": s["onset"],
            "duration": round(s["duration"], 3),
            "trial_type": "seizure",
            "value": 1,
            "sample": "NA"
        }
        for s in seizures
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
    """Print information about the CHB-MIT dataset."""
    print("=" * 80)
    print("CHB-MIT Scalp EEG Database - Complete Dataset")
    print("=" * 80)
    print("Source  : PhysioNet (https://physionet.org/content/chbmit/)")
    print("License : ODC-By 1.0 (open access, no login required)")
    print(f"Version : 1.0.0")
    print(f"Total Cases: {len(CHBMIT_SUBJECTS)} (from 22 unique subjects)")
    print(f"Total Files: 664 EDF files (~23 GB)")
    print(f"Seizure Files: 129 files containing seizures")
    print(f"Total Seizures: 198 seizure events")
    print()
    print("Subject Demographics:")
    print(f"  Males: 5 (ages 3-22)")
    print(f"  Females: 17 (ages 1.5-19)")
    print()
    print("Subjects:")
    for s in CHBMIT_SUBJECTS:
        age_str = f"{s['age']:>4}" if s['age'] is not None else " N/A"
        sex_str = s['sex'] if s['sex'] is not None else "?"
        print(f"  {s['subject_id']}: age={age_str} sex={sex_str}  group={s['group']}")
    print("=" * 80)