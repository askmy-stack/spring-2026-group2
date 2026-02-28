"""
SienaLoader — Siena Scalp EEG dataset loader.

Inherits from BaseDatasetLoader. All dataset-specific logic lives here:
    - Downloads from https://physionet.org/files/siena-scalp-eeg/1.0.0
    - Parses subject demographics from dataset metadata files
    - Parses Seizures-list-PNXX.csv for seizure start/end times

Metadata source:
    Siena provides subject demographics (age, sex) in metadata files
    within the dataset. The loader downloads and parses these at runtime
    instead of hardcoding subject information.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, *a, **kw): self.total = kw.get("total", 0); self.n = 0
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def update(self, n=1): self.n += n
        @staticmethod
        def write(s): print(s)

from .base import BaseDatasetLoader


# ═══════════════════════════════════════════════════════════
# DATASET CONSTANTS — URL and signal parameters only
# ═══════════════════════════════════════════════════════════

BASE_URL = "https://physionet.org/files/siena-scalp-eeg/1.0.0"
NATIVE_SFREQ = 512.0
TARGET_SFREQ = 256
NOTCH_FREQ = 50.0       # EU power line frequency
RAW_CACHE = "data/raw/siena"
POPULATION = "adult"

# Files that may contain subject demographics
METADATA_FILES = [
    "subject-info.csv",
    "SUBJECT-INFO",
    "subject-info",
    "participants.tsv",
]


class SienaLoader(BaseDatasetLoader):
    """Siena Scalp EEG loader — adult epilepsy, 512 Hz → 256 Hz, 14 subjects."""

    def __init__(self, force: bool = False, raw_cache: Optional[str] = None):
        super().__init__(
            name="siena",
            base_url=BASE_URL,
            raw_cache=raw_cache or RAW_CACHE,
            native_sfreq=NATIVE_SFREQ,
            target_sfreq=TARGET_SFREQ,
            notch_freq=NOTCH_FREQ,
            population=POPULATION,
            force=force,
        )
        self._subject_info_parsed = False
        self._all_metadata: Dict[str, Dict[str, Any]] = {}

    # ═══════════════════════════════════════════════════════
    # ABSTRACT IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════

    def _subject_prefix(self) -> str:
        return "PN"

    def _fetch_records(self) -> Dict[str, List[str]]:
        """Download RECORDS file and map subject → EDF paths."""
        records = self._fetch_lines(
            f"{self.base_url}/RECORDS",
            self.raw_cache / "RECORDS.txt",
        )
        edf_by_subj: Dict[str, List[str]] = {}
        for rel in records:
            if rel.endswith(".edf"):
                parts = rel.split("/")
                if len(parts) >= 2:
                    edf_by_subj.setdefault(parts[0], []).append(rel)
        return edf_by_subj

    def _download_subject(self, subject_id: str) -> None:
        """Download seizure CSV + all EDFs for one Siena subject."""
        subj_dir = self.raw_cache / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Download seizure annotation CSV
        csv_name = f"Seizures-list-{subject_id}.csv"
        csv_dest = subj_dir / csv_name
        if not csv_dest.exists() or self.force:
            self._download_file(
                f"{self.base_url}/{subject_id}/{csv_name}",
                csv_dest,
            )

        # Download EDFs
        edfs = self._edf_by_subj.get(subject_id, [])
        for rel in edfs:
            edf_name = rel.split("/", 1)[1] if "/" in rel else rel
            dest = subj_dir / edf_name
            if not dest.exists() or self.force:
                tqdm.write(f"  ↓ {rel}")
                if not self._download_file(f"{self.base_url}/{rel}", dest):
                    tqdm.write(f"  [WARN] Failed to download {rel}")

    def _parse_metadata(self, subject_id: str) -> Dict[str, Any]:
        """
        Parse age/sex from dataset metadata files.

        Tries multiple approaches:
            1. Look for a subject-info CSV/TSV at dataset root
            2. Parse from individual subject directories
            3. Fall back to {"age": "NA", "sex": "NA"}
        """
        if not self._subject_info_parsed:
            self._load_subject_info()

        return self._all_metadata.get(
            subject_id,
            {"age": "NA", "sex": "NA"},
        )

    def _parse_seizures(self, subject_id: str) -> Dict[str, List[Tuple[float, float]]]:
        """
        Parse seizure times from Seizures-list-{subject_id}.csv.

        CSV has columns for registration/file name, start time, end time.
        Column names vary between subjects — detection is flexible.
        """
        csv_path = self.raw_cache / subject_id / f"Seizures-list-{subject_id}.csv"
        return _parse_siena_seizure_csv(csv_path)

    # ═══════════════════════════════════════════════════════
    # INTERNAL — metadata parsing
    # ═══════════════════════════════════════════════════════

    def _load_subject_info(self):
        """
        Try to download and parse a subject metadata file from dataset root.
        Tries multiple known filenames. Caches results.
        """
        parsed = False

        for filename in METADATA_FILES:
            info_path = self.raw_cache / filename
            if not info_path.exists():
                self._download_file(f"{self.base_url}/{filename}", info_path)

            if info_path.exists() and info_path.stat().st_size > 0:
                try:
                    self._all_metadata = _parse_metadata_file(info_path)
                    if self._all_metadata:
                        parsed = True
                        break
                except Exception:
                    continue

        if not parsed:
            # Try parsing from individual subject EDF headers
            self._all_metadata = self._extract_metadata_from_edfs()

        self._subject_info_parsed = True

    def _extract_metadata_from_edfs(self) -> Dict[str, Dict[str, Any]]:
        """
        Last resort: extract age/sex from EDF file headers.
        EDF format includes a patient info field with demographics.
        """
        result: Dict[str, Dict[str, Any]] = {}

        for subj_dir in sorted(self.raw_cache.iterdir()):
            if not subj_dir.is_dir() or not subj_dir.name.startswith("PN"):
                continue

            edfs = sorted(subj_dir.glob("*.edf"))
            if not edfs:
                continue

            try:
                meta = _parse_edf_patient_info(edfs[0])
                if meta:
                    result[subj_dir.name] = meta
            except Exception:
                continue

        return result


# ═══════════════════════════════════════════════════════════
# PARSING FUNCTIONS — pure functions, no state
# ═══════════════════════════════════════════════════════════

def _parse_metadata_file(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse a subject metadata file (CSV, TSV, or text).

    Handles formats:
        - CSV/TSV with columns like: subject/case, age, sex/gender
        - Key-value text files
    """
    result: Dict[str, Dict[str, Any]] = {}
    text = path.read_text(errors="ignore").strip()

    if not text:
        return result

    # Try CSV/TSV parsing
    sep = "\t" if "\t" in text.split("\n")[0] else ","
    try:
        df = pd.read_csv(path, sep=sep)
        # Find relevant columns
        subj_col = _find_column(df.columns, ["subject", "case", "participant", "id", "patient"])
        age_col = _find_column(df.columns, ["age"])
        sex_col = _find_column(df.columns, ["sex", "gender"])

        if subj_col:
            for _, row in df.iterrows():
                subj = str(row[subj_col]).strip()
                age = "NA"
                sex = "NA"

                if age_col:
                    try:
                        age = int(float(row[age_col]))
                    except (ValueError, TypeError):
                        age = "NA"

                if sex_col:
                    s = str(row[sex_col]).strip().upper()
                    sex = s if s in ("M", "F") else "NA"

                result[subj] = {"age": age, "sex": sex}

            if result:
                return result
    except Exception:
        pass

    # Try text format parsing (line by line)
    lines = text.splitlines()
    current_subj = None
    current_meta: Dict[str, Any] = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()

            if any(k in key for k in ("subject", "case", "patient", "id")):
                if current_subj:
                    result[current_subj] = current_meta
                current_subj = val
                current_meta = {"age": "NA", "sex": "NA"}
            elif "age" in key:
                try:
                    current_meta["age"] = int(float(val))
                except ValueError:
                    pass
            elif any(k in key for k in ("sex", "gender")):
                s = val.upper()
                current_meta["sex"] = s if s in ("M", "F") else "NA"

    if current_subj:
        result[current_subj] = current_meta

    return result


def _find_column(columns, keywords: list) -> Optional[str]:
    """Find a column name matching any of the keywords (case-insensitive)."""
    for col in columns:
        col_lower = str(col).lower().strip()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None


def _parse_edf_patient_info(edf_path: Path) -> Dict[str, Any]:
    """
    Extract patient info from EDF file header.

    EDF header bytes 8-88 contain "patient identification" field.
    Format: "X X X X" where fields may include sex and birthdate.
    EDF+ format: "patient_code sex birthdate patient_name"
    """
    result = {"age": "NA", "sex": "NA"}

    with open(edf_path, "rb") as f:
        f.seek(8)
        patient_info = f.read(80).decode("ascii", errors="ignore").strip()

    if not patient_info or patient_info == "X":
        return result

    parts = patient_info.split()
    for p in parts:
        p_upper = p.upper().strip()
        if p_upper in ("M", "F", "MALE", "FEMALE"):
            result["sex"] = "M" if p_upper in ("M", "MALE") else "F"

    return result


def _parse_siena_seizure_csv(csv_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """
    Parse Siena seizure CSV → {edf_filename: [(start, end), ...]}.

    Column names vary — detection is flexible:
        - File/Registration/Name column → EDF filename
        - Start column → seizure start in seconds
        - End column → seizure end in seconds
    """
    if not csv_path.exists():
        return {}

    result: Dict[str, List[Tuple[float, float]]] = {}

    try:
        df = pd.read_csv(csv_path)

        name_col = _find_column(df.columns, ["name", "registration", "file"])
        start_col = _find_column(df.columns, ["start"])
        end_col = _find_column(df.columns, ["end"])

        if not (start_col and end_col):
            return {}

        for _, row in df.iterrows():
            try:
                s, e = float(row[start_col]), float(row[end_col])
                if e <= s:
                    continue

                edf_key = str(row[name_col]).strip() if name_col else "__all__"
                if not edf_key.lower().endswith(".edf"):
                    edf_key += ".edf"

                result.setdefault(edf_key, []).append((s, e))
            except (ValueError, TypeError):
                continue

    except Exception as exc:
        tqdm.write(f"  [WARN] Parse error {csv_path}: {exc}")

    return result