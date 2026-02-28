"""
CHBMITLoader — CHB-MIT Scalp EEG dataset loader.

Inherits from BaseDatasetLoader. All dataset-specific logic lives here:
    - Downloads from https://physionet.org/files/chbmit/1.0.0
    - Parses SUBJECT-INFO for age/sex (no hardcoding)
    - Parses chbXX-summary.txt for seizure start/end times

Metadata source:
    The SUBJECT-INFO file at the dataset root contains a table with
    Case, Gender, and Age columns. This loader downloads that file
    and parses it at runtime instead of hardcoding subject demographics.
"""

from __future__ import annotations
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
NATIVE_SFREQ = 256.0
TARGET_SFREQ = 256
NOTCH_FREQ = 60.0       # US power line frequency
RAW_CACHE = "data/raw/chbmit"
POPULATION = "pediatric"


class CHBMITLoader(BaseDatasetLoader):
    """CHB-MIT dataset loader — pediatric epilepsy, 256 Hz, 24 subjects."""

    def __init__(self, force: bool = False, raw_cache: Optional[str] = None):
        super().__init__(
            name="chbmit",
            base_url=BASE_URL,
            raw_cache=raw_cache or RAW_CACHE,
            native_sfreq=NATIVE_SFREQ,
            target_sfreq=TARGET_SFREQ,
            notch_freq=NOTCH_FREQ,
            population=POPULATION,
            force=force,
        )
        # Cache for parsed SUBJECT-INFO (all subjects at once)
        self._subject_info_parsed = False
        self._all_metadata: Dict[str, Dict[str, Any]] = {}

    # ═══════════════════════════════════════════════════════
    # ABSTRACT IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════

    def _subject_prefix(self) -> str:
        return "chb"

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
        """Download summary.txt + all EDFs for one CHB-MIT subject."""
        subj_dir = self.raw_cache / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Download summary file (seizure annotations)
        summary_path = subj_dir / f"{subject_id}-summary.txt"
        if not summary_path.exists() or self.force:
            self._download_file(
                f"{self.base_url}/{subject_id}/{subject_id}-summary.txt",
                summary_path,
            )

        # Download EDFs
        edfs = self._edf_by_subj.get(subject_id, [])
        for rel in edfs:
            edf_name = rel.split("/", 1)[1]
            dest = subj_dir / edf_name
            if not dest.exists() or self.force:
                tqdm.write(f"  ↓ {rel}")
                if not self._download_file(f"{self.base_url}/{rel}", dest):
                    tqdm.write(f"  [WARN] Failed to download {rel}")

    def _parse_metadata(self, subject_id: str) -> Dict[str, Any]:
        """
        Parse age/sex from the SUBJECT-INFO file.

        The file lives at the dataset root and contains all subjects.
        Downloaded once, parsed once, cached for all subsequent calls.

        Format (tab or space separated):
            Case    Gender  Age
            chb01   F       11
            chb02   M       11
            ...

        Falls back to {"age": "NA", "sex": "NA"} if parsing fails.
        """
        if not self._subject_info_parsed:
            self._load_subject_info()

        return self._all_metadata.get(
            subject_id,
            {"age": "NA", "sex": "NA"},
        )

    def _parse_seizures(self, subject_id: str) -> Dict[str, List[Tuple[float, float]]]:
        """
        Parse seizure times from {subject_id}-summary.txt.

        Format:
            File Name: chb01_03.edf
            ...
            Seizure Start Time: 2996 seconds
            Seizure End Time: 3036 seconds
        """
        summary_path = self.raw_cache / subject_id / f"{subject_id}-summary.txt"
        return _parse_chbmit_summary(summary_path)

    # ═══════════════════════════════════════════════════════
    # INTERNAL — metadata parsing
    # ═══════════════════════════════════════════════════════

    def _load_subject_info(self):
        """
        Download and parse the SUBJECT-INFO file from dataset root.
        Called once, caches results in self._all_metadata.
        """
        info_path = self.raw_cache / "SUBJECT-INFO"

        # Download if not cached
        if not info_path.exists():
            self._download_file(f"{self.base_url}/SUBJECT-INFO", info_path)

        # Parse the file
        if info_path.exists():
            try:
                text = info_path.read_text(errors="ignore")
                self._all_metadata = _parse_subject_info_file(text)
            except Exception as e:
                tqdm.write(f"  [WARN] Could not parse SUBJECT-INFO: {e}")
                self._all_metadata = {}

        self._subject_info_parsed = True


# ═══════════════════════════════════════════════════════════
# PARSING FUNCTIONS — pure functions, no state
# ═══════════════════════════════════════════════════════════

def _parse_subject_info_file(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse CHB-MIT SUBJECT-INFO text into {subject_id: {"age": int, "sex": str}}.

    Handles multiple formats:
        - Tab/space separated table with headers
        - Key-value blocks per subject
    """
    result: Dict[str, Dict[str, Any]] = {}

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return result

    # Try table format: "Case  Gender  Age" header
    for i, line in enumerate(lines):
        lower = line.lower()
        if "case" in lower and ("gender" in lower or "sex" in lower):
            # Found header — parse remaining lines as rows
            for row in lines[i + 1:]:
                parts = row.split()
                if len(parts) >= 3 and parts[0].startswith("chb"):
                    subject_id = parts[0]
                    # Find gender (M/F) and age (number)
                    sex = "NA"
                    age = "NA"
                    for p in parts[1:]:
                        if p.upper() in ("M", "F"):
                            sex = p.upper()
                        elif p.isdigit():
                            age = int(p)
                    result[subject_id] = {"age": age, "sex": sex}
            if result:
                return result

    # Try key-value format: "Subject: chb01\nGender: F\nAge: 11"
    current_subj = None
    current_meta: Dict[str, Any] = {}
    for line in lines:
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()

            if "subject" in key or "case" in key:
                if current_subj:
                    result[current_subj] = current_meta
                current_subj = val
                current_meta = {"age": "NA", "sex": "NA"}
            elif "gender" in key or "sex" in key:
                current_meta["sex"] = val.upper() if val.upper() in ("M", "F") else "NA"
            elif "age" in key:
                try:
                    current_meta["age"] = int(val)
                except ValueError:
                    current_meta["age"] = "NA"

    if current_subj:
        result[current_subj] = current_meta

    return result


def _parse_chbmit_summary(path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """
    Parse {edf_filename: [(start_sec, end_sec)]} from CHB-MIT summary.txt.

    Format blocks:
        File Name: chb01_03.edf
        ...
        Number of Seizures in File: 1
        Seizure Start Time: 2996 seconds
        Seizure End Time: 3036 seconds
    """
    if not path.exists():
        return {}

    text = path.read_text(errors="ignore")
    out: Dict[str, List[Tuple[float, float]]] = {}

    for block in re.split(r"(?=File Name\s*[:\-])", text, flags=re.I):
        fm = re.search(r"File Name\s*[:\-]\s*(\S+\.edf)", block, re.I)
        if not fm:
            continue

        starts = [int(m.group(1)) for m in re.finditer(
            r"Seizure.*?Start.*?:\s*(\d+)\s*sec", block, re.I
        )]
        ends = [int(m.group(1)) for m in re.finditer(
            r"Seizure.*?End.*?:\s*(\d+)\s*sec", block, re.I
        )]

        n = min(len(starts), len(ends))
        if n != len(starts) or n != len(ends):
            tqdm.write(f"  [WARN] {fm.group(1)}: {len(starts)} starts vs {len(ends)} ends")

        out[fm.group(1)] = [(float(s), float(e)) for s, e in zip(starts[:n], ends[:n])]

    return out