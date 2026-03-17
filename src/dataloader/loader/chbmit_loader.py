"""
CHBMITLoader — CHB-MIT Scalp EEG dataset loader.

Supports both:
- local-only processing from existing EDF files
- optional remote download from PhysioNet
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, *a, **kw):
            self.total = kw.get("total", 0)
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def update(self, n=1):
            self.n += n

        @staticmethod
        def write(s):
            print(s)

from .base import BaseDatasetLoader


DEFAULT_BASE_URL = os.getenv("CHBMIT_BASE_URL", "https://physionet.org/files/chbmit/1.0.0")
DEFAULT_RAW_CACHE = os.getenv("CHBMIT_RAW_CACHE", "/home/ubuntu/data/chbmit")
DEFAULT_LOCAL_ONLY = os.getenv("CHBMIT_LOCAL_ONLY", "1") == "1"
DEFAULT_NATIVE_SFREQ = 256.0
DEFAULT_TARGET_SFREQ = 256
DEFAULT_NOTCH_FREQ = 60.0
DEFAULT_POPULATION = "pediatric"


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load datasets.chbmit config if available, else return empty dict."""
    path = Path(config_path or "config.yaml")
    if not path.exists():
        path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    if not path.exists():
        return {}

    with open(path) as f:
        full = yaml.safe_load(f) or {}

    dataset_cfg = (full.get("datasets", {}) or {}).get("chbmit", {}) or {}
    signal_cfg = full.get("signal", {}) or {}
    if "target_sfreq" not in dataset_cfg:
        dataset_cfg["target_sfreq"] = signal_cfg.get("target_sfreq", DEFAULT_TARGET_SFREQ)
    return dataset_cfg


class CHBMITLoader(BaseDatasetLoader):
    """CHB-MIT dataset loader."""

    def __init__(
        self,
        force: bool = False,
        raw_cache: Optional[str] = None,
        config_path: Optional[str] = None,
        base_url: Optional[str] = None,
        local_only: Optional[bool] = None,
    ):
        cfg = _load_config(config_path)

        resolved_base_url = base_url if base_url is not None else cfg.get("url", DEFAULT_BASE_URL)
        if resolved_base_url is None:
            resolved_base_url = ""
        if isinstance(resolved_base_url, str):
            resolved_base_url = resolved_base_url.strip()
        resolved_raw_cache = raw_cache or cfg.get("raw_cache", DEFAULT_RAW_CACHE)
        resolved_local_only = (
            DEFAULT_LOCAL_ONLY if local_only is None else local_only
        ) or (not resolved_base_url)

        self.local_only = resolved_local_only

        super().__init__(
            name="chbmit",
            base_url=resolved_base_url,
            raw_cache=resolved_raw_cache,
            native_sfreq=cfg.get("native_sfreq", DEFAULT_NATIVE_SFREQ),
            target_sfreq=cfg.get("target_sfreq", DEFAULT_TARGET_SFREQ),
            notch_freq=cfg.get("notch_freq", DEFAULT_NOTCH_FREQ),
            population=cfg.get("population", DEFAULT_POPULATION),
            force=force,
        )

        self._subject_info_parsed = False
        self._all_metadata: Dict[str, Dict[str, Any]] = {}

    def _subject_prefix(self) -> str:
        return "chb"

    def _fetch_records(self) -> Dict[str, List[str]]:
        # Never call remote RECORDS when local-only is enabled or URL is empty.
        if self.local_only or not self.base_url:
            edf_by_subj: Dict[str, List[str]] = {}
            for subj_dir in sorted(self.raw_cache.glob("chb*")):
                if not subj_dir.is_dir():
                    continue
                edfs = sorted(p.name for p in subj_dir.glob("*.edf"))
                if edfs:
                    edf_by_subj[subj_dir.name] = [f"{subj_dir.name}/{e}" for e in edfs]
            return edf_by_subj

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
        # Local-only mode: never download
        if self.local_only or not self.base_url:
            return

        subj_dir = self.raw_cache / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        summary_path = subj_dir / f"{subject_id}-summary.txt"
        if not summary_path.exists() or self.force:
            self._download_file(
                f"{self.base_url}/{subject_id}/{subject_id}-summary.txt",
                summary_path,
            )

        edfs = self._edf_by_subj.get(subject_id, [])
        for rel in edfs:
            edf_name = rel.split("/", 1)[1] if "/" in rel else rel
            dest = subj_dir / edf_name
            if not dest.exists() or self.force:
                tqdm.write(f"  ↓ {rel}")
                if not self._download_file(f"{self.base_url}/{rel}", dest):
                    tqdm.write(f"  [WARN] Failed to download {rel}")

    def _parse_metadata(self, subject_id: str) -> Dict[str, Any]:
        if not self._subject_info_parsed:
            self._load_subject_info()

        return self._all_metadata.get(subject_id, {"age": "NA", "sex": "NA"})

    def _parse_seizures(self, subject_id: str) -> Dict[str, List[Tuple[float, float]]]:
        summary_path = self.raw_cache / subject_id / f"{subject_id}-summary.txt"
        return _parse_chbmit_summary(summary_path)

    def _load_subject_info(self):
        info_path = self.raw_cache / "SUBJECT-INFO"

        # Download only in remote mode
        if not info_path.exists() and (not self.local_only) and self.base_url:
            self._download_file(f"{self.base_url}/SUBJECT-INFO", info_path)

        if info_path.exists():
            try:
                text = info_path.read_text(errors="ignore")
                self._all_metadata = _parse_subject_info_file(text)
            except Exception as e:
                tqdm.write(f"  [WARN] Could not parse SUBJECT-INFO: {e}")
                self._all_metadata = {}

        self._subject_info_parsed = True


def _parse_subject_info_file(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse CHB-MIT SUBJECT-INFO text into {subject_id: {"age": int, "sex": str}}.
    """
    result: Dict[str, Dict[str, Any]] = {}

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return result

    # Table format: "Case  Gender  Age"
    for i, line in enumerate(lines):
        lower = line.lower()
        if "case" in lower and ("gender" in lower or "sex" in lower):
            for row in lines[i + 1 :]:
                parts = row.split()
                if len(parts) >= 3 and parts[0].startswith("chb"):
                    subject_id = parts[0]
                    sex = "NA"
                    age: Any = "NA"
                    for p in parts[1:]:
                        if p.upper() in ("M", "F"):
                            sex = p.upper()
                        elif p.isdigit():
                            age = int(p)
                    result[subject_id] = {"age": age, "sex": sex}
            if result:
                return result

    # Key-value format
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
