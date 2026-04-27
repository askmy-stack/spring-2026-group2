"""Siena Scalp EEG: download EDFs from PhysioNet, parse seizure CSVs."""

from __future__ import annotations
import time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from .config import SIENA_CONFIG, SUBJECT_META


def _download_file(url: str, dest: Path, retries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for i in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return True
        except (urllib.error.URLError, OSError):
            if i < retries - 1:
                time.sleep(2 ** i)
    return False


def _fetch_lines(url: str, cache: Path) -> List[str]:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, cache)
        except (urllib.error.URLError, OSError):
            return []
    return [l.strip() for l in cache.read_text().splitlines() if l.strip()] if cache.exists() else []


def parse_seizure_csv(csv_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Parse Siena seizure CSV → {edf_filename: [(start, end), ...]}."""
    if not csv_path.exists():
        return {}
    result: Dict[str, List[Tuple[float, float]]] = {}
    try:
        df = pd.read_csv(csv_path)
        name_col = start_col = end_col = None
        for c in df.columns:
            cl = c.lower().strip()
            if "name" in cl or "registration" in cl:
                name_col = c
            elif "start" in cl:
                start_col = c
            elif "end" in cl:
                end_col = c
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


class SienaDownloader:
    """Download and discover Siena Scalp EEG data."""

    def __init__(self, raw_cache: Optional[str] = None, force: bool = False):
        self.raw_cache = Path(raw_cache or SIENA_CONFIG["raw_cache"])
        self.raw_cache.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.base_url = SIENA_CONFIG["base_url"]
        self._edf_by_subj: Dict[str, List[str]] = {}
        self._seizure_cache: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

    def _ensure_records(self):
        if self._edf_by_subj:
            return
        records = _fetch_lines(f"{self.base_url}/RECORDS", self.raw_cache / "RECORDS.txt")
        for rel in records:
            if rel.endswith(".edf"):
                parts = rel.split("/")
                if len(parts) >= 2:
                    self._edf_by_subj.setdefault(parts[0], []).append(rel)

    def download(self, subjects: Optional[List[str]] = None, workers: int = 1) -> None:
        self._ensure_records()
        all_available = list(self._edf_by_subj.keys()) or list(SUBJECT_META.keys())
        target = subjects or all_available
        target = [s for s in target if s in all_available]

        for subj in target:
            subj_dir = self.raw_cache / subj
            subj_dir.mkdir(parents=True, exist_ok=True)
            csv_name = f"Seizures-list-{subj}.csv"
            csv_dest = subj_dir / csv_name
            if not csv_dest.exists() or self.force:
                _download_file(f"{self.base_url}/{subj}/{csv_name}", csv_dest)

        all_edfs = [(subj, rel) for subj in target for rel in self._edf_by_subj.get(subj, [])]

        def _dl(item):
            subj, rel = item
            edf_name = rel.split("/", 1)[1] if "/" in rel else rel
            dest = self.raw_cache / subj / edf_name
            if not dest.exists() or self.force:
                tqdm.write(f"  ↓ {rel}")
                if not _download_file(f"{self.base_url}/{rel}", dest):
                    raise RuntimeError(f"Failed: {rel}")

        if all_edfs:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_dl, item): item for item in all_edfs}
                with tqdm(total=len(all_edfs), desc="[siena] Downloading", unit="file") as pbar:
                    for fut in as_completed(futures):
                        if exc := fut.exception():
                            tqdm.write(f"  [WARN] {exc}")
                        pbar.update(1)

    def list_subjects(self) -> List[str]:
        dirs = sorted(p.name for p in self.raw_cache.iterdir()
                      if p.is_dir() and p.name.startswith("PN"))
        if dirs:
            return dirs
        self._ensure_records()
        return sorted(self._edf_by_subj.keys()) if self._edf_by_subj else sorted(SUBJECT_META.keys())

    def get_edf_paths(self, subject_id: str) -> List[Path]:
        subj_dir = self.raw_cache / subject_id
        return sorted(subj_dir.glob("*.edf")) if subj_dir.exists() else []

    def get_metadata(self, subject_id: str) -> Dict[str, Any]:
        meta = SUBJECT_META.get(subject_id, {"age": "NA", "sex": "NA"})
        return {"subject_id": subject_id, "age": meta["age"], "sex": meta["sex"], "dataset": "siena"}

    def get_seizure_intervals(self, edf_path: Path) -> List[Tuple[float, float]]:
        subject_id = edf_path.parent.name
        if subject_id not in self._seizure_cache:
            csv_path = self.raw_cache / subject_id / f"Seizures-list-{subject_id}.csv"
            self._seizure_cache[subject_id] = parse_seizure_csv(csv_path)
        seizures = self._seizure_cache[subject_id]
        return seizures.get(edf_path.name, seizures.get("__all__", []))