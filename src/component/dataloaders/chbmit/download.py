"""CHB-MIT: download EDFs from PhysioNet, parse summary.txt for seizures."""

from __future__ import annotations
import re, time, urllib.request, urllib.error
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

from .config import CHBMIT_CONFIG, SUBJECT_META


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


def parse_summary(path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """Parse {edf_filename: [(start_sec, end_sec)]} from CHB-MIT summary.txt."""
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    out: Dict[str, List[Tuple[float, float]]] = {}
    for block in re.split(r"(?=File Name\s*[:\-])", text, flags=re.I):
        fm = re.search(r"File Name\s*[:\-]\s*(\S+\.edf)", block, re.I)
        if not fm:
            continue
        starts = [int(m.group(1)) for m in re.finditer(r"Seizure.*?Start.*?:\s*(\d+)\s*sec", block, re.I)]
        ends   = [int(m.group(1)) for m in re.finditer(r"Seizure.*?End.*?:\s*(\d+)\s*sec",   block, re.I)]
        n = min(len(starts), len(ends))
        if n != len(starts) or n != len(ends):
            tqdm.write(f"  [WARN] {fm.group(1)}: {len(starts)} starts vs {len(ends)} ends")
        out[fm.group(1)] = [(float(s), float(e)) for s, e in zip(starts[:n], ends[:n])]
    return out


class CHBMITDownloader:
    """Download and discover CHB-MIT data."""

    def __init__(self, raw_cache: Optional[str] = None, force: bool = False):
        self.raw_cache = Path(raw_cache or CHBMIT_CONFIG["raw_cache"])
        self.raw_cache.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.base_url = CHBMIT_CONFIG["base_url"]
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
        target = subjects or list(self._edf_by_subj.keys())
        target = [s for s in target if s in self._edf_by_subj]

        for subj in target:
            subj_dir = self.raw_cache / subj
            subj_dir.mkdir(parents=True, exist_ok=True)
            summary_path = subj_dir / f"{subj}-summary.txt"
            if not summary_path.exists() or self.force:
                _download_file(f"{self.base_url}/{subj}/{subj}-summary.txt", summary_path)

        all_edfs = [(subj, rel) for subj in target for rel in self._edf_by_subj.get(subj, [])]

        def _dl(item):
            subj, rel = item
            edf_name = rel.split("/", 1)[1]
            dest = self.raw_cache / subj / edf_name
            if not dest.exists() or self.force:
                tqdm.write(f"  ↓ {rel}")
                if not _download_file(f"{self.base_url}/{rel}", dest):
                    raise RuntimeError(f"Failed: {rel}")

        if all_edfs:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_dl, item): item for item in all_edfs}
                with tqdm(total=len(all_edfs), desc="[chbmit] Downloading", unit="file") as pbar:
                    for fut in as_completed(futures):
                        if exc := fut.exception():
                            tqdm.write(f"  [WARN] {exc}")
                        pbar.update(1)

    def list_subjects(self) -> List[str]:
        dirs = sorted(p.name for p in self.raw_cache.iterdir()
                      if p.is_dir() and p.name.startswith("chb"))
        if dirs:
            return dirs
        self._ensure_records()
        return sorted(self._edf_by_subj.keys()) if self._edf_by_subj else sorted(SUBJECT_META.keys())

    def get_edf_paths(self, subject_id: str) -> List[Path]:
        subj_dir = self.raw_cache / subject_id
        return sorted(subj_dir.glob("*.edf")) if subj_dir.exists() else []

    def get_metadata(self, subject_id: str) -> Dict[str, Any]:
        meta = SUBJECT_META.get(subject_id, {"age": "NA", "sex": "NA"})
        return {"subject_id": subject_id, "age": meta["age"], "sex": meta["sex"], "dataset": "chbmit"}

    def get_seizure_intervals(self, edf_path: Path) -> List[Tuple[float, float]]:
        subject_id = edf_path.parent.name
        if subject_id not in self._seizure_cache:
            summary = self.raw_cache / subject_id / f"{subject_id}-summary.txt"
            self._seizure_cache[subject_id] = parse_summary(summary)
        return self._seizure_cache[subject_id].get(edf_path.name, [])