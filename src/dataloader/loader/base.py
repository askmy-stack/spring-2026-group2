"""
BaseDatasetLoader — Abstract parent class for all EEG dataset loaders.

Every dataset (CHB-MIT, Siena, future datasets) inherits from this class
and implements the abstract methods for dataset-specific logic:
    - _fetch_records()     → discover what EDF files exist
    - _download_subject()  → download one subject's files
    - _parse_metadata()    → parse age/sex from source (NOT hardcoded)
    - _parse_seizures()    → parse seizure annotations for a subject

Shared logic lives here:
    - download()           → orchestrate downloading all subjects
    - list_subjects()      → return available subject IDs
    - get_edf_paths()      → return EDF files for a subject
    - get_metadata()       → return parsed metadata dict
    - get_seizure_intervals() → return seizure times for an EDF
"""

from __future__ import annotations
from abc import ABC, abstractmethod
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


class BaseDatasetLoader(ABC):
    """Abstract base class for EEG dataset loaders."""

    def __init__(
        self,
        name: str,
        base_url: str,
        raw_cache: str,
        native_sfreq: float,
        target_sfreq: int,
        notch_freq: float,
        population: str = "unknown",
        force: bool = False,
    ):
        self.name = name
        self.base_url = base_url
        self.raw_cache = Path(raw_cache)
        self.raw_cache.mkdir(parents=True, exist_ok=True)
        self.native_sfreq = native_sfreq
        self.target_sfreq = target_sfreq
        self.notch_freq = notch_freq
        self.population = population
        self.force = force

        # Internal caches — populated lazily
        self._edf_by_subj: Dict[str, List[str]] = {}
        self._seizure_cache: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}

    # ═══════════════════════════════════════════════════════
    # ABSTRACT — subclasses MUST implement these
    # ═══════════════════════════════════════════════════════

    @abstractmethod
    def _fetch_records(self) -> Dict[str, List[str]]:
        """
        Discover available EDF files from the remote source.

        Returns:
            {subject_id: [relative/path/to/file.edf, ...]}

        Example implementation: download RECORDS file, parse subject/file mappings.
        """
        ...

    @abstractmethod
    def _download_subject(self, subject_id: str) -> None:
        """
        Download all files for a single subject (EDFs + annotations).

        Args:
            subject_id: e.g. "chb01" or "PN00"
        """
        ...

    @abstractmethod
    def _parse_metadata(self, subject_id: str) -> Dict[str, Any]:
        """
        Parse subject metadata (age, sex) from downloaded source files.
        NOT hardcoded — read from the actual dataset files.

        Args:
            subject_id: e.g. "chb01"

        Returns:
            {"age": <int or "NA">, "sex": <"M"/"F" or "NA">}
        """
        ...

    @abstractmethod
    def _parse_seizures(self, subject_id: str) -> Dict[str, List[Tuple[float, float]]]:
        """
        Parse seizure annotations for a subject from downloaded files.

        Args:
            subject_id: e.g. "chb01"

        Returns:
            {edf_filename: [(start_sec, end_sec), ...]}
        """
        ...

    @abstractmethod
    def _subject_prefix(self) -> str:
        """
        Return the prefix that identifies subject directories.
        e.g. "chb" for CHB-MIT, "PN" for Siena.
        """
        ...

    # ═══════════════════════════════════════════════════════
    # SHARED — inherited by all subclasses
    # ═══════════════════════════════════════════════════════

    def _ensure_records(self):
        """Populate _edf_by_subj if not already done."""
        if not self._edf_by_subj:
            self._edf_by_subj = self._fetch_records()

    def download(self, subjects: Optional[List[str]] = None, workers: int = 1) -> None:
        """
        Download EDF files for requested subjects.

        Args:
            subjects: List of subject IDs. None = all available.
            workers: Not used currently (sequential download).
        """
        self._ensure_records()
        available = list(self._edf_by_subj.keys())
        target = subjects or available
        target = [s for s in target if s in available]

        if not target:
            print(f"  [WARN] No subjects to download for {self.name}")
            return

        for subj in target:
            self._download_subject(subj)

    def list_subjects(self) -> List[str]:
        """Return sorted list of available subject IDs."""
        # First: check what's already downloaded
        dirs = sorted(
            p.name for p in self.raw_cache.iterdir()
            if p.is_dir() and p.name.startswith(self._subject_prefix())
        )
        if dirs:
            return dirs

        # Fallback: check remote records
        self._ensure_records()
        return sorted(self._edf_by_subj.keys())

    def get_edf_paths(self, subject_id: str) -> List[Path]:
        """Return sorted list of EDF file paths for a subject."""
        subj_dir = self.raw_cache / subject_id
        if not subj_dir.exists():
            return []
        return sorted(subj_dir.glob("*.edf"))

    def get_metadata(self, subject_id: str) -> Dict[str, Any]:
        """
        Get metadata for a subject. Parses from source files, caches result.

        Returns:
            {"subject_id": str, "age": int|str, "sex": str, "dataset": str}
        """
        if subject_id not in self._metadata_cache:
            try:
                parsed = self._parse_metadata(subject_id)
            except Exception as e:
                tqdm.write(f"  [WARN] Metadata parse failed for {subject_id}: {e}")
                parsed = {"age": "NA", "sex": "NA"}

            self._metadata_cache[subject_id] = {
                "subject_id": subject_id,
                "age": parsed.get("age", "NA"),
                "sex": parsed.get("sex", "NA"),
                "dataset": self.name,
            }

        return self._metadata_cache[subject_id]

    def get_seizure_intervals(self, edf_path: Path) -> List[Tuple[float, float]]:
        """
        Get seizure intervals for an EDF file. Parses from source, caches result.

        Returns:
            [(start_sec, end_sec), ...]
        """
        subject_id = edf_path.parent.name
        if subject_id not in self._seizure_cache:
            try:
                self._seizure_cache[subject_id] = self._parse_seizures(subject_id)
            except Exception as e:
                tqdm.write(f"  [WARN] Seizure parse failed for {subject_id}: {e}")
                self._seizure_cache[subject_id] = {}

        return self._seizure_cache[subject_id].get(edf_path.name, [])

    def get_config(self) -> Dict[str, Any]:
        """Return config dict for this dataset."""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "native_sfreq": self.native_sfreq,
            "target_sfreq": self.target_sfreq,
            "notch_freq": self.notch_freq,
            "raw_cache": str(self.raw_cache),
            "population": self.population,
        }

    # ═══════════════════════════════════════════════════════
    # UTILITIES — shared download helpers
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _download_file(url: str, dest: Path, retries: int = 3) -> bool:
        """Download a single file with retries."""
        import time
        import urllib.request
        import urllib.error

        dest.parent.mkdir(parents=True, exist_ok=True)
        for i in range(retries):
            try:
                urllib.request.urlretrieve(url, dest)
                return True
            except (urllib.error.URLError, OSError):
                if i < retries - 1:
                    time.sleep(2 ** i)
        return False

    @staticmethod
    def _fetch_lines(url: str, cache: Path) -> List[str]:
        """Download a text file and return non-empty lines. Caches locally."""
        import urllib.request
        import urllib.error

        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            try:
                urllib.request.urlretrieve(url, cache)
            except (urllib.error.URLError, OSError):
                return []
        if cache.exists():
            return [l.strip() for l in cache.read_text().splitlines() if l.strip()]
        return []