"""
Comprehensive tests for the wrappers architecture.

All tests run OFFLINE — no network access required.
Uses synthetic EDF files and directory structures.

Test categories:
    1. Data source adapters (base, chbmit, siena, local_folder, bids_existing)
    2. Acquisition wrappers (detect, discover, acquire, normalize)
    3. Pipeline runner wrappers (handoff → processing → outputs)
    4. Sanity checks (labels, splits, shapes, edge cases)

Run:
    cd src/
    python -m tests.test_wrappers
    # or
    python tests/test_wrappers.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasources.base import SubjectHandoff, DatasetHandoff
from datasources.registry import get_source, list_sources, auto_detect
from datasources.chbmit import CHBMITSource
from datasources.siena import SienaSource
from datasources.local_folder import LocalFolderSource
from datasources.bids_existing import BIDSExistingSource
from dataloaders.wrappers import acquire, _detect


# ── Test utilities ──────────────────────────────────────────

_passed = 0
_failed = 0
_errors: List[str] = []


def _test(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  ✅ {name}")
    else:
        _failed += 1
        msg = f"  ❌ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        _errors.append(msg)


def _create_synthetic_edf(path: Path, n_channels: int = 23, duration_sec: int = 60, sfreq: int = 256):
    """
    Create a minimal synthetic EDF file.
    Uses MNE to ensure it's valid and readable by read_raw().
    """
    try:
        import mne
        data = np.random.randn(n_channels, sfreq * duration_sec) * 1e-5
        ch_names = [f"EEG{i+1:03d}" for i in range(n_channels)]
        ch_types = ["eeg"] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw.export(str(path), fmt="edf", overwrite=True)
        return True
    except Exception as e:
        # If MNE export fails, create a dummy file for structural tests
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00" * 1024)
        return False


def _create_events_tsv(path: Path, seizures: List[Tuple[float, float]]):
    """Create a BIDS-style events TSV with seizure annotations."""
    rows = [{"onset": s, "duration": e - s, "trial_type": "seizure", "value": 1} for s, e in seizures]
    df = pd.DataFrame(rows, columns=["onset", "duration", "trial_type", "value"])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def _create_bids_structure(root: Path, subjects: List[str], seizures_per_subj: dict = None):
    """Create a synthetic BIDS directory."""
    root.mkdir(parents=True, exist_ok=True)

    # dataset_description.json (BIDS marker)
    (root / "dataset_description.json").write_text(json.dumps({
        "Name": "Test Dataset", "BIDSVersion": "1.8.0"
    }))

    # participants.tsv
    rows = [{"participant_id": f"sub-{s}", "age": 20 + i, "sex": "F" if i % 2 == 0 else "M"}
            for i, s in enumerate(subjects)]
    pd.DataFrame(rows).to_csv(root / "participants.tsv", sep="\t", index=False)

    for s in subjects:
        eeg_dir = root / f"sub-{s}" / "ses-001" / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        prefix = eeg_dir / f"sub-{s}_ses-001_task-seizure_eeg"
        _create_synthetic_edf(prefix.with_suffix(".edf"), n_channels=16, duration_sec=30)

        # Sidecar JSON
        (prefix.with_suffix(".json")).write_text(json.dumps({
            "SamplingFrequency": 256, "EEGChannelCount": 16,
            "TaskName": "seizure_monitoring"
        }))

        # Events TSV
        seizures = (seizures_per_subj or {}).get(s, [])
        events_path = eeg_dir / f"sub-{s}_ses-001_task-seizure_events.tsv"
        _create_events_tsv(events_path, seizures)


def _create_local_edf_folder(root: Path, layout: str = "nested"):
    """Create a local folder with EDF files + annotations."""
    root.mkdir(parents=True, exist_ok=True)

    if layout == "nested":
        for subj in ["subj01", "subj02", "subj03"]:
            subj_dir = root / subj
            subj_dir.mkdir(parents=True, exist_ok=True)
            _create_synthetic_edf(subj_dir / f"{subj}_01.edf", duration_sec=20)
            _create_synthetic_edf(subj_dir / f"{subj}_02.edf", duration_sec=20)
            # Add events for first EDF
            _create_events_tsv(
                subj_dir / f"{subj}_01_events.tsv",
                [(5.0, 10.0), (15.0, 18.0)]
            )
    elif layout == "flat":
        for subj in ["subj01", "subj02"]:
            for run in ["01", "02"]:
                _create_synthetic_edf(root / f"{subj}_{run}.edf", duration_sec=20)
            _create_events_tsv(root / f"{subj}_01_events.tsv", [(3.0, 7.0)])


# ════════════════════════════════════════════════════════════
# TEST GROUP 1: Data Source Base & Dataclasses
# ════════════════════════════════════════════════════════════

def test_subject_handoff():
    """Test SubjectHandoff dataclass."""
    print("\n── SubjectHandoff ──")
    h = SubjectHandoff(
        subject_id="test01",
        edf_paths=[Path("/a.edf"), Path("/b.edf")],
        seizure_intervals={"a.edf": [(10.0, 20.0)], "b.edf": []},
        metadata={"subject_id": "test01", "age": 25, "sex": "M"},
        native_sfreq=256.0,
        dataset_name="test",
    )
    _test("subject_id correct", h.subject_id == "test01")
    _test("edf_paths has 2 files", len(h.edf_paths) == 2)
    _test("seizure_intervals for a.edf", len(h.seizure_intervals["a.edf"]) == 1)
    _test("is_already_bids default False", h.is_already_bids is False)


def test_dataset_handoff():
    """Test DatasetHandoff dataclass."""
    print("\n── DatasetHandoff ──")
    s1 = SubjectHandoff("s1", [Path("/a.edf")], {"a.edf": [(1, 5)]}, {"subject_id": "s1", "age": 10, "sex": "F"}, 256.0, "test")
    s2 = SubjectHandoff("s2", [Path("/b.edf"), Path("/c.edf")], {"b.edf": [], "c.edf": [(2, 4)]}, {"subject_id": "s2", "age": 20, "sex": "M"}, 256.0, "test")
    dh = DatasetHandoff(dataset_name="test", subjects=[s1, s2], native_sfreq=256.0)

    _test("subject_ids", dh.subject_ids == ["s1", "s2"])
    _test("total_edfs = 3", dh.total_edfs == 3)
    _test("total_seizure_files = 2", dh.total_seizure_files == 2)


# ════════════════════════════════════════════════════════════
# TEST GROUP 2: Registry
# ════════════════════════════════════════════════════════════

def test_registry():
    """Test source registry."""
    print("\n── Registry ──")
    sources = list_sources()
    _test("chbmit registered", "chbmit" in sources)
    _test("siena registered", "siena" in sources)
    _test("local_folder registered", "local_folder" in sources)
    _test("bids_existing registered", "bids_existing" in sources)
    _test("at least 4 sources", len(sources) >= 4)

    # get_source should work
    with tempfile.TemporaryDirectory() as tmp:
        src = get_source("chbmit", raw_cache=tmp)
        _test("get_source returns CHBMITSource", isinstance(src, CHBMITSource))
        _test("name is chbmit", src.name == "chbmit")
        _test("native_sfreq is 256", src.native_sfreq == 256.0)

    # Unknown source should raise
    try:
        get_source("nonexistent")
        _test("unknown source raises", False, "Should have raised ValueError")
    except ValueError:
        _test("unknown source raises ValueError", True)


# ════════════════════════════════════════════════════════════
# TEST GROUP 3: CHB-MIT Adapter (offline)
# ════════════════════════════════════════════════════════════

def test_chbmit_adapter():
    """Test CHB-MIT adapter offline (no downloads)."""
    print("\n── CHBMITSource ──")
    with tempfile.TemporaryDirectory() as tmp:
        src = CHBMITSource(raw_cache=tmp)

        _test("name is chbmit", src.name == "chbmit")
        _test("native_sfreq is 256", src.native_sfreq == 256.0)
        _test("is_already_bids is False", src.is_already_bids is False)

        # Test metadata for known subjects
        meta = src.get_metadata("chb01")
        _test("chb01 age is 11", meta["age"] == 11)
        _test("chb01 sex is F", meta["sex"] == "F")
        _test("metadata has dataset field", meta["dataset"] == "chbmit")

        # Unknown subject
        meta_unk = src.get_metadata("chb99")
        _test("unknown subject has NA age", meta_unk["age"] == "NA")

        # Empty dir: no EDFs
        _test("no edfs before download", src.get_edf_paths("chb01") == [])

        # Create fake subject dir with EDFs
        subj_dir = Path(tmp) / "chb01"
        subj_dir.mkdir()
        (subj_dir / "chb01_01.edf").write_bytes(b"\x00" * 100)
        (subj_dir / "chb01_02.edf").write_bytes(b"\x00" * 100)

        edfs = src.get_edf_paths("chb01")
        _test("finds 2 EDFs", len(edfs) == 2)

        subjects = src.list_subjects()
        _test("lists chb01 from disk", "chb01" in subjects)

        # Seizure parsing with fake summary
        summary = subj_dir / "chb01-summary.txt"
        summary.write_text(
            "File Name: chb01_01.edf\n"
            "Seizure 1 Start Time: 100 seconds\n"
            "Seizure 1 End Time: 120 seconds\n"
            "Seizure 2 Start Time: 200 seconds\n"
            "Seizure 2 End Time: 210 seconds\n"
            "\n"
            "File Name: chb01_02.edf\n"
            "Number of Seizures in File: 0\n"
        )
        intervals_01 = src.get_seizure_intervals(subj_dir / "chb01_01.edf")
        intervals_02 = src.get_seizure_intervals(subj_dir / "chb01_02.edf")
        _test("chb01_01 has 2 seizures", len(intervals_01) == 2)
        _test("first seizure is (100, 120)", intervals_01[0] == (100.0, 120.0))
        _test("chb01_02 has 0 seizures", len(intervals_02) == 0)

        # Validation
        ok, msg = src.validate()
        _test("validation passes with data", ok, msg)

        # Handoff
        handoff = src.produce_handoff(subjects=["chb01"])
        _test("handoff has 1 subject", len(handoff.subjects) == 1)
        _test("handoff total_edfs = 2", handoff.total_edfs == 2)
        _test("handoff seizure_files = 1", handoff.total_seizure_files == 1)


# ════════════════════════════════════════════════════════════
# TEST GROUP 4: Siena Adapter (offline)
# ════════════════════════════════════════════════════════════

def test_siena_adapter():
    """Test Siena adapter offline."""
    print("\n── SienaSource ──")
    with tempfile.TemporaryDirectory() as tmp:
        src = SienaSource(raw_cache=tmp)

        _test("name is siena", src.name == "siena")
        _test("native_sfreq is 512", src.native_sfreq == 512.0)

        meta = src.get_metadata("PN00")
        _test("PN00 age is 50", meta["age"] == 50)
        _test("PN00 sex is F", meta["sex"] == "F")

        # Create fake Siena data
        subj_dir = Path(tmp) / "PN00"
        subj_dir.mkdir()
        (subj_dir / "PN00_1.edf").write_bytes(b"\x00" * 100)
        (subj_dir / "PN00_2.edf").write_bytes(b"\x00" * 100)

        # Create seizure CSV
        seizure_csv = subj_dir / "Seizures-list-PN00.csv"
        pd.DataFrame([
            {"Registration Name": "PN00_1", "Seizure Start (sec.)": 50, "Seizure End (sec.)": 70},
            {"Registration Name": "PN00_1", "Seizure Start (sec.)": 120, "Seizure End (sec.)": 135},
        ]).to_csv(seizure_csv, index=False)

        intervals_1 = src.get_seizure_intervals(subj_dir / "PN00_1.edf")
        intervals_2 = src.get_seizure_intervals(subj_dir / "PN00_2.edf")
        _test("PN00_1 has 2 seizures", len(intervals_1) == 2)
        _test("first seizure is (50, 70)", intervals_1[0] == (50.0, 70.0))
        _test("PN00_2 has 0 seizures", len(intervals_2) == 0)

        ok, msg = src.validate()
        _test("validation passes", ok, msg)


# ════════════════════════════════════════════════════════════
# TEST GROUP 5: Local Folder Adapter
# ════════════════════════════════════════════════════════════

def test_local_folder_nested():
    """Test local folder adapter with nested layout."""
    print("\n── LocalFolderSource (nested) ──")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _create_local_edf_folder(root, layout="nested")
        src = LocalFolderSource(raw_cache=str(root))
        src.download()  # No-op for local

        subjects = src.list_subjects()
        _test("finds 3 subjects", len(subjects) == 3)
        _test("subj01 in subjects", "subj01" in subjects)

        edfs = src.get_edf_paths("subj01")
        _test("subj01 has 2 EDFs", len(edfs) == 2)

        # Seizure intervals from events TSV
        first_edf = edfs[0]
        intervals = src.get_seizure_intervals(first_edf)
        _test("subj01_01 has 2 seizure intervals", len(intervals) == 2)
        _test("first interval is (5, 10)", intervals[0] == (5.0, 10.0))

        # Second EDF has no events file
        second_edf = edfs[1]
        intervals_2 = src.get_seizure_intervals(second_edf)
        _test("subj01_02 has 0 seizures", len(intervals_2) == 0)

        meta = src.get_metadata("subj01")
        _test("metadata has subject_id", meta["subject_id"] == "subj01")
        _test("metadata age is NA", meta["age"] == "NA")

        ok, msg = src.validate()
        _test("validation passes", ok, msg)

        handoff = src.produce_handoff()
        _test("handoff has 3 subjects", len(handoff.subjects) == 3)


def test_local_folder_flat():
    """Test local folder adapter with flat layout."""
    print("\n── LocalFolderSource (flat) ──")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _create_local_edf_folder(root, layout="flat")
        src = LocalFolderSource(raw_cache=str(root))
        src.download()

        subjects = src.list_subjects()
        _test("finds 2 subjects from flat", len(subjects) == 2)

        edfs = src.get_edf_paths("subj01")
        _test("subj01 has 2 EDFs from flat", len(edfs) == 2)


def test_local_folder_json_annotations():
    """Test JSON annotation parsing."""
    print("\n── LocalFolderSource (JSON annotations) ──")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "subj01"
        root.mkdir(parents=True)
        _create_synthetic_edf(root / "recording.edf", duration_sec=20)

        # JSON annotations
        (root / "recording.json").write_text(json.dumps({
            "seizures": [
                {"onset": 5.0, "end": 10.0},
                {"onset": 15.0, "duration": 3.0},
            ]
        }))

        src = LocalFolderSource(raw_cache=str(root.parent))
        src.download()
        edfs = src.get_edf_paths("subj01")
        intervals = src.get_seizure_intervals(edfs[0])
        _test("JSON: finds 2 seizures", len(intervals) == 2)
        _test("JSON: first is (5, 10)", intervals[0] == (5.0, 10.0))
        _test("JSON: second uses duration → (15, 18)", intervals[1] == (15.0, 18.0))


def test_local_folder_csv_annotations():
    """Test CSV annotation parsing."""
    print("\n── LocalFolderSource (CSV annotations) ──")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "subj01"
        root.mkdir(parents=True)
        _create_synthetic_edf(root / "rec.edf", duration_sec=20)

        pd.DataFrame([
            {"start_time": 3.0, "end_time": 8.0},
            {"start_time": 12.0, "end_time": 15.0},
        ]).to_csv(root / "rec_annotations.csv", index=False)

        src = LocalFolderSource(raw_cache=str(root.parent))
        src.download()
        edfs = src.get_edf_paths("subj01")
        intervals = src.get_seizure_intervals(edfs[0])
        _test("CSV: finds 2 seizures", len(intervals) == 2)
        _test("CSV: first is (3, 8)", intervals[0] == (3.0, 8.0))


# ════════════════════════════════════════════════════════════
# TEST GROUP 6: BIDS Existing Adapter
# ════════════════════════════════════════════════════════════

def test_bids_existing():
    """Test pre-existing BIDS adapter."""
    print("\n── BIDSExistingSource ──")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _create_bids_structure(root, ["s01", "s02", "s03"], {
            "s01": [(10.0, 20.0), (50.0, 60.0)],
            "s02": [(5.0, 15.0)],
            "s03": [],
        })

        src = BIDSExistingSource(raw_cache=str(root))

        _test("name is bids_existing", src.name == "bids_existing")
        _test("is_already_bids is True", src.is_already_bids is True)

        subjects = src.list_subjects()
        _test("finds 3 subjects", len(subjects) == 3)
        _test("s01 in subjects", "s01" in subjects)

        edfs = src.get_edf_paths("s01")
        _test("s01 has EDFs", len(edfs) >= 1)

        meta = src.get_metadata("s01")
        _test("s01 age from participants.tsv", meta["age"] == 20)
        _test("s01 sex from participants.tsv", meta["sex"] == "F")

        # Seizure intervals from events TSV
        if edfs:
            intervals = src.get_seizure_intervals(edfs[0])
            _test("s01 has 2 seizure intervals", len(intervals) == 2)

        ok, msg = src.validate()
        _test("validation passes", ok, msg)

        # Sidecar processing check
        if edfs:
            cfg = {"signal": {"target_sfreq": 256}, "channels": {"target_count": 16}}
            status = src.check_processing_status(edfs[0], cfg)
            _test("BIDS: convert_to_bids is False", status["convert_to_bids"] is False)
            _test("BIDS: resample check works", isinstance(status["resample"], bool))

        handoff = src.produce_handoff()
        _test("handoff is_already_bids", handoff.is_already_bids is True)
        _test("handoff bids_root set", handoff.bids_root is not None)


def test_bids_not_bids():
    """Test BIDS adapter rejects non-BIDS directory."""
    print("\n── BIDSExistingSource (not BIDS) ──")
    with tempfile.TemporaryDirectory() as tmp:
        src = BIDSExistingSource(raw_cache=tmp)
        ok, msg = src.validate()
        _test("rejects non-BIDS dir", not ok)
        _test("error mentions dataset_description", "dataset_description" in msg.lower() or "sub-" in msg.lower())


# ════════════════════════════════════════════════════════════
# TEST GROUP 7: Auto-Detection
# ════════════════════════════════════════════════════════════

def test_auto_detect():
    """Test auto_detect picks the right adapter."""
    print("\n── Auto-Detection ──")

    # BIDS directory
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _create_bids_structure(root, ["s01"])
        src = auto_detect(str(root))
        _test("BIDS detected", isinstance(src, BIDSExistingSource))

    # Local EDF folder
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _create_local_edf_folder(root, layout="nested")
        src = auto_detect(str(root))
        _test("local EDF detected", isinstance(src, LocalFolderSource))

    # Non-existent path
    try:
        auto_detect("/nonexistent/path/12345")
        _test("nonexistent raises error", False)
    except FileNotFoundError:
        _test("nonexistent raises FileNotFoundError", True)

    # Empty directory
    with tempfile.TemporaryDirectory() as tmp:
        try:
            auto_detect(tmp)
            _test("empty dir raises error", False)
        except ValueError:
            _test("empty dir raises ValueError", True)


# ════════════════════════════════════════════════════════════
# TEST GROUP 8: Acquisition Wrapper
# ════════════════════════════════════════════════════════════

def test_acquisition_detect():
    """Test the _detect function in acquisition wrappers."""
    print("\n── Acquisition _detect ──")

    # Known dataset name
    with tempfile.TemporaryDirectory() as tmp:
        adapter = _detect("chbmit", raw_cache=tmp, force=False)
        _test("chbmit detected by name", isinstance(adapter, CHBMITSource))

    with tempfile.TemporaryDirectory() as tmp:
        adapter = _detect("siena", raw_cache=tmp, force=False)
        _test("siena detected by name", isinstance(adapter, SienaSource))

    # Local path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _create_bids_structure(root, ["s01"])
        adapter = _detect(str(root), raw_cache=None, force=False)
        _test("local BIDS path detected", isinstance(adapter, BIDSExistingSource))

    # Unknown name, no path
    try:
        _detect("nonexistent_dataset", raw_cache=None, force=False)
        _test("unknown raises error", False)
    except ValueError:
        _test("unknown source raises ValueError", True)


def test_acquisition_full_bids():
    """Test full acquisition flow with pre-existing BIDS."""
    print("\n── Acquisition (BIDS path) ──")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _create_bids_structure(root, ["s01", "s02", "s03"], {
            "s01": [(10.0, 20.0)],
        })

        handoff = acquire(str(root))
        _test("handoff is DatasetHandoff", isinstance(handoff, DatasetHandoff))
        _test("handoff has 3 subjects", len(handoff.subjects) == 3)
        _test("handoff is_already_bids", handoff.is_already_bids is True)
        _test("s01 has seizure intervals", len(handoff.subjects[0].seizure_intervals) > 0)


def test_acquisition_local_folder():
    """Test full acquisition flow with local EDF folder."""
    print("\n── Acquisition (local folder) ──")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _create_local_edf_folder(root, layout="nested")

        handoff = acquire(str(root))
        _test("handoff created", isinstance(handoff, DatasetHandoff))
        _test("handoff has subjects", len(handoff.subjects) > 0)
        _test("handoff is NOT bids", handoff.is_already_bids is False)


# ════════════════════════════════════════════════════════════
# TEST GROUP 9: Handoff Sanity Checks
# ════════════════════════════════════════════════════════════

def test_handoff_sanity():
    """Validate handoff objects have correct structure and types."""
    print("\n── Handoff Sanity ──")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _create_bids_structure(root, ["s01", "s02"], {
            "s01": [(5.0, 15.0)],
            "s02": [],
        })

        handoff = acquire(str(root))

        for subj in handoff.subjects:
            _test(f"{subj.subject_id}: has subject_id", isinstance(subj.subject_id, str) and len(subj.subject_id) > 0)
            _test(f"{subj.subject_id}: edf_paths is list", isinstance(subj.edf_paths, list))
            _test(f"{subj.subject_id}: all paths are Path", all(isinstance(p, Path) for p in subj.edf_paths))
            _test(f"{subj.subject_id}: seizure_intervals is dict", isinstance(subj.seizure_intervals, dict))
            _test(f"{subj.subject_id}: metadata has age", "age" in subj.metadata)
            _test(f"{subj.subject_id}: metadata has sex", "sex" in subj.metadata)
            _test(f"{subj.subject_id}: native_sfreq > 0", subj.native_sfreq > 0)

            # Seizure intervals must be list of (float, float) tuples
            for edf_name, intervals in subj.seizure_intervals.items():
                _test(f"{subj.subject_id}/{edf_name}: intervals is list", isinstance(intervals, list))
                for iv in intervals:
                    _test(f"  interval is tuple of 2", isinstance(iv, tuple) and len(iv) == 2)
                    _test(f"  end > start", iv[1] > iv[0])


# ════════════════════════════════════════════════════════════
# TEST GROUP 10: Edge Cases
# ════════════════════════════════════════════════════════════

def test_edge_empty_dataset():
    """Test handling of empty datasets."""
    print("\n── Edge: Empty Dataset ──")
    with tempfile.TemporaryDirectory() as tmp:
        src = LocalFolderSource(raw_cache=tmp)
        src.download()
        subjects = src.list_subjects()
        _test("empty dir: 0 subjects", len(subjects) == 0)

        ok, msg = src.validate()
        _test("empty dir: validation fails", not ok)

        handoff = src.produce_handoff()
        _test("empty handoff: 0 subjects", len(handoff.subjects) == 0)
        _test("empty handoff: 0 edfs", handoff.total_edfs == 0)


def test_edge_subject_filter():
    """Test that subject filtering works correctly."""
    print("\n── Edge: Subject Filtering ──")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _create_bids_structure(root, ["s01", "s02", "s03", "s04"])

        src = BIDSExistingSource(raw_cache=str(root))
        handoff_all = src.produce_handoff()
        _test("all subjects: 4", len(handoff_all.subjects) == 4)

        handoff_filtered = src.produce_handoff(subjects=["s01", "s03"])
        _test("filtered: 2 subjects", len(handoff_filtered.subjects) == 2)
        _test("filtered: correct IDs", handoff_filtered.subject_ids == ["s01", "s03"])

        handoff_nonexistent = src.produce_handoff(subjects=["s99"])
        _test("nonexistent subject: 0", len(handoff_nonexistent.subjects) == 0)


def test_edge_mismatched_seizure_starts_ends():
    """Test CHB-MIT summary with mismatched start/end counts."""
    print("\n── Edge: Mismatched Seizure Starts/Ends ──")
    with tempfile.TemporaryDirectory() as tmp:
        subj_dir = Path(tmp) / "chb01"
        subj_dir.mkdir(parents=True)
        (subj_dir / "chb01_01.edf").write_bytes(b"\x00" * 100)

        # 2 starts but only 1 end
        summary = subj_dir / "chb01-summary.txt"
        summary.write_text(
            "File Name: chb01_01.edf\n"
            "Seizure 1 Start Time: 100 seconds\n"
            "Seizure 1 End Time: 120 seconds\n"
            "Seizure 2 Start Time: 200 seconds\n"
        )
        src = CHBMITSource(raw_cache=tmp)
        intervals = src.get_seizure_intervals(subj_dir / "chb01_01.edf")
        _test("mismatched: truncated to 1 seizure", len(intervals) == 1)
        _test("mismatched: correct interval", intervals[0] == (100.0, 120.0))


def test_edge_special_characters():
    """Test handling of unusual filenames."""
    print("\n── Edge: Special Characters ──")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "subj-01_test"
        root.mkdir(parents=True)
        _create_synthetic_edf(root / "rec (1).edf", duration_sec=10)
        _create_synthetic_edf(root / "rec (2).edf", duration_sec=10)

        src = LocalFolderSource(raw_cache=str(root.parent))
        src.download()
        subjects = src.list_subjects()
        _test("handles special chars in dirname", len(subjects) > 0)


def test_edge_no_seizure_annotations():
    """Test that datasets without any seizure annotations work."""
    print("\n── Edge: No Seizure Annotations ──")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _create_bids_structure(root, ["s01", "s02"], {})  # No seizures for anyone

        handoff = acquire(str(root))
        _test("handoff created without seizures", len(handoff.subjects) == 2)
        _test("total_seizure_files = 0", handoff.total_seizure_files == 0)


# ════════════════════════════════════════════════════════════
# TEST GROUP 11: Pipeline Runner (unit tests — no core/* needed)
# ════════════════════════════════════════════════════════════

def test_pipeline_runner_split_logic():
    """Test the subject splitting logic from pipeline_runner."""
    print("\n── Pipeline Runner: Split Logic ──")
    from dataloaders.wrappers import _split_subjects

    cfg = {"split": {"train": 0.70, "val": 0.15, "test": 0.15, "seed": 42},
           "stratification": {"enable": False}}

    # 5 subjects
    df = pd.DataFrame([
        {"subject_id": f"s{i:02d}", "age": 20 + i, "sex": "M", "has_seizure": i < 2}
        for i in range(5)
    ])
    train, val, test = _split_subjects(df, cfg)
    all_ids = set(train + val + test)
    _test("5 subj: all assigned", len(all_ids) == 5)
    _test("5 subj: no overlap", len(train) + len(val) + len(test) == 5)
    _test("5 subj: train >= 1", len(train) >= 1)

    # 3 subjects (minimum for 3-way split)
    df3 = pd.DataFrame([
        {"subject_id": f"s{i:02d}", "age": 20, "sex": "M", "has_seizure": False}
        for i in range(3)
    ])
    train3, val3, test3 = _split_subjects(df3, cfg)
    _test("3 subj: all assigned", len(set(train3 + val3 + test3)) == 3)

    # 2 subjects (too few for val/test)
    df2 = pd.DataFrame([
        {"subject_id": "s00", "age": 20, "sex": "M", "has_seizure": False},
        {"subject_id": "s01", "age": 25, "sex": "F", "has_seizure": True},
    ])
    train2, val2, test2 = _split_subjects(df2, cfg)
    _test("2 subj: all in train", len(train2) == 2)
    _test("2 subj: val empty", len(val2) == 0)
    _test("2 subj: test empty", len(test2) == 0)

    # 1 subject
    df1 = pd.DataFrame([{"subject_id": "s00", "age": 20, "sex": "M", "has_seizure": False}])
    train1, val1, test1 = _split_subjects(df1, cfg)
    _test("1 subj: in train", len(train1) == 1)

    # Reproducibility: same seed → same split
    train_a, _, _ = _split_subjects(df, cfg)
    train_b, _, _ = _split_subjects(df, cfg)
    _test("reproducible splits", train_a == train_b)


def test_pipeline_runner_intervals_to_events():
    """Test interval → BIDS event conversion."""
    print("\n── Pipeline Runner: Intervals to Events ──")
    from dataloaders.wrappers import _intervals_to_events

    events = _intervals_to_events([(10.0, 20.0), (50.5, 60.0)], sfreq=256.0)
    _test("2 events created", len(events) == 2)
    _test("event 0 onset", events[0]["onset"] == 10.0)
    _test("event 0 duration", events[0]["duration"] == 10.0)
    _test("event 0 sample", events[0]["sample"] == 2560)
    _test("event 1 onset", events[1]["onset"] == 50.5)
    _test("event 1 trial_type", events[1]["trial_type"] == "seizure")

    # Empty intervals
    events_empty = _intervals_to_events([], sfreq=256.0)
    _test("empty intervals → empty events", events_empty == [])


# ════════════════════════════════════════════════════════════
# TEST GROUP 12: Integration — Handoff → Pipeline (if core/* available)
# ════════════════════════════════════════════════════════════

def test_integration_handoff_to_labels():
    """Test that handoff data can be fed to build_window_index."""
    print("\n── Integration: Handoff → Labels ──")
    try:
        from core.labels import build_window_index, balance_index

        cfg = {
            "windowing": {"window_sec": 1.0, "stride_sec": 1.0,
                          "exclude_negatives_within_sec": 5},
            "labeling": {"mode": "binary", "overlap_threshold": 0.5},
            "balance": {"enable": True, "seizure_ratio": 0.3,
                        "method": "oversample", "seed": 42},
        }

        # Simulate what pipeline_runner does with handoff data
        win_df = build_window_index(
            eeg_path="/fake/path.edf",
            duration_sec=60.0,
            subject_id="test_subj",
            seizure_intervals=[(10.0, 20.0), (40.0, 45.0)],
            cfg=cfg,
            metadata={"age": 25, "sex": "M"},
        )

        _test("window_index created", not win_df.empty)
        _test("has required columns", all(c in win_df.columns for c in ["path", "subject_id", "start_sec", "end_sec", "label"]))
        _test("windows are 1 sec", all((win_df["end_sec"] - win_df["start_sec"]).round(6) == 1.0))
        _test("has seizure labels (label=1)", (win_df["label"] == 1).any())
        _test("has background labels (label=0)", (win_df["label"] == 0).any())

        n_seizure = (win_df["label"] == 1).sum()
        n_background = (win_df["label"] == 0).sum()
        _test(f"seizure windows exist ({n_seizure})", n_seizure > 0)
        _test(f"background windows exist ({n_background})", n_background > 0)

        # Test balancing
        balanced = balance_index(win_df, cfg)
        n_pos_bal = (balanced["label"] == 1).sum()
        ratio = n_pos_bal / len(balanced) if len(balanced) > 0 else 0
        _test(f"balanced ratio near 0.3 (got {ratio:.2f})", 0.2 <= ratio <= 0.4)

        print(f"    (before balance: {len(win_df)} windows, {n_seizure} seizure)")
        print(f"    (after balance: {len(balanced)} windows, {n_pos_bal} seizure, ratio={ratio:.2f})")

    except ImportError:
        print("  ⚠️  Skipped — core.labels not available")


def test_integration_labels_subject_independent():
    """Verify that subject-independent splits work correctly."""
    print("\n── Integration: Subject-Independent Splits ──")
    try:
        from core.labels import build_window_index
        from core.stratify import assign_split_column
        from dataloaders.wrappers import _split_subjects

        cfg = {
            "windowing": {"window_sec": 1.0, "stride_sec": 1.0,
                          "exclude_negatives_within_sec": 5},
            "labeling": {"mode": "binary", "overlap_threshold": 0.5},
            "split": {"train": 0.70, "val": 0.15, "test": 0.15, "seed": 42},
            "stratification": {"enable": False},
        }

        # Build windows for 4 subjects
        all_windows = []
        for i in range(4):
            sid = f"subj{i:02d}"
            win_df = build_window_index(
                eeg_path=f"/fake/{sid}.edf",
                duration_sec=30.0,
                subject_id=sid,
                seizure_intervals=[(5.0, 10.0)] if i < 2 else [],
                cfg=cfg,
                metadata={"age": 20 + i, "sex": "M"},
            )
            all_windows.append(win_df)

        combined = pd.concat(all_windows, ignore_index=True)
        subjects_df = pd.DataFrame([
            {"subject_id": f"subj{i:02d}", "age": 20 + i, "sex": "M", "has_seizure": i < 2}
            for i in range(4)
        ])

        train_ids, val_ids, test_ids = _split_subjects(subjects_df, cfg)
        combined = assign_split_column(combined, train_ids, val_ids, test_ids)

        # CRITICAL: No subject should appear in multiple splits
        for split in ["train", "val", "test"]:
            split_subjs = set(combined[combined["split"] == split]["subject_id"].unique())
            other_splits = combined[combined["split"] != split]
            other_subjs = set(other_splits["subject_id"].unique()) if not other_splits.empty else set()
            overlap = split_subjs & other_subjs
            _test(f"{split}: no subject leakage", len(overlap) == 0,
                  f"leaked subjects: {overlap}" if overlap else "")

        _test("all subjects assigned", combined["split"].notna().all())

    except ImportError:
        print("  ⚠️  Skipped — core modules not available")


# ════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════

def run_all():
    global _passed, _failed, _errors
    _passed = 0
    _failed = 0
    _errors = []

    print("=" * 60)
    print("  WRAPPER & DATA SOURCE TESTS")
    print("=" * 60)

    test_groups = [
        # Group 1: Dataclasses
        test_subject_handoff,
        test_dataset_handoff,
        # Group 2: Registry
        test_registry,
        # Group 3-4: Dataset adapters (offline)
        test_chbmit_adapter,
        test_siena_adapter,
        # Group 5: Local folder
        test_local_folder_nested,
        test_local_folder_flat,
        test_local_folder_json_annotations,
        test_local_folder_csv_annotations,
        # Group 6: BIDS existing
        test_bids_existing,
        test_bids_not_bids,
        # Group 7: Auto-detection
        test_auto_detect,
        # Group 8: Acquisition wrappers
        test_acquisition_detect,
        test_acquisition_full_bids,
        test_acquisition_local_folder,
        # Group 9: Handoff sanity
        test_handoff_sanity,
        # Group 10: Edge cases
        test_edge_empty_dataset,
        test_edge_subject_filter,
        test_edge_mismatched_seizure_starts_ends,
        test_edge_special_characters,
        test_edge_no_seizure_annotations,
        # Group 11: Pipeline runner unit tests
        test_pipeline_runner_split_logic,
        test_pipeline_runner_intervals_to_events,
        # Group 12: Integration (if core/* available)
        test_integration_handoff_to_labels,
        test_integration_labels_subject_independent,
    ]

    for test_fn in test_groups:
        try:
            test_fn()
        except Exception as e:
            _failed += 1
            msg = f"  ❌ {test_fn.__name__} CRASHED: {e}"
            print(msg)
            _errors.append(msg)
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {_passed} passed, {_failed} failed")
    print(f"{'='*60}")

    if _errors:
        print("\nFailed tests:")
        for e in _errors:
            print(e)

    return _failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
