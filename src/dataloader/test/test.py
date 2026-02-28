"""
Comprehensive tests for the refactored dataloader/loader/ architecture.

All tests run OFFLINE — no network access, no real EDF files required.
Uses synthetic data, temp directories, and direct function calls.

Test groups:
    1.  BaseDatasetLoader — abstract class contract
    2.  CHBMITLoader — instantiation, config, subject_prefix
    3.  CHB-MIT SUBJECT-INFO parsing (table + key-value formats)
    4.  CHB-MIT summary.txt seizure parsing
    5.  CHBMITLoader integration — metadata + seizures via loader API
    6.  SienaLoader — instantiation, config, subject_prefix
    7.  Siena seizure CSV parsing
    8.  SienaLoader integration — metadata + seizures via loader API
    9.  Windowing — build_windows (labels, capping, seizure preservation)
    10. Windowing — balance_windows (oversampling)
    11. Splits — subject_split (integrity, stratification, edge cases)
    12. Edge cases — empty data, mismatched annotations, malformed files

Run:
    cd src/
    python -m tests.test_loader_pipeline
    # or
    python tests/test_loader_pipeline.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

# ── Ensure src/ is importable ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataloader.loader.base import BaseDatasetLoader
from dataloader.loader.chbmit_loader import (
    CHBMITLoader,
    _parse_subject_info_file,
    _parse_chbmit_summary,
)
from dataloader.loader.siena_loader import (
    SienaLoader,
    _parse_siena_seizure_csv,
    _parse_metadata_file,
    _parse_edf_patient_info,
)
from dataloader.loader.windowing import build_windows, balance_windows
from dataloader.loader.splits import subject_split


# ═══════════════════════════════════════════════════════════
# TEST FRAMEWORK — lightweight, no pytest dependency
# ═══════════════════════════════════════════════════════════

_passed = 0
_failed = 0
_errors: List[str] = []


def _test(name: str, condition: bool, detail: str = ""):
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"   {name}")
    else:
        _failed += 1
        msg = f"   {name}" + (f" — {detail}" if detail else "")
        print(msg)
        _errors.append(msg)


# ═══════════════════════════════════════════════════════════
# SYNTHETIC DATA HELPERS
# ═══════════════════════════════════════════════════════════

def _make_chbmit_subject_info() -> str:
    """Return realistic CHB-MIT SUBJECT-INFO file content."""
    return (
        "Case\tGender\tAge\n"
        "chb01\tF\t11\n"
        "chb02\tM\t11\n"
        "chb03\tF\t14\n"
        "chb04\tM\t22\n"
        "chb05\tF\t7\n"
        "chb06\tF\t1\n"
        "chb07\tF\t14\n"
        "chb08\tM\t3\n"
    )


def _make_chbmit_summary(subject: str = "chb01") -> str:
    """Return realistic CHB-MIT summary.txt content."""
    return (
        f"Data Sampling Rate: 256 Hz\n"
        f"Channel 1: FP1-F7\n"
        f"\n"
        f"File Name: {subject}_01.edf\n"
        f"File Start Time: 11:42:54\n"
        f"File End Time: 12:42:54\n"
        f"Number of Seizures in File: 0\n"
        f"\n"
        f"File Name: {subject}_03.edf\n"
        f"File Start Time: 13:42:54\n"
        f"File End Time: 14:42:54\n"
        f"Number of Seizures in File: 2\n"
        f"Seizure 1 Start Time: 2996 seconds\n"
        f"Seizure 1 End Time: 3036 seconds\n"
        f"Seizure 2 Start Time: 3470 seconds\n"
        f"Seizure 2 End Time: 3522 seconds\n"
        f"\n"
        f"File Name: {subject}_04.edf\n"
        f"File Start Time: 14:42:54\n"
        f"File End Time: 15:42:54\n"
        f"Number of Seizures in File: 1\n"
        f"Seizure Start Time: 1467 seconds\n"
        f"Seizure End Time: 1494 seconds\n"
    )


def _make_siena_seizure_csv(subject: str = "PN00") -> str:
    """Return realistic Siena seizure CSV content."""
    return (
        "Registration Name,Seizure Start (sec.),Seizure End (sec.)\n"
        f"{subject}_1,50,70\n"
        f"{subject}_1,120,135\n"
        f"{subject}_2,200,220\n"
    )


def _make_siena_metadata_csv() -> str:
    """Return Siena subject metadata CSV."""
    return (
        "subject,age,sex\n"
        "PN00,50,F\n"
        "PN01,35,M\n"
        "PN03,62,F\n"
        "PN05,28,M\n"
    )


# ═══════════════════════════════════════════════════════════
# GROUP 1: BaseDatasetLoader — abstract class contract
# ═══════════════════════════════════════════════════════════

def test_base_is_abstract():
    """BaseDatasetLoader cannot be instantiated directly."""
    print("\n── BaseDatasetLoader ──")
    try:
        BaseDatasetLoader(
            name="test", base_url="http://x", raw_cache="/tmp/test_base",
            native_sfreq=256, target_sfreq=256, notch_freq=60,
        )
        _test("cannot instantiate ABC", False, "Should have raised TypeError")
    except TypeError:
        _test("cannot instantiate ABC — TypeError raised", True)

    # Verify abstract methods exist
    import inspect
    abstracts = [
        m for m in dir(BaseDatasetLoader)
        if getattr(getattr(BaseDatasetLoader, m, None), "__isabstractmethod__", False)
    ]
    _test("has 5 abstract methods", len(abstracts) == 5,
          f"found {len(abstracts)}: {abstracts}")

    for method in ["_fetch_records", "_download_subject", "_parse_metadata",
                    "_parse_seizures", "_subject_prefix"]:
        _test(f"  abstract: {method}", method in abstracts)


# ═══════════════════════════════════════════════════════════
# GROUP 2: CHBMITLoader — instantiation, config
# ═══════════════════════════════════════════════════════════

def test_chbmit_loader_init():
    """CHBMITLoader instantiation and config."""
    print("\n── CHBMITLoader Init ──")
    with tempfile.TemporaryDirectory() as tmp:
        loader = CHBMITLoader(raw_cache=tmp)

        _test("name is chbmit", loader.name == "chbmit")
        _test("native_sfreq is 256", loader.native_sfreq == 256.0)
        _test("target_sfreq is 256", loader.target_sfreq == 256)
        _test("notch_freq is 60 (US)", loader.notch_freq == 60.0)
        _test("population is pediatric", loader.population == "pediatric")
        _test("_subject_prefix is chb", loader._subject_prefix() == "chb")

        config = loader.get_config()
        _test("config has name", config["name"] == "chbmit")
        _test("config has base_url", "physionet" in config["base_url"])

        # No downloads yet — empty subjects
        _test("no edf_paths before download", loader.get_edf_paths("chb01") == [])


# ═══════════════════════════════════════════════════════════
# GROUP 3: CHB-MIT SUBJECT-INFO parsing
# ═══════════════════════════════════════════════════════════

def test_chbmit_subject_info_table():
    """Parse tab-separated SUBJECT-INFO table."""
    print("\n── CHB-MIT SUBJECT-INFO (table) ──")
    text = _make_chbmit_subject_info()
    result = _parse_subject_info_file(text)

    _test("parsed 8 subjects", len(result) == 8, f"got {len(result)}")
    _test("chb01 age=11", result["chb01"]["age"] == 11)
    _test("chb01 sex=F", result["chb01"]["sex"] == "F")
    _test("chb02 sex=M", result["chb02"]["sex"] == "M")
    _test("chb04 age=22", result["chb04"]["age"] == 22)
    _test("chb06 age=1", result["chb06"]["age"] == 1)


def test_chbmit_subject_info_keyvalue():
    """Parse key-value format SUBJECT-INFO."""
    print("\n── CHB-MIT SUBJECT-INFO (key-value) ──")
    text = (
        "Subject: chb01\n"
        "Gender: F\n"
        "Age: 11\n"
        "\n"
        "Subject: chb02\n"
        "Gender: M\n"
        "Age: 16\n"
    )
    result = _parse_subject_info_file(text)
    _test("parsed 2 subjects", len(result) == 2)
    _test("chb01 age=11", result["chb01"]["age"] == 11)
    _test("chb02 sex=M", result["chb02"]["sex"] == "M")
    _test("chb02 age=16", result["chb02"]["age"] == 16)


def test_chbmit_subject_info_empty():
    """Empty or garbage input returns empty dict."""
    print("\n── CHB-MIT SUBJECT-INFO (edge) ──")
    _test("empty string → {}", _parse_subject_info_file("") == {})
    _test("whitespace → {}", _parse_subject_info_file("   \n\n  ") == {})
    _test("garbage → {}", _parse_subject_info_file("no useful data here") == {})


# ═══════════════════════════════════════════════════════════
# GROUP 4: CHB-MIT summary.txt seizure parsing
# ═══════════════════════════════════════════════════════════

def test_chbmit_summary_parsing():
    """Parse seizure times from realistic summary.txt."""
    print("\n── CHB-MIT Summary Parsing ──")
    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "chb01-summary.txt"
        summary_path.write_text(_make_chbmit_summary("chb01"))

        result = _parse_chbmit_summary(summary_path)

        _test("3 EDF files parsed", len(result) == 3, f"got {len(result)}")

        # chb01_01.edf — 0 seizures
        _test("chb01_01 has 0 seizures", len(result.get("chb01_01.edf", [])) == 0)

        # chb01_03.edf — 2 seizures
        sz03 = result.get("chb01_03.edf", [])
        _test("chb01_03 has 2 seizures", len(sz03) == 2)
        _test("seizure 1 = (2996, 3036)", sz03[0] == (2996.0, 3036.0))
        _test("seizure 2 = (3470, 3522)", sz03[1] == (3470.0, 3522.0))

        # chb01_04.edf — 1 seizure
        sz04 = result.get("chb01_04.edf", [])
        _test("chb01_04 has 1 seizure", len(sz04) == 1)
        _test("seizure = (1467, 1494)", sz04[0] == (1467.0, 1494.0))


def test_chbmit_summary_missing_file():
    """Non-existent summary returns empty dict."""
    print("\n── CHB-MIT Summary (missing) ──")
    result = _parse_chbmit_summary(Path("/nonexistent/chb99-summary.txt"))
    _test("missing file → {}", result == {})


def test_chbmit_summary_mismatched():
    """Handle mismatched start/end counts (truncate to minimum)."""
    print("\n── CHB-MIT Summary (mismatched) ──")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chb01-summary.txt"
        path.write_text(
            "File Name: chb01_01.edf\n"
            "Seizure 1 Start Time: 100 seconds\n"
            "Seizure 1 End Time: 120 seconds\n"
            "Seizure 2 Start Time: 200 seconds\n"
            # Missing end for seizure 2
        )
        result = _parse_chbmit_summary(path)
        intervals = result.get("chb01_01.edf", [])
        _test("truncated to 1 seizure", len(intervals) == 1)
        _test("kept valid pair (100, 120)", intervals[0] == (100.0, 120.0))


# ═══════════════════════════════════════════════════════════
# GROUP 5: CHBMITLoader integration — metadata + seizures
# ═══════════════════════════════════════════════════════════

def test_chbmit_loader_integration():
    """Full CHBMITLoader with synthetic data on disk."""
    print("\n── CHBMITLoader Integration ──")
    with tempfile.TemporaryDirectory() as tmp:
        loader = CHBMITLoader(raw_cache=tmp)

        # Create fake subject directory
        subj_dir = Path(tmp) / "chb01"
        subj_dir.mkdir()
        (subj_dir / "chb01_01.edf").write_bytes(b"\x00" * 100)
        (subj_dir / "chb01_03.edf").write_bytes(b"\x00" * 100)
        (subj_dir / "chb01_04.edf").write_bytes(b"\x00" * 100)

        # Write summary
        (subj_dir / "chb01-summary.txt").write_text(_make_chbmit_summary("chb01"))

        # Write SUBJECT-INFO at dataset root
        (Path(tmp) / "SUBJECT-INFO").write_text(_make_chbmit_subject_info())

        # list_subjects
        subjects = loader.list_subjects()
        _test("finds chb01", "chb01" in subjects)

        # get_edf_paths
        edfs = loader.get_edf_paths("chb01")
        _test("3 EDFs found", len(edfs) == 3, f"got {len(edfs)}")

        # get_metadata — parsed from SUBJECT-INFO
        meta = loader.get_metadata("chb01")
        _test("metadata age=11", meta["age"] == 11)
        _test("metadata sex=F", meta["sex"] == "F")
        _test("metadata dataset=chbmit", meta["dataset"] == "chbmit")
        _test("metadata subject_id=chb01", meta["subject_id"] == "chb01")

        # get_metadata for unknown subject
        meta_unk = loader.get_metadata("chb99")
        _test("unknown → age=NA", meta_unk["age"] == "NA")

        # get_seizure_intervals — parsed from summary
        sz_01 = loader.get_seizure_intervals(subj_dir / "chb01_01.edf")
        sz_03 = loader.get_seizure_intervals(subj_dir / "chb01_03.edf")
        sz_04 = loader.get_seizure_intervals(subj_dir / "chb01_04.edf")
        _test("chb01_01: 0 seizures", len(sz_01) == 0)
        _test("chb01_03: 2 seizures", len(sz_03) == 2)
        _test("chb01_04: 1 seizure", len(sz_04) == 1)

        # Caching — second call should hit cache
        meta_cached = loader.get_metadata("chb01")
        _test("metadata cache hit", meta_cached["age"] == 11)
        sz_cached = loader.get_seizure_intervals(subj_dir / "chb01_03.edf")
        _test("seizure cache hit", len(sz_cached) == 2)


# ═══════════════════════════════════════════════════════════
# GROUP 6: SienaLoader — instantiation, config
# ═══════════════════════════════════════════════════════════

def test_siena_loader_init():
    """SienaLoader instantiation and config."""
    print("\n── SienaLoader Init ──")
    with tempfile.TemporaryDirectory() as tmp:
        loader = SienaLoader(raw_cache=tmp)

        _test("name is siena", loader.name == "siena")
        _test("native_sfreq is 512", loader.native_sfreq == 512.0)
        _test("target_sfreq is 256", loader.target_sfreq == 256)
        _test("notch_freq is 50 (EU)", loader.notch_freq == 50.0)
        _test("population is adult", loader.population == "adult")
        _test("_subject_prefix is PN", loader._subject_prefix() == "PN")


# ═══════════════════════════════════════════════════════════
# GROUP 7: Siena seizure CSV parsing
# ═══════════════════════════════════════════════════════════

def test_siena_seizure_csv():
    """Parse Siena seizure CSV with Registration Name column."""
    print("\n── Siena Seizure CSV ──")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "Seizures-list-PN00.csv"
        csv_path.write_text(_make_siena_seizure_csv("PN00"))

        result = _parse_siena_seizure_csv(csv_path)

        # PN00_1.edf — 2 seizures
        sz1 = result.get("PN00_1.edf", [])
        _test("PN00_1 has 2 seizures", len(sz1) == 2, f"got {len(sz1)}")
        _test("first = (50, 70)", sz1[0] == (50.0, 70.0))
        _test("second = (120, 135)", sz1[1] == (120.0, 135.0))

        # PN00_2.edf — 1 seizure
        sz2 = result.get("PN00_2.edf", [])
        _test("PN00_2 has 1 seizure", len(sz2) == 1)
        _test("(200, 220)", sz2[0] == (200.0, 220.0))


def test_siena_seizure_csv_missing():
    """Missing CSV returns empty dict."""
    print("\n── Siena Seizure CSV (missing) ──")
    result = _parse_siena_seizure_csv(Path("/nonexistent.csv"))
    _test("missing → {}", result == {})


def test_siena_seizure_csv_alternate_columns():
    """Parse CSV with different column names."""
    print("\n── Siena Seizure CSV (alternate cols) ──")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test.csv"
        csv_path.write_text(
            "File Name,Start Time,End Time\n"
            "rec_01,10,25\n"
            "rec_02,40,55\n"
        )
        result = _parse_siena_seizure_csv(csv_path)
        _test("rec_01 parsed", len(result.get("rec_01.edf", [])) == 1)
        _test("rec_02 parsed", len(result.get("rec_02.edf", [])) == 1)
        _test("correct values", result["rec_01.edf"][0] == (10.0, 25.0))


def test_siena_metadata_csv_parsing():
    """Parse Siena metadata from CSV file."""
    print("\n── Siena Metadata CSV ──")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "subject-info.csv"
        csv_path.write_text(_make_siena_metadata_csv())

        result = _parse_metadata_file(csv_path)
        _test("parsed 4 subjects", len(result) == 4, f"got {len(result)}")
        _test("PN00 age=50", result["PN00"]["age"] == 50)
        _test("PN00 sex=F", result["PN00"]["sex"] == "F")
        _test("PN01 sex=M", result["PN01"]["sex"] == "M")
        _test("PN05 age=28", result["PN05"]["age"] == 28)


# ═══════════════════════════════════════════════════════════
# GROUP 8: SienaLoader integration
# ═══════════════════════════════════════════════════════════

def test_siena_loader_integration():
    """Full SienaLoader with synthetic data on disk."""
    print("\n── SienaLoader Integration ──")
    with tempfile.TemporaryDirectory() as tmp:
        loader = SienaLoader(raw_cache=tmp)

        # Create fake subject
        subj_dir = Path(tmp) / "PN00"
        subj_dir.mkdir()
        (subj_dir / "PN00_1.edf").write_bytes(b"\x00" * 100)
        (subj_dir / "PN00_2.edf").write_bytes(b"\x00" * 100)

        # Seizure CSV
        (subj_dir / "Seizures-list-PN00.csv").write_text(_make_siena_seizure_csv("PN00"))

        # Metadata at root
        (Path(tmp) / "subject-info.csv").write_text(_make_siena_metadata_csv())

        # list_subjects
        subjects = loader.list_subjects()
        _test("finds PN00", "PN00" in subjects)

        # get_edf_paths
        edfs = loader.get_edf_paths("PN00")
        _test("2 EDFs found", len(edfs) == 2)

        # get_metadata
        meta = loader.get_metadata("PN00")
        _test("metadata age=50", meta["age"] == 50)
        _test("metadata sex=F", meta["sex"] == "F")
        _test("metadata dataset=siena", meta["dataset"] == "siena")

        # get_seizure_intervals
        sz1 = loader.get_seizure_intervals(subj_dir / "PN00_1.edf")
        sz2 = loader.get_seizure_intervals(subj_dir / "PN00_2.edf")
        _test("PN00_1: 2 seizures", len(sz1) == 2)
        _test("PN00_2: 1 seizure", len(sz2) == 1)


# ═══════════════════════════════════════════════════════════
# GROUP 9: Windowing — build_windows
# ═══════════════════════════════════════════════════════════

def test_build_windows_basic():
    """Build windows from a 60-second recording with 1 seizure."""
    print("\n── build_windows (basic) ──")
    fake_path = Path("/fake/sub-test/ses-001/eeg/test.edf")

    df = build_windows(
        edf_path=fake_path,
        seizure_intervals=[(10.0, 15.0)],
        subject_id="test01",
        metadata={"age": 25, "sex": "M"},
        duration_sec=60.0,
        window_sec=1.0,
        stride_sec=1.0,
        exclude_near_seizure_sec=0.0,   # disable exclusion for basic test
        max_background_per_file=0,       # no cap
    )

    _test("non-empty DataFrame", not df.empty)
    _test("has required columns", all(
        c in df.columns for c in ["path", "subject_id", "start_sec", "end_sec", "label", "age", "sex"]
    ))

    n_seizure = (df["label"] == 1).sum()
    n_background = (df["label"] == 0).sum()
    _test("has seizure windows", n_seizure > 0, f"got {n_seizure}")
    _test("has background windows", n_background > 0)
    _test("total = 60 windows (1s stride, 60s)", len(df) == 60, f"got {len(df)}")

    # Seizure windows should be exactly within 10-15s
    seizure_starts = df[df["label"] == 1]["start_sec"].tolist()
    _test("seizure windows in [10, 14]", all(10 <= s <= 14 for s in seizure_starts),
          f"got {seizure_starts}")
    _test("5 seizure windows (10,11,12,13,14)", n_seizure == 5, f"got {n_seizure}")

    _test("subject_id correct", (df["subject_id"] == "test01").all())
    _test("age propagated", (df["age"] == 25).all())


def test_build_windows_background_cap():
    """Background capping limits background windows, preserves ALL seizure windows."""
    print("\n── build_windows (background cap) ──")
    fake_path = Path("/fake/test.edf")

    df = build_windows(
        edf_path=fake_path,
        seizure_intervals=[(10.0, 15.0)],
        subject_id="test01",
        metadata={"age": 25, "sex": "M"},
        duration_sec=1000.0,           # large recording
        window_sec=1.0,
        stride_sec=1.0,
        exclude_near_seizure_sec=0.0,  # disable exclusion
        max_background_per_file=50,    # cap at 50
    )

    n_seizure = (df["label"] == 1).sum()
    n_background = (df["label"] == 0).sum()
    _test("seizure windows preserved (5)", n_seizure == 5)
    _test("background capped at 50", n_background == 50, f"got {n_background}")
    _test("total = 55", len(df) == 55)


def test_build_windows_no_seizures():
    """Recording with no seizures — all background."""
    print("\n── build_windows (no seizures) ──")
    df = build_windows(
        edf_path=Path("/fake/test.edf"),
        seizure_intervals=[],
        subject_id="test01",
        metadata={"age": 30, "sex": "F"},
        duration_sec=30.0,
        window_sec=1.0,
        stride_sec=1.0,
        exclude_near_seizure_sec=0.0,
        max_background_per_file=0,
    )
    _test("30 windows", len(df) == 30)
    _test("all background", (df["label"] == 0).all())


def test_build_windows_exclusion_zone():
    """Windows near seizures are excluded when exclude_near_seizure_sec > 0."""
    print("\n── build_windows (exclusion zone) ──")
    df = build_windows(
        edf_path=Path("/fake/test.edf"),
        seizure_intervals=[(50.0, 55.0)],
        subject_id="test01",
        metadata={"age": 20, "sex": "M"},
        duration_sec=100.0,
        window_sec=1.0,
        stride_sec=1.0,
        exclude_near_seizure_sec=10.0,
        max_background_per_file=0,
    )

    n_seizure = (df["label"] == 1).sum()
    n_background = (df["label"] == 0).sum()
    _test("seizure windows exist", n_seizure > 0)
    _test("fewer background (exclusion zone)", n_background < 80,
          f"got {n_background}, expected < 80 due to exclusion")

    # No background windows in [40, 65] range (except actual seizures)
    bg = df[df["label"] == 0]
    bg_near = bg[(bg["start_sec"] >= 40) & (bg["start_sec"] <= 64)]
    _test("no background in exclusion zone", len(bg_near) == 0,
          f"found {len(bg_near)} in exclusion zone")


def test_build_windows_short_recording():
    """Recording shorter than one window."""
    print("\n── build_windows (short recording) ──")
    df = build_windows(
        edf_path=Path("/fake/test.edf"),
        seizure_intervals=[],
        subject_id="test01",
        metadata={"age": 20, "sex": "M"},
        duration_sec=0.5,   # half a second — less than 1s window
        window_sec=1.0,
        stride_sec=1.0,
        max_background_per_file=0,
    )
    _test("no windows from 0.5s recording", len(df) == 0)


# ═══════════════════════════════════════════════════════════
# GROUP 10: balance_windows
# ═══════════════════════════════════════════════════════════

def test_balance_windows_oversample():
    """Oversampling seizure windows to target ratio."""
    print("\n── balance_windows ──")
    # 100 background, 5 seizure → 5% ratio, target 30%
    rows = (
        [{"label": 0, "subject_id": "s1", "path": "/fake", "start_sec": i, "end_sec": i + 1}
         for i in range(100)]
        + [{"label": 1, "subject_id": "s1", "path": "/fake", "start_sec": i, "end_sec": i + 1}
           for i in range(100, 105)]
    )
    df = pd.DataFrame(rows)

    balanced = balance_windows(df, seizure_ratio=0.3, seed=42)
    n = len(balanced)
    n_sz = (balanced["label"] == 1).sum()
    ratio = n_sz / n if n > 0 else 0

    _test("more windows after balance", n > len(df))
    _test(f"ratio near 0.3 (got {ratio:.2f})", 0.25 <= ratio <= 0.35)
    _test("original background preserved", (balanced["label"] == 0).sum() == 100)


def test_balance_windows_already_balanced():
    """No oversampling needed if already at target."""
    print("\n── balance_windows (already balanced) ──")
    rows = (
        [{"label": 0, "subject_id": "s1"} for _ in range(70)]
        + [{"label": 1, "subject_id": "s1"} for _ in range(30)]
    )
    df = pd.DataFrame(rows)
    balanced = balance_windows(df, seizure_ratio=0.3)
    _test("same size (already ≥ 30%)", len(balanced) == 100)


def test_balance_windows_empty():
    """Empty DataFrame returns empty."""
    print("\n── balance_windows (empty) ──")
    df = pd.DataFrame(columns=["label", "subject_id"])
    balanced = balance_windows(df)
    _test("empty in → empty out", balanced.empty)


def test_balance_windows_all_seizure():
    """All seizure windows — no oversampling needed."""
    print("\n── balance_windows (all seizure) ──")
    rows = [{"label": 1, "subject_id": "s1"} for _ in range(20)]
    df = pd.DataFrame(rows)
    balanced = balance_windows(df, seizure_ratio=0.3)
    _test("unchanged", len(balanced) == 20)


# ═══════════════════════════════════════════════════════════
# GROUP 11: subject_split
# ═══════════════════════════════════════════════════════════

def test_subject_split_basic():
    """5 subjects: 70/15/15 split with no subject leakage."""
    print("\n── subject_split (5 subjects) ──")
    rows = []
    for i in range(5):
        sid = f"subj{i:02d}"
        for w in range(20):
            rows.append({
                "subject_id": sid,
                "label": 1 if (w < 3 and i < 3) else 0,
                "path": f"/fake/{sid}.edf",
                "start_sec": w,
                "end_sec": w + 1,
            })
    df = pd.DataFrame(rows)

    train, val, test = subject_split(df, seed=42)

    all_train_subjs = set(train["subject_id"].unique())
    all_val_subjs = set(val["subject_id"].unique())
    all_test_subjs = set(test["subject_id"].unique())

    _test("train non-empty", len(train) > 0)
    _test("val non-empty", len(val) > 0)
    _test("test non-empty", len(test) > 0)

    # CRITICAL: no subject leakage
    _test("no train-val overlap", len(all_train_subjs & all_val_subjs) == 0)
    _test("no train-test overlap", len(all_train_subjs & all_test_subjs) == 0)
    _test("no val-test overlap", len(all_val_subjs & all_test_subjs) == 0)

    # All subjects accounted for
    total_subjs = all_train_subjs | all_val_subjs | all_test_subjs
    _test("all 5 subjects assigned", len(total_subjs) == 5)

    # All windows accounted for
    _test("all windows preserved", len(train) + len(val) + len(test) == len(df))


def test_subject_split_reproducible():
    """Same seed → same split."""
    print("\n── subject_split (reproducible) ──")
    rows = [{"subject_id": f"s{i}", "label": 0} for i in range(10) for _ in range(5)]
    df = pd.DataFrame(rows)

    train_a, val_a, test_a = subject_split(df, seed=42)
    train_b, val_b, test_b = subject_split(df, seed=42)

    _test("train same", set(train_a["subject_id"]) == set(train_b["subject_id"]))
    _test("val same", set(val_a["subject_id"]) == set(val_b["subject_id"]))
    _test("test same", set(test_a["subject_id"]) == set(test_b["subject_id"]))


def test_subject_split_two_subjects():
    """2 subjects: all go to train (can't make 3-way split)."""
    print("\n── subject_split (2 subjects) ──")
    rows = [{"subject_id": f"s{i}", "label": 0} for i in range(2) for _ in range(10)]
    df = pd.DataFrame(rows)

    train, val, test = subject_split(df, seed=42)
    _test("all in train", len(train) == 20)
    _test("val empty", len(val) == 0)
    _test("test empty", len(test) == 0)


def test_subject_split_one_subject():
    """1 subject: all in train."""
    print("\n── subject_split (1 subject) ──")
    rows = [{"subject_id": "s0", "label": 0} for _ in range(10)]
    df = pd.DataFrame(rows)

    train, val, test = subject_split(df, seed=42)
    _test("all in train", len(train) == 10)


def test_subject_split_empty():
    """Empty DataFrame."""
    print("\n── subject_split (empty) ──")
    df = pd.DataFrame(columns=["subject_id", "label"])
    train, val, test = subject_split(df, seed=42)
    _test("all empty", len(train) == 0 and len(val) == 0 and len(test) == 0)


# ═══════════════════════════════════════════════════════════
# GROUP 12: Edge cases
# ═══════════════════════════════════════════════════════════

def test_edge_chbmit_no_subject_info_file():
    """CHBMITLoader handles missing SUBJECT-INFO gracefully."""
    print("\n── Edge: Missing SUBJECT-INFO ──")
    with tempfile.TemporaryDirectory() as tmp:
        loader = CHBMITLoader(raw_cache=tmp)

        # Create subject dir without SUBJECT-INFO
        subj_dir = Path(tmp) / "chb01"
        subj_dir.mkdir()
        (subj_dir / "chb01_01.edf").write_bytes(b"\x00" * 100)

        meta = loader.get_metadata("chb01")
        _test("falls back to age=NA", meta["age"] == "NA")
        _test("falls back to sex=NA", meta["sex"] == "NA")
        _test("still has subject_id", meta["subject_id"] == "chb01")


def test_edge_siena_no_metadata():
    """SienaLoader handles missing metadata gracefully."""
    print("\n── Edge: Missing Siena Metadata ──")
    with tempfile.TemporaryDirectory() as tmp:
        loader = SienaLoader(raw_cache=tmp)

        subj_dir = Path(tmp) / "PN00"
        subj_dir.mkdir()
        (subj_dir / "PN00_1.edf").write_bytes(b"\x00" * 100)

        meta = loader.get_metadata("PN00")
        _test("age is NA", meta["age"] == "NA")
        _test("sex is NA", meta["sex"] == "NA")


def test_edge_siena_no_seizure_csv():
    """SienaLoader handles missing seizure CSV."""
    print("\n── Edge: Missing Seizure CSV ──")
    with tempfile.TemporaryDirectory() as tmp:
        loader = SienaLoader(raw_cache=tmp)

        subj_dir = Path(tmp) / "PN00"
        subj_dir.mkdir()
        (subj_dir / "PN00_1.edf").write_bytes(b"\x00" * 100)

        intervals = loader.get_seizure_intervals(subj_dir / "PN00_1.edf")
        _test("empty intervals", len(intervals) == 0)


def test_edge_multiple_subjects_split_integrity():
    """10 subjects split — exhaustive no-leakage check."""
    print("\n── Edge: 10-Subject Split Integrity ──")
    rows = []
    for i in range(10):
        sid = f"subj{i:02d}"
        for w in range(15):
            label = 1 if (i < 5 and w < 2) else 0
            rows.append({"subject_id": sid, "label": label, "path": f"/fake/{sid}.edf",
                         "start_sec": w, "end_sec": w + 1})
    df = pd.DataFrame(rows)

    train, val, test = subject_split(df, seed=99)
    train_s = set(train["subject_id"].unique())
    val_s = set(val["subject_id"].unique())
    test_s = set(test["subject_id"].unique())

    _test("no leakage (train∩val)", len(train_s & val_s) == 0)
    _test("no leakage (train∩test)", len(train_s & test_s) == 0)
    _test("no leakage (val∩test)", len(val_s & test_s) == 0)
    _test("all subjects assigned", len(train_s | val_s | test_s) == 10)
    _test("all windows preserved", len(train) + len(val) + len(test) == 150)


def test_edge_build_windows_multiple_seizures():
    """Recording with 3 non-overlapping seizures."""
    print("\n── Edge: Multiple Seizures ──")
    df = build_windows(
        edf_path=Path("/fake/test.edf"),
        seizure_intervals=[(10.0, 15.0), (30.0, 35.0), (50.0, 55.0)],
        subject_id="test01",
        metadata={"age": 20, "sex": "M"},
        duration_sec=100.0,
        window_sec=1.0,
        stride_sec=1.0,
        exclude_near_seizure_sec=0.0,
        max_background_per_file=0,
    )
    n_sz = (df["label"] == 1).sum()
    _test("15 seizure windows (5+5+5)", n_sz == 15, f"got {n_sz}")


def test_edge_siena_bad_csv_values():
    """Siena CSV with invalid values (non-numeric, end < start)."""
    print("\n── Edge: Bad CSV Values ──")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "test.csv"
        csv_path.write_text(
            "Registration Name,Seizure Start (sec.),Seizure End (sec.)\n"
            "rec_01,abc,def\n"          # non-numeric → skip
            "rec_02,100,50\n"           # end < start → skip
            "rec_03,10,20\n"            # valid
        )
        result = _parse_siena_seizure_csv(csv_path)
        _test("only valid row kept", len(result) == 1)
        _test("rec_03 parsed", "rec_03.edf" in result)


def test_edge_loader_get_config():
    """Verify get_config returns correct structure."""
    print("\n── Edge: get_config ──")
    with tempfile.TemporaryDirectory() as tmp:
        chb = CHBMITLoader(raw_cache=tmp)
        cfg = chb.get_config()
        for key in ["name", "base_url", "native_sfreq", "target_sfreq", "notch_freq", "raw_cache"]:
            _test(f"config has {key}", key in cfg)


# ═══════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════

def run_all():
    global _passed, _failed, _errors
    _passed = 0
    _failed = 0
    _errors = []

    print("=" * 60)
    print("  DATALOADER/LOADER PIPELINE TESTS")
    print("=" * 60)

    test_groups = [
        # Group 1: Base class
        test_base_is_abstract,
        # Group 2: CHBMITLoader init
        test_chbmit_loader_init,
        # Group 3: SUBJECT-INFO parsing
        test_chbmit_subject_info_table,
        test_chbmit_subject_info_keyvalue,
        test_chbmit_subject_info_empty,
        # Group 4: Summary parsing
        test_chbmit_summary_parsing,
        test_chbmit_summary_missing_file,
        test_chbmit_summary_mismatched,
        # Group 5: CHBMITLoader integration
        test_chbmit_loader_integration,
        # Group 6: SienaLoader init
        test_siena_loader_init,
        # Group 7: Siena CSV parsing
        test_siena_seizure_csv,
        test_siena_seizure_csv_missing,
        test_siena_seizure_csv_alternate_columns,
        test_siena_metadata_csv_parsing,
        # Group 8: SienaLoader integration
        test_siena_loader_integration,
        # Group 9: Windowing
        test_build_windows_basic,
        test_build_windows_background_cap,
        test_build_windows_no_seizures,
        test_build_windows_exclusion_zone,
        test_build_windows_short_recording,
        # Group 10: Balance
        test_balance_windows_oversample,
        test_balance_windows_already_balanced,
        test_balance_windows_empty,
        test_balance_windows_all_seizure,
        # Group 11: Splits
        test_subject_split_basic,
        test_subject_split_reproducible,
        test_subject_split_two_subjects,
        test_subject_split_one_subject,
        test_subject_split_empty,
        # Group 12: Edge cases
        test_edge_chbmit_no_subject_info_file,
        test_edge_siena_no_metadata,
        test_edge_siena_no_seizure_csv,
        test_edge_multiple_subjects_split_integrity,
        test_edge_build_windows_multiple_seizures,
        test_edge_siena_bad_csv_values,
        test_edge_loader_get_config,
    ]

    for test_fn in test_groups:
        try:
            test_fn()
        except Exception as e:
            _failed += 1
            msg = f"  {test_fn.__name__} CRASHED: {e}"
            print(msg)
            _errors.append(msg)
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {_passed} passed, {_failed} failed")
    print(f"{'=' * 60}")

    if _errors:
        print("\nFailed tests:")
        for e in _errors:
            print(e)

    return _failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)