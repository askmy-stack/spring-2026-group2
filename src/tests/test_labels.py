"""
Label Verification Test for CHB-MIT EEG Data Loader
=====================================================
Checks that every window in window_index_{train,val,test}.csv has a label
that is consistent with the BIDS events TSV for the same EDF run.

Ground truth: _events.tsv next to the BIDS EDF (written during BIDS conversion)
  - If events TSV exists  -> file has seizures; windows overlapping seizure intervals = 1
  - If no events TSV      -> file has no seizures; ALL windows must be 0

Run on server:
    python3 src/data_loader/tests/test_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
RESULTS_ROOT = Path("/home/amir/Desktop/GWU/Research/EEG/results")
BIDS_ROOT    = RESULTS_ROOT / "bids_dataset"
DATALOADER   = RESULTS_ROOT / "dataloader"
# Paths stored in the CSV are relative to the data_loader working directory
CSV_BASE_DIR = Path("/home/amir/Desktop/GWU/Research/EEG/src/data_loader")

SPLITS = ["train", "val", "test"]
OVERLAP_THRESHOLD = 0.5   # must match config.yaml labeling.overlap_threshold


# ── Helpers ────────────────────────────────────────────────────────────────

def load_events_tsv(events_path: Path) -> List[Tuple[float, float]]:
    """Return list of (onset, end) seizure intervals from a BIDS events TSV."""
    try:
        df = pd.read_csv(events_path, sep="\t")
    except Exception:
        return []
    if df.empty or "onset" not in df.columns:
        return []
    intervals = []
    for _, row in df.iterrows():
        onset = float(row["onset"])
        dur   = float(row.get("duration", 0.0))
        intervals.append((onset, onset + dur))
    return intervals


def overlap_ratio(win_start: float, win_end: float,
                  intervals: List[Tuple[float, float]]) -> float:
    """Return fraction of the window that overlaps any seizure interval."""
    dur = win_end - win_start
    if dur <= 0:
        return 0.0
    total = 0.0
    for s, e in intervals:
        o = min(win_end, e) - max(win_start, s)
        if o > 0:
            total += o
    return total / dur


def resolve_events_path(path_str: str) -> Path:
    """
    Given the 'path' column value (may be relative to CSV_BASE_DIR),
    return the absolute path of the sibling _events.tsv file.
    Paths in the CSV look like: ../../results/bids_dataset/sub-.../eeg/...edf
    They are relative to src/data_loader (CSV_BASE_DIR).
    """
    p = Path(path_str)
    if not p.is_absolute():
        p = (CSV_BASE_DIR / p).resolve()
    stem = p.stem  # e.g. sub-chb10_task-eeg_run-09_eeg
    return p.parent / (stem + "_events.tsv")


# ── Label verification ─────────────────────────────────────────────────────

def verify_split(split: str) -> dict:
    csv_path = DATALOADER / f"window_index_{split}.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return {}

    df = pd.read_csv(csv_path)
    total  = len(df)
    n_pos  = int((df["label"] == 1).sum())
    n_neg  = int((df["label"] == 0).sum())

    print(f"\n{'='*65}")
    print(f"Split: {split.upper()}  |  total={total:,}  label=1: {n_pos:,}  label=0: {n_neg:,}")
    print(f"{'='*65}")

    # labeled 1 but no events TSV at all (file has no seizure in summary)
    errors_no_events   = []
    # labeled 1 but window does not overlap any seizure
    errors_false_pos   = []
    # labeled 0 but window fully overlaps a seizure
    errors_false_neg   = []

    # Cache: path_str -> list of seizure intervals (empty = no seizures)
    events_cache: Dict[str, List[Tuple[float, float]]] = {}
    exists_cache: Dict[str, bool] = {}

    for _, row in df.iterrows():
        path_str  = row["path"]
        win_start = float(row["start_sec"])
        win_end   = float(row["end_sec"])
        label     = int(row["label"])

        if path_str not in events_cache:
            ev_path = resolve_events_path(path_str)
            exists_cache[path_str] = ev_path.exists()
            events_cache[path_str] = load_events_tsv(ev_path) if exists_cache[path_str] else []

        has_events = exists_cache[path_str]
        intervals  = events_cache[path_str]
        ratio      = overlap_ratio(win_start, win_end, intervals)
        expected   = 1 if ratio >= OVERLAP_THRESHOLD else 0

        if label == 1 and not has_events:
            errors_no_events.append(row.to_dict())
        elif label == 1 and expected == 0:
            errors_false_pos.append({
                "path":             path_str,
                "start_sec":        win_start,
                "end_sec":          win_end,
                "overlap_ratio":    round(ratio, 4),
                "seizure_intervals": intervals,
            })
        elif label == 0 and expected == 1:
            errors_false_neg.append({
                "path":             path_str,
                "start_sec":        win_start,
                "end_sec":          win_end,
                "overlap_ratio":    round(ratio, 4),
                "seizure_intervals": intervals,
            })

    # ── Print results ──
    print(f"\n  [CHECK 1] label=1 but NO events TSV (file has no seizure):")
    if errors_no_events:
        bad = pd.DataFrame(errors_no_events)
        by_path = bad.groupby("path").size().reset_index(name="count")
        print(f"    FAIL — {len(errors_no_events):,} bad windows across {len(by_path)} runs:")
        for _, r in by_path.iterrows():
            print(f"      {r['path'][-80:]}  ({r['count']} windows)")
    else:
        print("    PASS")

    print(f"\n  [CHECK 2] label=1 but overlap ratio < {OVERLAP_THRESHOLD} (false positive):")
    if errors_false_pos:
        print(f"    FAIL — {len(errors_false_pos):,} windows")
        for e in errors_false_pos[:8]:
            print(f"      [{e['start_sec']}-{e['end_sec']}s] ratio={e['overlap_ratio']}  "
                  f"seizures={e['seizure_intervals']}  ...{e['path'][-55:]}")
        if len(errors_false_pos) > 8:
            print(f"      ... and {len(errors_false_pos) - 8} more")
    else:
        print("    PASS")

    print(f"\n  [CHECK 3] label=0 but overlap ratio >= {OVERLAP_THRESHOLD} (false negative):")
    if errors_false_neg:
        print(f"    FAIL — {len(errors_false_neg):,} windows")
        for e in errors_false_neg[:8]:
            print(f"      [{e['start_sec']}-{e['end_sec']}s] ratio={e['overlap_ratio']}  "
                  f"seizures={e['seizure_intervals']}  ...{e['path'][-55:]}")
        if len(errors_false_neg) > 8:
            print(f"      ... and {len(errors_false_neg) - 8} more")
    else:
        print("    PASS")

    total_errors = len(errors_no_events) + len(errors_false_pos) + len(errors_false_neg)
    print(f"\n  Total errors in '{split}': {total_errors:,}")

    return {
        "split":               split,
        "total":               total,
        "n_pos":               n_pos,
        "n_neg":               n_neg,
        "no_events_but_pos":   len(errors_no_events),
        "false_positives":     len(errors_false_pos),
        "false_negatives":     len(errors_false_neg),
    }


# ── Windowing sanity check ─────────────────────────────────────────────────

def windowing_sanity_check(split: str = "train"):
    """
    Per-run checks:
      - All windows have the same size
      - Windows are contiguous (no gaps, no overlaps between consecutive windows)
    """
    csv_path = DATALOADER / f"window_index_{split}.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    df_unique = df.drop_duplicates(subset=["path", "start_sec", "end_sec"])
    n_dupes = total_rows - len(df_unique)

    print(f"\n{'='*65}")
    print(f"Windowing sanity check: {split.upper()}")
    print(f"{'='*65}")
    print(f"\n  Total rows       : {total_rows:,}")
    print(f"  Unique windows   : {len(df_unique):,}")
    print(f"  Duplicate rows   : {n_dupes:,}  (expected from balance_index oversampling in train)")

    sizes = (df_unique["end_sec"] - df_unique["start_sec"]).round(6)
    unique_sizes = sizes.value_counts().sort_index()
    print(f"\n  Window size distribution (expected: all same):")
    for sz, cnt in unique_sizes.items():
        print(f"    {sz}s  ->  {cnt:,} windows")

    overlap_errors = []
    gap_errors     = []

    for path_str, grp in df_unique.groupby("path"):
        grp = grp.sort_values("start_sec").reset_index(drop=True)
        for i in range(1, len(grp)):
            prev_end   = round(float(grp.loc[i - 1, "end_sec"]),   6)
            curr_start = round(float(grp.loc[i,     "start_sec"]), 6)
            diff = round(curr_start - prev_end, 6)
            if diff < -1e-4:
                overlap_errors.append({
                    "path":       path_str,
                    "prev_end":   prev_end,
                    "curr_start": curr_start,
                    "overlap_s":  round(-diff, 6),
                })
            elif diff > 1e-4:
                gap_errors.append({
                    "path":       path_str,
                    "prev_end":   prev_end,
                    "curr_start": curr_start,
                    "gap_s":      round(diff, 6),
                })

    print(f"\n  Overlapping windows (curr_start < prev_end): {len(overlap_errors)}")
    for e in overlap_errors[:5]:
        print(f"    ...{e['path'][-50:]}  prev_end={e['prev_end']}  "
              f"curr_start={e['curr_start']}  overlap={e['overlap_s']}s")

    expected_gaps = [e for e in gap_errors if abs(e["gap_s"] - 300.0) < 1.0]
    unexpected_gaps = [e for e in gap_errors if abs(e["gap_s"] - 300.0) >= 1.0]

    print(f"\n  Gap between consecutive windows (curr_start > prev_end): {len(gap_errors)}")
    print(f"    Expected 300s gaps (exclude_negatives_within_sec=300): {len(expected_gaps)}")
    print(f"    Unexpected gaps (NOT 300s) — real windowing bugs      : {len(unexpected_gaps)}")
    for e in unexpected_gaps[:5]:
        print(f"    ...{e['path'][-50:]}  prev_end={e['prev_end']}  "
              f"curr_start={e['curr_start']}  gap={e['gap_s']}s")

    real_issues = len(overlap_errors) + len(unexpected_gaps)
    if real_issues == 0:
        print("\n  PASS — no real windowing bugs (300s gaps are expected exclusion zones)")
    else:
        print(f"\n  FAIL — {real_issues} real windowing issues found")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("CHB-MIT Label & Windowing Verification")
    print(f"BIDS root : {BIDS_ROOT}")
    print(f"CSV root  : {DATALOADER}")

    summaries = []
    for split in SPLITS:
        result = verify_split(split)
        if result:
            summaries.append(result)

    windowing_sanity_check("train")

    print(f"\n{'='*65}")
    print("OVERALL SUMMARY")
    print(f"{'='*65}")
    total_errors = 0
    for s in summaries:
        errs = s["no_events_but_pos"] + s["false_positives"] + s["false_negatives"]
        total_errors += errs
        status = "PASS" if errs == 0 else "FAIL"
        print(f"  [{status}] {s['split']:5s} | windows={s['total']:>9,} | "
              f"pos={s['n_pos']:>7,} | neg={s['n_neg']:>9,} | errors={errs:,}")

    print()
    if total_errors == 0:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  TOTAL LABEL ERRORS: {total_errors:,}")
        sys.exit(1)