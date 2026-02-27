from __future__ import annotations
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt

from eeg_pipeline.core.yaml_utils import get


@dataclass
class RecordingRow:
    eeg_path: str
    subject_id: str
    duration_sec: float
    sfreq: float
    n_channels: int


class DatasetOverview:
    """
    Non-hardcoded CSV schema:
      - windows are stored as dicts
      - columns are either:
          (A) config whitelist: overview.windows_columns
          (B) union of keys present across all window dicts
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.out_root = Path(get(cfg, "outputs.overview_root", "results/preprocess/overview"))
        self.out_root.mkdir(parents=True, exist_ok=True)
        (self.out_root / "charts").mkdir(exist_ok=True)

        self._recordings_by_path: Dict[str, RecordingRow] = {}
        self.windows: List[Dict[str, Any]] = []  # ✅ no dataclass here

    # ---------- recording level ----------
    def add_recording_from_path(
        self,
        *,
        eeg_path: str,
        subject_id: str,
        sfreq: float,
        n_channels: int,
        duration_sec: float,
    ) -> None:
        if eeg_path in self._recordings_by_path:
            return
        self._recordings_by_path[eeg_path] = RecordingRow(
            eeg_path=str(eeg_path),
            subject_id=str(subject_id),
            duration_sec=float(duration_sec),
            sfreq=float(sfreq),
            n_channels=int(n_channels),
        )

    # ---------- window level ----------
    def add_window(
        self,
        *,
        rid: str,
        eeg_path: str,
        subject_id: str,
        label: int,
        start: float,
        end: float,
        age: Optional[str],
        sex: Optional[str],
        qc: Optional[Dict[str, Any]] = None,
        n_bads: Optional[int] = None,
        eda_artifacts: Optional[Dict[str, Any]] = None,
    ) -> None:
        qc = qc or {}
        eda_artifacts = eda_artifacts or {}

        row: Dict[str, Any] = {
            "rid": str(rid),
            "eeg_path": str(eeg_path),
            "subject_id": str(subject_id),
            "label": int(label),
            "start_sec": float(start),
            "end_sec": float(end),
            "duration_sec": float(max(0.0, end - start)),
        }

        # Add demographics only if present
        if age is not None:
            row["age"] = age
        if sex is not None:
            row["sex"] = sex

        # Add QC only if present
        if qc:
            if "has_nan" in qc:
                row["qc_has_nan"] = qc.get("has_nan")
            if "nan_frac" in qc:
                row["qc_nan_frac"] = qc.get("nan_frac")
            # Noisy fraction can come from different keys depending on QC implementation
            if "noisy_frac" in qc:
                row["qc_noisy_frac"] = qc.get("noisy_frac")
            elif "noisy_channel_frac" in qc:
                row["qc_noisy_frac"] = qc.get("noisy_channel_frac")

        if n_bads is not None:
            row["n_bads"] = int(n_bads)

        # Add artifact pointers ONLY if they exist (no empty columns if you don't add them)
        # Example: keep only qc_json if you want
        if "qc_json" in eda_artifacts and eda_artifacts["qc_json"]:
            row["qc_json"] = eda_artifacts["qc_json"]

        # If later you want psd_csv, add it here conditionally:
        # if "psd_csv" in eda_artifacts and eda_artifacts["psd_csv"]:
        #     row["psd_csv"] = eda_artifacts["psd_csv"]

        self.windows.append(row)

    # ---------- finalize ----------
    def finalize(self) -> Dict[str, str]:
        rec_csv = self.out_root / "recordings.csv"
        win_csv = self.out_root / "windows.csv"
        summary_json = self.out_root / "summary.json"
        charts_dir = self.out_root / "charts"

        # recordings CSV (stable schema from dataclass)
        rec_rows = [asdict(r) for r in self._recordings_by_path.values()]
        self._write_csv(rec_csv, rec_rows)

        # windows CSV (dynamic schema)
        self._write_windows_csv(win_csv, self.windows)

        # summary
        summary = self._compute_summary()
        summary_json.write_text(json.dumps(summary, indent=2))

        # charts
        self._write_charts(charts_dir)

        return {
            "recordings_csv": str(rec_csv),
            "windows_csv": str(win_csv),
            "summary_json": str(summary_json),
            "charts_dir": str(charts_dir),
        }

    def _write_windows_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            path.write_text("")
            return

        # Option A: whitelist columns from config (recommended)
        whitelist = get(self.cfg, "overview.windows_columns", None)
        if isinstance(whitelist, list) and whitelist:
            header = [str(c) for c in whitelist]
        else:
            # Option B: union of keys actually present (no hard-coded / no empty columns)
            keys = set()
            for r in rows:
                keys.update(r.keys())
            # Put common fields first if they exist
            preferred = ["rid", "subject_id", "label", "start_sec", "end_sec", "duration_sec", "eeg_path"]
            header = [c for c in preferred if c in keys] + sorted(k for k in keys if k not in preferred)

        self._write_csv_with_header(path, rows, header)

    def _compute_summary(self) -> Dict[str, Any]:
        recordings = list(self._recordings_by_path.values())
        n_recordings = len(recordings)
        n_subjects = len(set(r.subject_id for r in recordings))
        n_windows = len(self.windows)

        labels = [int(w.get("label", 0)) for w in self.windows]
        y = np.array(labels, dtype=int) if labels else np.array([], dtype=int)

        n_seiz = int(np.sum(y == 1)) if y.size else 0
        n_non = int(np.sum(y == 0)) if y.size else 0

        total_hours = float(sum(r.duration_sec for r in recordings) / 3600.0) if recordings else 0.0

        def _mean_key(label_value: int, key: str) -> float:
            vals = [w.get(key) for w in self.windows if int(w.get("label", -1)) == label_value]
            vals = [v for v in vals if v is not None and np.isfinite(float(v))]
            return float(np.mean([float(v) for v in vals])) if vals else float("nan")

        return {
            "n_subjects": n_subjects,
            "n_recordings": n_recordings,
            "n_windows": n_windows,
            "n_nonseizure_windows": n_non,
            "n_seizure_windows": n_seiz,
            "total_hours_recorded": total_hours,
            "qc_mean_noisy_frac_label0": _mean_key(0, "qc_noisy_frac"),
            "qc_mean_noisy_frac_label1": _mean_key(1, "qc_noisy_frac"),
            "qc_mean_nan_frac_label0": _mean_key(0, "qc_nan_frac"),
            "qc_mean_nan_frac_label1": _mean_key(1, "qc_nan_frac"),
        }

    def _write_charts(self, charts_dir: Path) -> None:
        if not self.windows:
            return

        y = np.array([int(w.get("label", 0)) for w in self.windows], dtype=int)
        labels = ["nonseizure (0)", "seizure (1)"]
        counts = [int(np.sum(y == 0)), int(np.sum(y == 1))]

        fig = plt.figure()
        plt.bar(labels, counts)
        plt.title("Window label counts")
        plt.ylabel("Count")
        plt.tight_layout()
        fig.savefig(charts_dir / "window_label_counts.png", dpi=150)
        plt.close(fig)

        subs = [str(w.get("subject_id", "")) for w in self.windows]
        uniq = sorted(set(subs))
        counts = [subs.count(u) for u in uniq]

        fig = plt.figure()
        plt.bar(uniq, counts)
        plt.title("Windows per subject")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(charts_dir / "windows_per_subject.png", dpi=150)
        plt.close(fig)

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            path.write_text("")
            return
        header = list(rows[0].keys())
        DatasetOverview._write_csv_with_header(path, rows, header)

    @staticmethod
    def _write_csv_with_header(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(["" if r.get(h) is None else r.get(h) for h in header])