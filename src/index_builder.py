# src/index_builder.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import csv
import mne

from yaml_utils import get
from time_domain import TimeDomainModule, Interval


@dataclass
class WindowRow:
    path: str
    start_sec: float
    end_sec: float
    label: int
    recording_id: str
    subject: str
    session: Optional[str]
    task: Optional[str]
    run: Optional[str]
    sfreq: float


class WindowIndexBuilder:
    """
    Producer-side artifact: creates window_index_{split}.csv
    from cleaned FIF + events.tsv seizure intervals.
    Dataloader will later consume ONLY this CSV + the referenced FIF.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.window_sec = float(get(cfg, "analysis.time_domain.epoching.length_sec", 1.0))  # fallback
        # Prefer dataloader-like windowing config if present; else use 1 sec
        self.window_sec = float(get(cfg, "windowing.window_sec", self.window_sec))
        self.stride_sec = float(get(cfg, "windowing.stride_sec", self.window_sec))

        # If you want overlap rule:
        self.overlap_threshold = float(get(cfg, "labeling.overlap_threshold", 0.5))

        self.td = TimeDomainModule(cfg)

    @staticmethod
    def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        lo = max(a0, b0)
        hi = min(a1, b1)
        return max(0.0, hi - lo)

    def _label_window(self, w0: float, w1: float, seizures: List[Interval]) -> int:
        # label=1 if overlap fraction >= threshold
        dur = max(1e-9, w1 - w0)
        for s in seizures:
            ov = self._overlap(w0, w1, s.start, s.end)
            if (ov / dur) >= self.overlap_threshold:
                return 1
        return 0

    def build_rows_for_recording(
        self,
        *,
        cleaned_fif_path: str,
        recording_id: str,
        subject: str,
        session: Optional[str],
        task: Optional[str],
        run: Optional[str],
        events_tsv_path: Optional[str],
    ) -> List[WindowRow]:
        raw = mne.io.read_raw_fif(cleaned_fif_path, preload=False, verbose=False)
        sfreq = float(raw.info["sfreq"])
        dur_sec = float(raw.times[-1]) if raw.n_times else 0.0

        seizures: List[Interval] = []
        if events_tsv_path:
            seizures = self.td.load_seizure_intervals_from_tsv(events_tsv_path)

        rows: List[WindowRow] = []
        t = 0.0
        while t + self.window_sec <= dur_sec:
            w0 = t
            w1 = t + self.window_sec
            lab = self._label_window(w0, w1, seizures)
            rows.append(
                WindowRow(
                    path=cleaned_fif_path,
                    start_sec=w0,
                    end_sec=w1,
                    label=lab,
                    recording_id=recording_id,
                    subject=subject,
                    session=session,
                    task=task,
                    run=run,
                    sfreq=sfreq,
                )
            )
            t += self.stride_sec

        return rows

    @staticmethod
    def write_csv(csv_path: str | Path, rows: List[WindowRow]) -> str:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        header = [
            "path","start_sec","end_sec","label",
            "recording_id","subject","session","task","run","sfreq"
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow([
                    r.path, f"{r.start_sec:.6f}", f"{r.end_sec:.6f}", int(r.label),
                    r.recording_id, r.subject, r.session or "", r.task or "", r.run or "", f"{r.sfreq:.6f}"
                ])
        return str(csv_path)
