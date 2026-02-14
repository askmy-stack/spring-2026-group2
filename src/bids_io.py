from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import mne

try:
    from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
    MNE_BIDS_AVAILABLE = True
except Exception:
    MNE_BIDS_AVAILABLE = False
    BIDSPath = object  # type: ignore


@dataclass(frozen=True)
class BIDSRecording:
    subject: str
    session: Optional[str] = None
    task: Optional[str] = None
    run: Optional[str] = None
    datatype: str = "eeg"
    suffix: str = "eeg"


class BIDSLoader:
    def __init__(self, bids_root: str | Path, datatype: str = "eeg", suffix: str = "eeg"):
        self.bids_root = Path(bids_root)
        self.datatype = datatype
        self.suffix = suffix

    def is_available(self) -> bool:
        return MNE_BIDS_AVAILABLE and (self.bids_root / "dataset_description.json").exists()

    def list_recordings(self, task: Optional[str] = None) -> List[BIDSRecording]:
        if not MNE_BIDS_AVAILABLE:
            return []

        subs = get_entity_vals(self.bids_root, "subject") or []
        sess = get_entity_vals(self.bids_root, "session") or [None]
        tasks = get_entity_vals(self.bids_root, "task") or [None]
        runs = get_entity_vals(self.bids_root, "run") or [None]

        if task is not None:
            tasks = [t for t in tasks if t == task] or [task]

        recs: List[BIDSRecording] = []
        for s in subs:
            for se in sess:
                for t in tasks:
                    for r in runs:
                        recs.append(BIDSRecording(
                            subject=s, session=se, task=t, run=r,
                            datatype=self.datatype, suffix=self.suffix
                        ))
        return recs

    def make_bidspath(self, rec: BIDSRecording, extension: Optional[str] = None) -> "BIDSPath":
        return BIDSPath(
            root=self.bids_root,
            subject=rec.subject,
            session=rec.session,
            task=rec.task,
            run=rec.run,
            datatype=rec.datatype,
            suffix=rec.suffix,
            extension=extension,
        )

    def load_raw(
        self,
        rec: BIDSRecording,
        *,
        preload: bool = True,
        verbose: bool | str = False,
        extension: Optional[str] = None,
    ) -> mne.io.BaseRaw:
        if not MNE_BIDS_AVAILABLE:
            raise ImportError("mne-bids is required. Install with: pip install mne-bids")

        bp = self.make_bidspath(rec, extension=extension)
        raw = read_raw_bids(bp, verbose=verbose)
        if preload and not raw.preload:
            raw.load_data()
        return raw

    @staticmethod
    def recording_id(rec: BIDSRecording) -> str:
        parts = [f"sub-{rec.subject}"]
        if rec.session:
            parts.append(f"ses-{rec.session}")
        if rec.task:
            parts.append(f"task-{rec.task}")
        if rec.run:
            parts.append(f"run-{rec.run}")
        return "_".join(parts)
