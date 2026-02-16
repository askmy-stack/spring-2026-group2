from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import mne

try:
    from mne_bids import write_raw_bids, BIDSPath
    MNE_BIDS_AVAILABLE = True
except Exception:
    MNE_BIDS_AVAILABLE = False
    BIDSPath = object  # type: ignore


class BIDSCleanedWriter:
    def __init__(self, bids_out_root: str | Path):
        if not MNE_BIDS_AVAILABLE:
            raise ImportError("mne-bids is required. Install with: pip install mne-bids")
        self.root = Path(bids_out_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_dataset_description()

    def _ensure_dataset_description(self) -> None:
        dd = self.root / "dataset_description.json"
        if dd.exists():
            return
        payload: Dict[str, Any] = {
            "Name": "EEG Preprocessed (Cleaned Raw)",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {"Name": "MNE-BIDS", "Version": "unknown"},
                {"Name": "EEGPreprocessor", "Version": "1.0"}
            ],
        }
        dd.write_text(json.dumps(payload, indent=2))

    def write_cleaned_raw(
        self,
        raw: mne.io.BaseRaw,
        *,
        subject: str,
        session: Optional[str],
        task: Optional[str],
        run: Optional[str],
        datatype: str = "eeg",
        suffix: str = "eeg",
        overwrite: bool = True,
    ) -> str:
        bp = BIDSPath(
            root=self.root,
            subject=subject,
            session=session,
            task=task,
            run=run,
            datatype=datatype,
            suffix=suffix,
            extension=".fif",
        )
        write_raw_bids(raw, bp, overwrite=overwrite, verbose=False)
        return str(bp.fpath)
