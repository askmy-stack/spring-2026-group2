from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import mne

try:
    from mne_bids import BIDSPath, write_raw_bids
except Exception as e:
    raise ImportError("mne-bids is required. Install with: pip install mne-bids") from e


class BIDSCleanedWriter:
    """
    Writes cleaned EEG as BIDS *derivatives* in EDF format (NO FIF).
    """

    def __init__(self, bids_out_root: str | Path):
        self.root = Path(bids_out_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_dataset_description()

    def _ensure_dataset_description(self) -> None:
        dd = self.root / "dataset_description.json"
        if dd.exists():
            return
        payload: Dict[str, Any] = {
            "Name": "EEG Preprocessed (Derivatives, EDF)",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "EEGPipeline", "Version": "1.0"}],
        }
        dd.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _safe_meas_date(raw: mne.io.BaseRaw) -> None:
        # robust: drop invalid meas_date to avoid writer errors
        try:
            md = raw.info.get("meas_date", None)
            import datetime as _dt
            if isinstance(md, _dt.datetime):
                if md.tzinfo is None:
                    md = md.replace(tzinfo=_dt.timezone.utc)
                md = (int(md.timestamp()), int(md.microsecond))
            if isinstance(md, tuple) and len(md) >= 1:
                sec = int(md[0])
                if sec > 2147483647 or sec < -2147483648:
                    raw.set_meas_date(None)
        except Exception:
            raw.set_meas_date(None)

    def write_cleaned_raw(
        self,
        raw: mne.io.BaseRaw,
        *,
        subject: str,
        session: Optional[str] = None,
        task: Optional[str] = None,
        run: Optional[str] = None,
        datatype: str = "eeg",
        suffix: str = "eeg",
        overwrite: bool = True,
    ) -> str:
        self._safe_meas_date(raw)
        subject = str(subject).strip()
        subject = subject.replace("sub-", "").replace("-", "").replace("_", "")

        if run is not None:
            run = str(run).strip()
            if not run.isdigit():
                run = "01"

        if task is not None:
            task = str(task).strip().replace("-", "").replace("_", "")
        if session is not None:
            session = str(session).strip().replace("-", "").replace("_", "")
        bids_path = BIDSPath(
            root=self.root,
            subject=subject,
            session=session,
            task=task,
            run=run,
            datatype=datatype,
            suffix=suffix,
            extension=".edf",
        )

        write_raw_bids(
            raw,
            bids_path=bids_path,
            overwrite=overwrite,
            format="EDF",
            allow_preload=True,
            verbose=False,
        )

        return str(bids_path.fpath)