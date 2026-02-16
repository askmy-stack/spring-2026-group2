from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import mne


class BIDSCleanedWriter:
    """
    Writes cleaned data as FIF into a BIDS-derivatives style folder:
      {root}/sub-XX[/ses-YY]/{datatype}/sub-XX[_ses-YY][_task-ZZ][_run-AA]_{suffix}.fif

    This is a derivative output (not raw BIDS EEG), so we do NOT use write_raw_bids().
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
            "Name": "EEG Preprocessed (Cleaned Raw)",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {"Name": "EEGPreprocessor", "Version": "1.0"},
                {"Name": "MNE", "Version": mne.__version__},
            ],
        }
        dd.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _build_bids_basename(
        *,
        subject: str,
        session: Optional[str],
        task: Optional[str],
        run: Optional[str],
        suffix: str,
    ) -> str:
        parts = [f"sub-{subject}"]
        if session:
            parts.append(f"ses-{session}")
        if task:
            parts.append(f"task-{task}")
        if run:
            parts.append(f"run-{run}")
        parts.append(suffix)
        return "_".join(parts)

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
        # Optional: copy these BIDS sidecars from input raw BIDS
        copy_sidecars_from: Optional[Path] = None,  # directory containing input sidecars
        input_basename: Optional[str] = None,       # e.g., sub-001_task-xxx_eeg
    ) -> str:
        # Derivative folder structure
        sub_dir = self.root / f"sub-{subject}"
        if session:
            sub_dir = sub_dir / f"ses-{session}"
        out_dir = sub_dir / datatype
        out_dir.mkdir(parents=True, exist_ok=True)

        base = self._build_bids_basename(
            subject=subject, session=session, task=task, run=run, suffix=suffix
        )
        fif_path = out_dir / f"{base}.fif"

        raw.save(fif_path, overwrite=overwrite, verbose=False)

        # Optionally copy sidecars so you keep channels/events json/tsv near the fif
        # (This is optional but helpful for traceability.)
        if copy_sidecars_from is not None and input_basename is not None:
            for ext in (".json", "_channels.tsv", "_events.tsv"):
                # input files are usually like: {input_basename}_eeg.json, etc.
                # so we copy any matching file names if they exist.
                # Example: sub-001_task-xxx_eeg.json
                cand = copy_sidecars_from / f"{input_basename}{ext}"
                if cand.exists():
                    shutil.copy2(cand, out_dir / cand.name)

        return str(fif_path)
