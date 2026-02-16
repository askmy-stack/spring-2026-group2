from __future__ import annotations

from pathlib import Path

from config_io import load_yaml, get
from bids_io import BIDSLoader
from preprocessor import EEGPreprocessor


def main(cfg_path: str = "configs/preprocess.yaml"):
    cfg = load_yaml(cfg_path)

    bids_root = Path(get(cfg, "dataset.bids_root", "results/bids_dataset"))
    datatype = get(cfg, "preprocess.bids.datatype", "eeg")
    suffix = get(cfg, "preprocess.bids.suffix", "eeg")
    extension = get(cfg, "preprocess.bids.extension", None)
    task = get(cfg, "preprocess.bids.task", None)

    loader = BIDSLoader(bids_root=bids_root, datatype=datatype, suffix=suffix)
    if not loader.is_available():
        raise RuntimeError(
            f"BIDS dataset not available at {bids_root}. "
            f"Missing dataset_description.json or mne-bids not installed."
        )

    preproc = EEGPreprocessor(cfg)

    recs = loader.list_recordings(task=task)
    if not recs:
        raise RuntimeError(f"No BIDS recordings found under: {bids_root}")

    for rec in recs:
        rid = loader.recording_id(rec)
        try:
            raw = loader.load_raw(rec, preload=True, extension=extension)
        except Exception as e:
            print(f"[SKIP] {rid} (could not load): {e}")
            continue

        out = preproc.run(raw, rid, save_clean_fif=True)
        print(f"[OK] {rid} -> {out['saved_clean_fif']}")

    print("Done.")


if __name__ == "__main__":
    main()
