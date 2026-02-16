from __future__ import annotations

from pathlib import Path
import mne

from config_io import load_yaml, get
from time_domain import TimeDomainProcessor


def main(cfg_path: str = "configs/preprocess.yaml"):
    cfg = load_yaml(cfg_path)

    out_root = Path(get(cfg, "preprocess.out_root", "results/preprocess"))
    cleaned_dir = out_root / "cleaned_raw"

    fif_files = sorted(cleaned_dir.glob("*_raw.fif"))
    if not fif_files:
        raise RuntimeError(f"No preprocessed .fif found in: {cleaned_dir}")

    td = TimeDomainProcessor(cfg)

    for fp in fif_files:
        rid = fp.stem.replace("_filt_raw", "").replace("_nowave_raw", "").replace("_wave_raw", "")
        raw = mne.io.read_raw_fif(fp, preload=True, verbose=False)

        out = td.run(raw, rid)
        print(f"[OK] {rid} -> epochs: {out['saved_epochs_fif']} | stats: {out['epoch_stats_csv']}")

    print("Done.")


if __name__ == "__main__":
    main()
