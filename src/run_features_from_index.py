import argparse
from pathlib import Path

import yaml
import pandas as pd
import mne

from feature_engineering import AdvancedFeatureExtractor


def pick_col(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    """Pick the first existing column from candidates (helps support multiple index schemas)."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Missing required {kind} column. Tried: {candidates}. "
        f"Available: {list(df.columns)}"
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Extract EEG features from window index CSVs.")
    ap.add_argument("--config", required=True, help="Path to configs/fe.yaml")
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    return ap.parse_args()


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_window_index(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Window index not found: {path}")
    return pd.read_csv(path)


def resolve_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    idx_cfg = cfg.get("window_index", {})
    train_csv = resolve_path(idx_cfg.get("train_csv", "results/dataloader/window_index_train.csv"))
    val_csv = resolve_path(idx_cfg.get("val_csv", "results/dataloader/window_index_val.csv"))
    test_csv = resolve_path(idx_cfg.get("test_csv", "results/dataloader/window_index_test.csv"))

    out_dir = resolve_path(cfg.get("io", {}).get("output_dir", "results/features"))
    out_dir.mkdir(parents=True, exist_ok=True)

    limits_cfg = cfg.get("limits", {})
    max_files = limits_cfg.get("max_files", None)
    max_windows = limits_cfg.get("max_windows", None)

    fe_cfg = cfg.get("fe", {})
    sfreq_mode = str(fe_cfg.get("sfreq_mode", "auto")).lower()  # auto | force
    target_sfreq = float(fe_cfg.get("target_sfreq", 256))

    def run_split(split_name: str, csv_path: Path):
        df_idx = load_window_index(csv_path)

        path_col = pick_col(df_idx, ["eeg_path", "recording_path", "bids_path", "path"], "path")
        start_col = pick_col(df_idx, ["start_sec", "t_start", "window_start", "start"], "start_sec")
        end_col = pick_col(df_idx, ["end_sec", "t_end", "window_end", "end"], "end_sec")
        label_col = pick_col(df_idx, ["label", "y", "target"], "label")

        subj_col = next((c for c in ["subject_id", "subject", "sub_id"] if c in df_idx.columns), None)

        rows = []
        file_count = 0
        total_windows = 0

        # Group by recording path so we load each EDF once
        for rec_path, g in df_idx.groupby(path_col, sort=False):
            file_count += 1
            if max_files is not None and file_count > int(max_files):
                break

            edf_path = Path(rec_path)
            if not edf_path.exists():
                alt = Path.cwd() / edf_path
                if alt.exists():
                    edf_path = alt
                else:
                    print(f"[WARN] Missing EDF path, skipping: {rec_path}")
                    continue

            print(f"[{split_name}] Loading EDF: {edf_path}")
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
            raw.pick_types(eeg=True, meg=False, eog=False, ecg=False, stim=False)

            orig_sfreq = float(raw.info["sfreq"])
            if sfreq_mode == "force" and abs(orig_sfreq - target_sfreq) > 1e-6:
                print(f"[{split_name}] Resampling {orig_sfreq:.2f} Hz -> {target_sfreq:.2f} Hz")
                raw.resample(target_sfreq)
                orig_sfreq = float(raw.info["sfreq"])

            extractor = AdvancedFeatureExtractor(sfreq=orig_sfreq, cfg=cfg)

            for _, r in g.iterrows():
                start = float(r[start_col])
                end = float(r[end_col])
                label = int(r[label_col])

                if end <= start:
                    continue

                # Crop window and extract features
                epoch = raw.copy().crop(tmin=start, tmax=end)
                data = epoch.get_data()  # shape: (n_channels, n_times)
                feats = extractor.extract(data)

                feats.update(
                    {
                        "split": split_name,
                        "label": label,
                        "recording_path": str(edf_path),
                        "start_sec": start,
                        "end_sec": end,
                    }
                )
                if subj_col is not None:
                    feats["subject_id"] = r[subj_col]

                rows.append(feats)
                total_windows += 1

                if max_windows is not None and total_windows >= int(max_windows):
                    break

            if max_windows is not None and total_windows >= int(max_windows):
                break

        out_csv = out_dir / f"features_{split_name}.csv"
        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_csv, index=False)

        print(f"\nSaved: {out_csv}")
        print("Shape:", out_df.shape)
        if "label" in out_df.columns and len(out_df) > 0:
            print("Label counts:\n", out_df["label"].value_counts(dropna=False))

    if args.split in ("train", "all"):
        run_split("train", train_csv)
    if args.split in ("val", "all"):
        run_split("val", val_csv)
    if args.split in ("test", "all"):
        run_split("test", test_csv)


if __name__ == "__main__":
    main()