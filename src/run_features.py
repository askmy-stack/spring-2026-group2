import argparse
from pathlib import Path

import yaml
import pandas as pd
import mne

from src.feature_engineering import AdvancedFeatureExtractor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to FE config YAML")
    return ap.parse_args()


def main():
    args = parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Paths
    index_path = Path(cfg["io"]["window_index_csv"]).resolve()
    output_csv = Path(cfg["io"]["output_csv"]).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Column names (so we never hardcode)
    col_path = cfg["columns"]["path"]
    col_start = cfg["columns"]["start_sec"]
    col_end = cfg["columns"]["end_sec"]
    col_label = cfg["columns"]["label"]

    # Load window index
    df_index = pd.read_csv(index_path)

    # Feature extractor
    extractor = AdvancedFeatureExtractor(sfreq=cfg["fe"]["sfreq"], cfg=cfg)

    max_rows = cfg["fe"].get("max_rows", None)

    feature_rows = []
    cache = {}  # cache opened FIFs so we don’t re-read for every row

    for i, row in df_index.iterrows():
        if max_rows is not None and i >= int(max_rows):
            break

        rec_path = Path(row[col_path])
        start = float(row[col_start])
        end = float(row[col_end])
        label = int(row[col_label])

        if rec_path not in cache:
            raw = mne.io.read_raw_fif(rec_path, preload=True, verbose=False)
            cache[rec_path] = raw
        else:
            raw = cache[rec_path]

        # Crop window
        epoch = raw.copy().crop(tmin=start, tmax=end)
        data = epoch.get_data()  # shape: (channels, time)

        feats = extractor.extract(data)

        # Add metadata
        feats["label"] = label
        feats["recording_path"] = str(rec_path)
        feats["start_sec"] = start
        feats["end_sec"] = end

        feature_rows.append(feats)

        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{len(df_index)} windows...")

    df_features = pd.DataFrame(feature_rows)
    df_features.to_csv(output_csv, index=False)

    print("✅ Features saved to:", output_csv)
    print("Shape:", df_features.shape)


if __name__ == "__main__":
    main()
