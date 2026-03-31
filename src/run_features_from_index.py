#!/usr/bin/env python3
"""
Feature extraction from window index CSVs.
- Caches features per unique (recording, start, end) to avoid redundant extraction
- Does NOT deduplicate during merge (preserves oversampled seizure windows)
- Checkpoint/resume support
"""

import os, sys, gc, csv, time, argparse
import numpy as np
import pandas as pd
import yaml, mne
from pathlib import Path
from joblib import Parallel, delayed
from feature_engineering import AdvancedFeatureExtractor

mne.set_log_level("ERROR")

# ------------------------------------------------------------------ #
#  helpers
# ------------------------------------------------------------------ #
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def get_sfreq_from_edf(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    sfreq = raw.info["sfreq"]
    del raw
    return sfreq

def process_one_file(edf_path, windows_df, extractor, sfreq):
    """Extract features for all windows in one EDF file.
    Uses a cache so duplicate windows (from oversampling) are extracted only once.
    Returns list of dicts (one per row in windows_df, preserving duplicates).
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"  [SKIP] Cannot read {edf_path}: {e}")
        return []

    data = raw.get_data()
    max_time = data.shape[1] / sfreq
    n_channels = data.shape[0]
    results = []
    cache = {}  # (start, end) -> feature_dict

    for _, row in windows_df.iterrows():
        start = float(row["start_sec"])
        end = float(row["end_sec"])

        if end > max_time:
            continue

        key = (start, end)
        if key not in cache:
            s_idx = int(round(start * sfreq))
            e_idx = int(round(end * sfreq))
            if e_idx > data.shape[1]:
                continue
            segment = data[:, s_idx:e_idx]
            if segment.shape[1] == 0:
                continue

            try:
                feats = extractor.extract(segment)
            except Exception as e:
                print(f"  [SKIP] Feature error {edf_path} [{start}-{end}]: {e}")
                continue
            cache[key] = feats

        if key in cache:
            feat_row = dict(cache[key])
            feat_row["path"] = str(row.get("path", edf_path))
            feat_row["start_sec"] = start
            feat_row["end_sec"] = end
            feat_row["label"] = int(row["label"])
            results.append(feat_row)

    del data, raw
    gc.collect()
    print(f"  [DONE] {Path(edf_path).name}: {len(results)} windows "
          f"({len(cache)} unique, {len(results)-len(cache)} cached)")
    return results

# ------------------------------------------------------------------ #
#  merge chunks -- NO deduplication
# ------------------------------------------------------------------ #
def merge_chunks_no_dedup(chunk_dir, output_path):
    """Stream-merge all chunk CSVs into one file. No deduplication."""
    chunk_files = sorted(
        [f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".csv")]
    )
    if not chunk_files:
        print("No chunks to merge.")
        return

    print(f"Merging {len(chunk_files)} chunks (no dedup)...")
    header_written = False
    total_rows = 0

    with open(output_path, "w", newline="") as out_f:
        writer = None
        for cf in chunk_files:
            cf_path = os.path.join(chunk_dir, cf)
            with open(cf_path, "r") as in_f:
                reader = csv.DictReader(in_f)
                if not header_written:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                    header_written = True
                for row in reader:
                    writer.writerow(row)
                    total_rows += 1
            if total_rows % 200000 == 0:
                print(f"  ... {total_rows:,} rows merged")

    print(f"Merged {total_rows:,} rows -> {output_path}")

# ------------------------------------------------------------------ #
#  main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="fe.yaml")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = args.split

    # paths
    wi_key = f"window_index_{split}"
    wi_path = cfg.get("window_index", {}).get(wi_key, "")
    if not wi_path:
        wi_path = f"/home/ubuntu/capstone_repo_dataloader/results/dataloader/window_index_{split}.csv"

    out_dir = cfg.get("io", {}).get("output_dir", "results/features_raw")
    if not os.path.isabs(out_dir):
        out_dir = os.path.join("/home/ubuntu/capstone_repo_dataloader", out_dir)

    ckpt_dir = os.path.join(out_dir, f"checkpoints_{split}")
    os.makedirs(ckpt_dir, exist_ok=True)
    output_csv = os.path.join(out_dir, f"features_{split}.csv")
    done_file = os.path.join(ckpt_dir, "files_done.txt")

    # merge-only mode
    if args.merge_only:
        merge_chunks_no_dedup(ckpt_dir, output_csv)
        return

    # load window index
    print(f"Loading window index: {wi_path}")
    wi_df = pd.read_csv(wi_path)
    print(f"  Total windows: {len(wi_df):,}")

    # resolve EDF paths
    base_dir = os.path.dirname(wi_path)
    def resolve_path(p):
        if os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(base_dir, p))

    wi_df["path_abs"] = wi_df["path"].apply(resolve_path)

    # group by file
    grouped = dict(list(wi_df.groupby("path_abs")))
    all_files = sorted(grouped.keys())
    print(f"  Unique EDF files: {len(all_files)}")

    # resume: skip already-done files
    done_files = set()
    if os.path.exists(done_file):
        with open(done_file) as f:
            done_files = set(line.strip() for line in f if line.strip())
    remaining = [f for f in all_files if f not in done_files]
    print(f"  Already done: {len(done_files)}, remaining: {len(remaining)}")

    if not remaining:
        print("All files done. Merging...")
        merge_chunks_no_dedup(ckpt_dir, output_csv)
        return

    # get sfreq from first file
    sfreq = get_sfreq_from_edf(remaining[0])
    print(f"  Sample freq: {sfreq} Hz")

    # extractor
    extractor = AdvancedFeatureExtractor(sfreq=sfreq, cfg=cfg)

    # count existing chunks
    existing_chunks = len([f for f in os.listdir(ckpt_dir) if f.startswith("chunk_") and f.endswith(".csv")])
    chunk_idx = existing_chunks

    # process files
    buffer = []
    buffer_rows = 0
    t0 = time.time()

    for i, edf_path in enumerate(remaining):
        print(f"[{i+1}/{len(remaining)}] {Path(edf_path).name} "
              f"({len(grouped[edf_path]):,} windows)")

        rows = process_one_file(edf_path, grouped[edf_path], extractor, sfreq)
        buffer.extend(rows)
        buffer_rows += len(rows)

        # mark done
        with open(done_file, "a") as f:
            f.write(edf_path + "\n")

        # checkpoint
        if buffer_rows >= args.chunk_size:
            chunk_path = os.path.join(ckpt_dir, f"chunk_{chunk_idx:04d}.csv")
            pd.DataFrame(buffer).to_csv(chunk_path, index=False)
            print(f"  >> Saved chunk {chunk_idx}: {buffer_rows:,} rows")
            chunk_idx += 1
            buffer = []
            buffer_rows = 0
            gc.collect()

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 60
        print(f"  Time: {elapsed/60:.1f}min, Rate: {rate:.1f} files/min")

    # save remaining buffer
    if buffer:
        chunk_path = os.path.join(ckpt_dir, f"chunk_{chunk_idx:04d}.csv")
        pd.DataFrame(buffer).to_csv(chunk_path, index=False)
        print(f"  >> Saved final chunk {chunk_idx}: {buffer_rows:,} rows")

    # merge
    print("\n=== Merging all chunks ===")
    merge_chunks_no_dedup(ckpt_dir, output_csv)

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()
