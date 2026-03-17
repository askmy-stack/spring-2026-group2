#!/usr/bin/env python
"""
main_loader.py — Single entry point for the EEG data pipeline.

Lives in src/ — all dataloader code is inside src/dataloader/loader/.

Interactive:
    python main_loader.py

CLI:
    python main_loader.py --dataset chbmit --subjects chb01 chb02 chb03 chb04 chb05 --generate
    python main_loader.py --dataset siena --subjects PN00 PN01 PN03 --generate
    python main_loader.py --dataset chbmit --generate --max-bg-per-file 200
    python main_loader.py --dataset chbmit --info
    python main_loader.py --dataset chbmit --test-load

From code:
    from dataloader.loader.loader import get_dataloaders
    train_dl, val_dl, test_dl = get_dataloaders("chbmit")

All defaults (batch size, seed, paths, etc.) are read from config.yaml.
No hardcoded values in this file.
"""

from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

import yaml

# Add src/ to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from dataloader.loader.loader import (
    generate,
    get_dataloaders,
    _load_full_config,
    _csv_dir,
    _tensor_dir,
    _tensors_exist,
    _csvs_exist,
)


# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

def _get_config(config_path: str = "config.yaml") -> dict:
    """Load config.yaml with fallback search."""
    path = Path(config_path)
    if not path.exists():
        path = Path(__file__).resolve().parent / "config.yaml"
    if not path.exists():
        print(f"  [WARN] config.yaml not found, using defaults.")
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _show_info(dataset: str):
    """Show dataset status — what exists on disk."""
    csv_dir = _csv_dir(dataset)
    tensor_dir = _tensor_dir(dataset)

    print(f"\n{'='*50}")
    print(f"  Dataset: {dataset.upper()}")
    print(f"{'='*50}")
    print(f"  CSVs dir     : {csv_dir}")
    print(f"  Tensors dir  : {tensor_dir}")
    print(f"  CSVs exist   : {_csvs_exist(dataset)}")
    print(f"  Tensors exist: {_tensors_exist(dataset)}")

    if _tensors_exist(dataset):
        import torch
        for split in ["train", "val", "test"]:
            meta_path = tensor_dir / split / "metadata.pt"
            if meta_path.exists():
                meta = torch.load(meta_path, weights_only=True)
                n = meta.get("n_windows", 0)
                n_sz = meta.get("n_seizure", 0)
                ratio = n_sz / n if n > 0 else 0
                subjects = meta.get("subject_ids", [])
                print(f"  {split:5s}: {n:,} windows, {n_sz:,} seizure ({ratio:.1%}), subjects={subjects}")
            else:
                data_path = tensor_dir / split / "data.pt"
                if data_path.exists():
                    data = torch.load(data_path, weights_only=True)
                    print(f"  {split:5s}: {len(data):,} tensors, shape {tuple(data.shape[1:])}")

    elif _csvs_exist(dataset):
        import pandas as pd
        for split in ["train", "val", "test"]:
            csv_path = csv_dir / f"window_index_{split}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                n = len(df)
                n_sz = (df["label"] == 1).sum() if not df.empty else 0
                ratio = n_sz / n if n > 0 else 0
                subs = df["subject_id"].nunique() if not df.empty else 0
                print(f"  {split:5s}: {n:,} windows, {n_sz:,} seizure ({ratio:.1%}), {subs} subjects")
    else:
        print(f"  No data found. Run pipeline first.")

    print()


def _test_load(dataset: str, batch_size: int = None):
    """Test loading DataLoaders and print one batch."""
    config = _get_config()
    if batch_size is None:
        batch_size = config.get("pytorch", {}).get("batch_size", 64)

    print(f"\nLoading DataLoaders for {dataset}...")
    train_dl, val_dl, test_dl = get_dataloaders(dataset, batch_size=batch_size)

    batch_data, batch_labels = next(iter(train_dl))
    print(f"\n  Sample batch:")
    print(f"    Data shape : {batch_data.shape}")
    print(f"    Labels     : {batch_labels.shape}")
    print(f"    Label dist : bg={int((batch_labels == 0).sum())}, sz={int((batch_labels == 1).sum())}")
    print(f"    Data range : [{batch_data.min():.3f}, {batch_data.max():.3f}]")
    print(f"    Data mean  : {batch_data.mean():.4f}")
    print(f"\n  DataLoaders working!\n")


def _clear_data(dataset: str):
    """Clear generated CSVs and tensors for a dataset."""
    csv_dir = _csv_dir(dataset)
    tensor_dir = _tensor_dir(dataset)

    removed = []
    if csv_dir.exists():
        shutil.rmtree(csv_dir)
        removed.append(str(csv_dir))
    if tensor_dir.exists():
        shutil.rmtree(tensor_dir)
        removed.append(str(tensor_dir))

    if removed:
        print(f"  Cleared: {', '.join(removed)}")
    else:
        print(f"  Nothing to clear for {dataset}.")


def _ask_subjects(dataset: str) -> list:
    """Ask user which subjects to process."""
    from dataloader.loader.loader import _get_loader
    loader = _get_loader(dataset)
    loader._ensure_records()
    available = loader.list_subjects()
    print(f"\n  Available subjects: {available}")

    user_input = input("\n  Enter subject IDs (space-separated), or 'all' for all: ").strip()
    if user_input.lower() == "all" or user_input == "":
        return None
    return user_input.split()


def _ask_max_background() -> int:
    """Ask user for max background windows per file."""
    config = _get_config()
    default = config.get("windowing", {}).get("max_background_per_file", 150)

    user_input = input(f"\n  Max background windows per EDF file (default {default}, 0 = no cap): ").strip()
    if not user_input:
        return default
    try:
        val = int(user_input)
        return max(0, val)
    except ValueError:
        print(f"  Invalid number, using default {default}.")
        return default


def _download_sample_edf():
    """Download a sample EDF file and run pipeline on it."""
    config = _get_config()
    sample_dir = Path(config.get("paths", {}).get("sample_data_dir", "../data/raw"))
    sample_dir.mkdir(parents=True, exist_ok=True)

    sample_url = "https://physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf"
    sample_path = sample_dir / "chb01" / "chb01_03.edf"

    if sample_path.exists():
        print(f"\n  Sample EDF already exists: {sample_path}")
    else:
        print(f"\n  Downloading sample EDF...")
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        try:
            urllib.request.urlretrieve(sample_url, sample_path)
            print(f"  Downloaded: {sample_path}")
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            return

    # Run CHB-MIT pipeline on just chb01
    print(f"\n  Running pipeline on sample...")
    generate(dataset="chbmit", subjects=["chb01"])


def _run_full_pipeline():
    """
    Run full pipeline — ingest → BIDS → tensors.
    Asks user which dataset to process from available datasets in config.
    """
    config = _get_config()
    datasets = config.get("datasets", {})

    print("\n  Available datasets:")
    keys = list(datasets.keys())
    for i, key in enumerate(keys, 1):
        ds = datasets[key]
        name = ds.get("name", key)
        desc = ds.get("description", "")
        print(f"    {chr(96+i)}) {name}" + (f"  — {desc}" if desc else ""))

    sub = input("  Select: ").strip().lower()

    # Map selection to dataset key
    dataset = None
    for i, key in enumerate(keys):
        if sub == chr(96 + i + 1) or sub == key:
            dataset = key
            break

    if not dataset:
        # Try direct letter mapping
        try:
            idx = ord(sub) - ord('a')
            if 0 <= idx < len(keys):
                dataset = keys[idx]
        except (TypeError, ValueError):
            pass

    if not dataset:
        print("  Invalid choice.")
        return

    subjects = _ask_subjects(dataset)
    max_bg = _ask_max_background()
    generate(dataset=dataset, subjects=subjects, max_background_per_file=max_bg)


def _benchmark_cache():
    """Benchmark tensor loading performance."""
    import time
    config = _get_config()
    pytorch_cfg = config.get("pytorch", {})
    batch_size = pytorch_cfg.get("batch_size", 64)

    print("\n  Which dataset?")
    print("    a) CHB-MIT")
    print("    b) Siena")
    sub = input("  Select: ").strip().lower()
    dataset = "chbmit" if sub in ("a", "chbmit") else "siena" if sub in ("b", "siena") else None

    if not dataset:
        print("  Invalid choice.")
        return

    if not _tensors_exist(dataset):
        print(f"\n  No tensors found for {dataset}. Run pipeline first.")
        return

    print(f"\n  Benchmarking tensor loading for {dataset}...")

    # Time loading
    start = time.time()
    train_dl, val_dl, test_dl = get_dataloaders(dataset, batch_size=batch_size)
    load_time = time.time() - start
    print(f"  DataLoader creation: {load_time:.2f}s")

    # Time iteration
    start = time.time()
    n_batches = 0
    for batch_data, batch_labels in train_dl:
        n_batches += 1
    iter_time = time.time() - start
    print(f"  Full train iteration ({n_batches} batches): {iter_time:.2f}s")
    print(f"  Per batch: {iter_time/max(n_batches,1)*1000:.1f}ms")
    print()


# ═══════════════════════════════════════════════════════════
# INTERACTIVE MENU
# ═══════════════════════════════════════════════════════════

def menu():
    print("\n" + "=" * 60)
    print("  EEG DATA PIPELINE")
    print("=" * 60)
    print("  1. Run full pipeline (ingest → BIDS → tensors)")
    print("  2. Download sample EDF and run pipeline")
    print("  3. Show dataset statistics")
    print("  4. Get PyTorch DataLoaders")
    print("  5. Benchmark cache performance")
    print("  6. Clear generated data")
    print("  7. Download CHB-MIT seizure dataset and run pipeline")
    print("  0. Exit")
    print("=" * 60)


def interactive():
    """Run the interactive menu."""
    while True:
        menu()
        choice = input("  Select: ").strip()

        if not choice:
            continue

        if choice == "0":
            print("  Bye!")
            sys.exit(0)

        elif choice == "1":
            _run_full_pipeline()

        elif choice == "2":
            _download_sample_edf()

        elif choice == "3":
            print("\n  Which dataset?")
            print("    a) CHB-MIT")
            print("    b) Siena")
            print("    c) Both")
            sub = input("  Select: ").strip().lower()
            if sub in ("a", "chbmit"):
                _show_info("chbmit")
            elif sub in ("b", "siena"):
                _show_info("siena")
            elif sub in ("c", "both"):
                _show_info("chbmit")
                _show_info("siena")
            else:
                print("  Invalid choice.")

        elif choice == "4":
            print("\n  Which dataset?")
            print("    a) CHB-MIT")
            print("    b) Siena")
            sub = input("  Select: ").strip().lower()
            dataset = "chbmit" if sub in ("a", "chbmit") else "siena" if sub in ("b", "siena") else None
            if not dataset:
                print("  Invalid choice.")
            else:
                try:
                    train_dl, val_dl, test_dl = get_dataloaders(dataset)
                    print(f"\n  Train batches : {len(train_dl)}")
                    print(f"  Val batches   : {len(val_dl)}")
                    print(f"  Test batches  : {len(test_dl)}")
                except FileNotFoundError as e:
                    print(f"\n  {e}")

        elif choice == "5":
            _benchmark_cache()

        elif choice == "6":
            print("\n  Clear generated data for:")
            print("    a) CHB-MIT")
            print("    b) Siena")
            print("    c) Both")
            sub = input("  Select: ").strip().lower()
            confirm = input("  Are you sure? (y/n): ").strip().lower()
            if confirm == "y":
                if sub in ("a", "chbmit"):
                    _clear_data("chbmit")
                elif sub in ("b", "siena"):
                    _clear_data("siena")
                elif sub in ("c", "both"):
                    _clear_data("chbmit")
                    _clear_data("siena")
            else:
                print("  Cancelled.")

        elif choice == "7":
            print("\n  [CHB-MIT] Pediatric epilepsy, 256 Hz, 24 subjects")
            subjects = _ask_subjects("chbmit")
            max_bg = _ask_max_background()
            generate(dataset="chbmit", subjects=subjects, max_background_per_file=max_bg)

        else:
            print("  Invalid option.")
            continue

        input("\n  Press Enter to continue...")


# ═══════════════════════════════════════════════════════════
# CLI MODE
# ═══════════════════════════════════════════════════════════

def cli():
    """Run in CLI mode with arguments."""
    config = _get_config()

    # Read defaults from config
    balance_cfg = config.get("balance", {})
    windowing_cfg = config.get("windowing", {})
    pytorch_cfg = config.get("pytorch", {})
    system_cfg = config.get("system", {})
    available_datasets = list(config.get("datasets", {}).keys())

    parser = argparse.ArgumentParser(
        description="EEG Data Pipeline — download, process, and serve PyTorch DataLoaders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_loader.py                                                           (interactive menu)
  python main_loader.py --dataset chbmit --subjects chb01 chb02 chb03 --generate  (CLI generate)
  python main_loader.py --dataset chbmit --generate --max-bg-per-file 200         (custom cap)
  python main_loader.py --dataset siena --generate                                (all Siena subjects)
  python main_loader.py --dataset chbmit --info                                   (show stats)
  python main_loader.py --dataset chbmit --test-load                              (verify DataLoaders)
        """,
    )
    parser.add_argument("--dataset", type=str, choices=available_datasets or ["chbmit", "siena"],
                        help="Dataset to process (from config.yaml)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subject IDs (default: all available)")
    parser.add_argument("--generate", action="store_true",
                        help="Run full pipeline: download → process → CSVs → tensors")
    parser.add_argument("--info", action="store_true",
                        help="Show dataset status")
    parser.add_argument("--test-load", action="store_true",
                        help="Test loading DataLoaders and print one batch")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download and re-process")
    parser.add_argument("--seizure-ratio", type=float,
                        default=balance_cfg.get("seizure_ratio", 0.3),
                        help=f"Target seizure ratio (config default: {balance_cfg.get('seizure_ratio', 0.3)})")
    parser.add_argument("--max-bg-per-file", type=int,
                        default=windowing_cfg.get("max_background_per_file", 150),
                        help=f"Max background windows per EDF (config default: {windowing_cfg.get('max_background_per_file', 150)}, 0 = no cap)")
    parser.add_argument("--seed", type=int,
                        default=system_cfg.get("random_seed", 42),
                        help=f"Random seed (config default: {system_cfg.get('random_seed', 42)})")
    parser.add_argument("--batch-size", type=int,
                        default=pytorch_cfg.get("batch_size", 64),
                        help=f"Batch size (config default: {pytorch_cfg.get('batch_size', 64)})")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--clear", action="store_true",
                        help="Clear generated CSVs and tensors")

    args = parser.parse_args()

    if args.clear:
        if not args.dataset:
            print("Error: --clear requires --dataset")
            return
        _clear_data(args.dataset)
        return

    if args.info:
        if not args.dataset:
            for ds in available_datasets:
                _show_info(ds)
        else:
            _show_info(args.dataset)
        return

    if args.generate:
        if not args.dataset:
            print("Error: --generate requires --dataset")
            return
        generate(
            dataset=args.dataset,
            subjects=args.subjects,
            force=args.force,
            seizure_ratio=args.seizure_ratio,
            seed=args.seed,
            max_background_per_file=args.max_bg_per_file,
            config_path=args.config,
        )

    if args.test_load:
        if not args.dataset:
            print("Error: --test-load requires --dataset")
            return
        _test_load(args.dataset, args.batch_size)

    if not args.generate and not args.info and not args.test_load:
        parser.print_help()


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) == 1:
        interactive()
    else:
        cli()
