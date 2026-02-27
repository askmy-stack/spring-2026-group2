from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Optional (recommended): makes it runnable even WITHOUT PYTHONPATH=src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # .../src

from dataloader.pipeline.ingest import run_pipeline, run_chbmit_pipeline
from dataloader.dataset.factory import create_loader, create_pytorch_dataloaders


def menu():
    print("=" * 60)
    print("  EEG DATA PIPELINE")
    print("=" * 60)
    print("1. Run full pipeline (ingest -> BIDS -> index)")
    print("2. Download sample EDF and run pipeline")
    print("3. Show dataset statistics")
    print("4. Get PyTorch DataLoaders")
    print("5. Benchmark cache performance")
    print("6. Clear cache")
    print("7. Download CHB-MIT seizure dataset and run pipeline")
    print("0. Exit")
    print("=" * 60)


def show_stats(config_path: str):
    for mode in ["train", "val", "test"]:
        loader = create_loader("standard", config_path=config_path, mode=mode)
        s = loader.get_summary()
        print(f"\n{mode.upper()}: {s['total_windows']} windows | "
              f"subjects={s['n_subjects']} | "
              f"labels={s['labels']}")


def get_dataloaders(config_path: str):
    train_dl, val_dl, test_dl = create_pytorch_dataloaders(
        config_path=config_path,
        loader_type="enhanced",
        augment_train=True,
    )
    print(f"Train batches : {len(train_dl)}")
    print(f"Val batches   : {len(val_dl)}")
    print(f"Test batches  : {len(test_dl)}")
    return train_dl, val_dl, test_dl


def benchmark(config_path: str):
    loader = create_loader("enhanced", config_path=config_path, mode="train")
    if len(loader) == 0:
        print("No data. Run the pipeline first.")
        return
    r = loader.benchmark_cache(num_samples=20)
    print(f"Cache speedup: {r['speedup']:.2f}x")


def clear_cache(config_path: str):
    loader = create_loader("cached", config_path=config_path, mode="train")
    loader.clear_cache()
    print("Cache cleared.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to dataloader config YAML")
    args = ap.parse_args()
    CONFIG = args.config  # ✅ use the passed config

    while True:
        menu()
        choice = input("Select: ").strip()
        if not choice:
            continue

        if choice == "0":
            sys.exit(0)
        elif choice == "1":
            run_pipeline(CONFIG, download_sample=False)
        elif choice == "2":
            run_pipeline(CONFIG, download_sample=True)
        elif choice == "3":
            show_stats(CONFIG)
        elif choice == "4":
            get_dataloaders(CONFIG)
        elif choice == "5":
            benchmark(CONFIG)
        elif choice == "6":
            clear_cache(CONFIG)
        elif choice == "7":
            run_chbmit_pipeline(CONFIG)  # ✅ now it will open the right YAML
        else:
            print("Invalid option.")
            continue

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()