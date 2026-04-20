from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from data_loader.main import CONFIG
from data_loader.pipeline.ingest import run_chbmit_pipeline
from data_loader.dataset.factory import create_pytorch_dataloaders


def main():
    print("=" * 80)
    print("  EEG SEIZURE DETECTION PIPELINE")
    print("=" * 80)
    print(f"Config: {CONFIG}")

    print("\n[1/2] Running dataloader pipeline...")
    output_dir = run_chbmit_pipeline(CONFIG)

    print("\n[2/2] Building PyTorch DataLoaders...")
    train_dl, val_dl, test_dl = create_pytorch_dataloaders(
        config_path=CONFIG,
        loader_type="enhanced",
        augment_train=True,
    )

    print("\n" + "=" * 80)
    print("  PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Train batches: {len(train_dl)}")
    print(f"Val batches  : {len(val_dl)}")
    print(f"Test batches : {len(test_dl)}")


if __name__ == "__main__":
    main()
