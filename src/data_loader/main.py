from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.ingest import run_chbmit_pipeline
from dataset.factory import create_loader, create_pytorch_dataloaders

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = "config.yaml"

# ============================================================================
# STEP 1: Download full CHB-MIT dataset and run complete pipeline
# This includes: download -> BIDS conversion -> windowing -> labeling -> splitting
# ============================================================================
print("=" * 80)
print("  FULL CHB-MIT EEG PIPELINE")
print("=" * 80)
print("\n[1/2] Downloading and processing CHB-MIT dataset...")

output_dir = run_chbmit_pipeline(CONFIG)

# ============================================================================
# STEP 2: Create PyTorch DataLoaders with caching and augmentation
# ============================================================================
print("\n[2/2] Creating PyTorch DataLoaders...")
train_dl, val_dl, test_dl = create_pytorch_dataloaders(config_path=CONFIG,loader_type="enhanced",augment_train=True,)

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("  PIPELINE COMPLETE")
print("=" * 80)
print(f"Output directory: {output_dir}")

# Show DataLoader batch counts
print(f"\nDataLoader statistics:")
print(f"  Train batches: {len(train_dl)}")
print(f"  Val batches  : {len(val_dl)}")
print(f"  Test batches : {len(test_dl)}")

# Show detailed dataset statistics
print(f"\nDataset details:")
for mode in ["train", "val", "test"]:
    loader = create_loader("standard", config_path=CONFIG, mode=mode)
    s = loader.get_summary()
    print(f"  {mode.upper():5s}: {s['total_windows']:5d} windows | "
          f"{s['n_subjects']:2d} subjects | "
          f"seizure={s['labels'].get(1, 0):4d}")

print("\nData is ready for model training!")
print("=" * 80)
