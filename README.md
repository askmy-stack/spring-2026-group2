# EEG Seizure Detection Dataloader

A modular EEG data pipeline for seizure detection, supporting ingestion, BIDS conversion, windowing, balancing, stratified splitting, caching, augmentation, and PyTorch dataloader creation.

## Repository Structure

```
spring-2026-group2/
├── README.md
├── .gitignore
└── src/
    ├── main.py               # CLI menu interface
    ├── verify.py             # Test runner (calls src/tests/test_pipeline.py)
    ├── config.yaml           # All pipeline configuration
    ├── core/
    │   ├── io.py             # EDF reading, directory scanning, sample download
    │   ├── signal.py         # Preprocessing (bandpass, notch, resample, normalize)
    │   ├── channels.py       # Channel standardization to 16-channel 10-20 set
    │   ├── labels.py         # Window indexing, seizure labeling, dataset balancing
    │   ├── stratify.py       # Subject-independent stratified train/val/test split
    │   ├── augment.py        # Data augmentation (time warp, magnitude scale, noise, shift)
    │   ├── bids.py           # BIDS conversion and participants.tsv management
    │   ├── cache.py          # Pickle-based disk/memory cache with hit/miss stats
    │   └── download.py       # Dataset download utilities
    ├── pipeline/
    │   └── ingest.py         # Full pipeline orchestration (run_pipeline, run_chbmit_pipeline)
    ├── dataset/
    │   ├── base.py           # BaseEEGDataset (abstract PyTorch Dataset)
    │   ├── loaders.py        # StandardEEGLoader, CachedEEGLoader, ParallelEEGLoader, EnhancedEEGLoader
    │   └── factory.py        # create_loader() and create_pytorch_dataloaders() factory functions
    ├── tests/
    │   └── test_pipeline.py  # 19 unit tests covering all core modules
    └── results/
        ├── bids_dataset/     # BIDS-formatted output (sub-*/eeg/*)
        └── dataloader/
            ├── window_index_train.csv
            ├── window_index_val.csv
            └── window_index_test.csv
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy pandas scipy mne mne-bids PyYAML dask
```

### 2. Configure Paths

Edit `src/config.yaml`:
```yaml
dataset:
  raw_root: "../data/raw"
  results_root: "results"
```

### 3. Run Pipeline

```bash
cd src
python main.py
# Select option 1: Run full pipeline (ingest -> BIDS -> index)
```

### 4. Run Tests

```bash
cd src
python verify.py
```

All 19 tests should pass.

## Main Menu Options

```
1. Run full pipeline (ingest -> BIDS -> index)   - Process raw EEG data end-to-end
2. Download sample EDF and run pipeline          - Download a sample EDF and process it
3. Show dataset statistics                       - View window counts and label distribution
4. Get PyTorch DataLoaders                       - Create train/val/test DataLoaders
5. Benchmark cache performance                   - Measure cache speedup
6. Clear cache                                   - Remove cached data
7. Download CHB-MIT seizure dataset and run pipeline
0. Exit
```

## Test Suite

Tests are in `src/tests/test_pipeline.py` and run via `src/verify.py`.

| Test | Description |
|------|-------------|
| `test_download_sample_edf` | Downloads a sample EDF file |
| `test_read_raw` | Reads an EDF file with MNE |
| `test_preprocess` | Bandpass filter + resample to target sfreq |
| `test_channel_standardization_reduce` | Reduces channels to standard 16 |
| `test_channel_standardization_expand` | Expands sparse channels to 16 via interpolation |
| `test_normalize_signal` | Z-score normalization of EEG signal |
| `test_augmentation` | Full augmentation pipeline on a window |
| `test_augmentation_functions` | Individual augmentation functions (warp, scale, noise, shift) |
| `test_cache_put_get` | Cache write and read |
| `test_cache_miss` | Cache miss returns None |
| `test_cache_stats` | Hit/miss counters |
| `test_build_window_index` | Window index creation with and without seizure intervals |
| `test_balance_index` | Oversampling minority class to target seizure ratio |
| `test_stratify_subjects` | Subject-independent split with no overlap between splits |
| `test_bids_conversion` | EDF -> BIDS format conversion |
| `test_loader_class_hierarchy` | All loaders are subclasses of BaseEEGDataset |
| `test_loader_empty` | create_loader() returns correct loader type |
| `test_loader_shape` | Loaded tensor shape matches config (channels x window_samples) |
| `test_class_weights` | get_class_weights() returns a 2-element tensor |

## Configuration

Key settings in `src/config.yaml`:

```yaml
signal:
  target_sfreq: 256          # Resample to 256 Hz
  bandpass: [1.0, 50.0]      # Bandpass filter
  notch: 60.0                # Notch filter (60 Hz)
  reference: "average"       # Average reference

channels:
  target_count: 16
  policy: "spatial_selection"
  standard_set: ["Fp1","Fp2","F3","F4","F7","F8","Fz","Cz",
                 "T7","T8","P7","P8","C3","C4","O1","O2"]

windowing:
  window_sec: 1.0
  stride_sec: 1.0

balance:
  enable: true
  seizure_ratio: 0.3
  method: "oversample"

split:
  policy: "subject_independent"
  train: 0.70
  val: 0.15
  test: 0.15

caching:
  enable: true
  max_memory_mb: 4096

pytorch:
  batch_size: 32
  num_workers: 4

augmentation:
  enable: true
  train_only: true
  seizure_prob: 0.8
  background_prob: 0.3
```

## Loader Types

| Type | Class | Description |
|------|-------|-------------|
| `standard` | `StandardEEGLoader` | Basic window-based loader |
| `cached` | `CachedEEGLoader` | Disk/memory cache for fast repeated access |
| `parallel` | `ParallelEEGLoader` | Multi-worker parallel loading |
| `enhanced` | `EnhancedEEGLoader` | Cache + augmentation + class weights |

## API Usage

```python
from dataset.factory import create_loader, create_pytorch_dataloaders

# Single loader
loader = create_loader("enhanced", config_path="config.yaml", mode="train", augment_data=True)
data, label = loader[0]  # torch.Tensor (16, 256), torch.Tensor scalar

# Class weights for weighted loss
weights = loader.get_class_weights()  # torch.Tensor([w_neg, w_pos])

# PyTorch DataLoaders (train/val/test)
train_dl, val_dl, test_dl = create_pytorch_dataloaders(
    config_path="config.yaml",
    loader_type="enhanced",
    augment_train=True,
)
```

## Compatible Datasets

### CHB-MIT Scalp EEG (PhysioNet)
- **Link**: https://physionet.org/content/chbmit/1.0.0/
- 24 pediatric subjects, 916 hours, 198 seizures, EDF format

```bash
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/
```

### TUH EEG Seizure Corpus
- **Link**: https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
- Largest public seizure corpus, requires registration

### Siena Scalp EEG (PhysioNet)
- **Link**: https://physionet.org/content/siena-scalp-eeg/1.0.0/
- 14 subjects, absence seizures, 29 channels standardized to 16

### OpenNeuro EEG (BIDS format)
- **Link**: https://openneuro.org — filter by EEG modality

## Output

After running the pipeline, all datasets are standardized to:
- **16 channels** (standard 10-20 system)
- **256 Hz** sampling rate
- **1-second windows**
- **Binary labels**: 0 = background, 1 = seizure

```
src/results/
├── bids_dataset/
│   └── sub-<id>/eeg/
│       ├── sub-<id>_task-eeg_run-01_eeg.edf
│       ├── sub-<id>_task-eeg_run-01_eeg.json
│       ├── sub-<id>_task-eeg_run-01_eeg_channels.tsv
│       └── sub-<id>_task-eeg_run-01_eeg_events.tsv
└── dataloader/
    ├── window_index_train.csv
    ├── window_index_val.csv
    └── window_index_test.csv
```

## Requirements

```
Python >= 3.8
torch >= 1.9
numpy >= 1.21
pandas >= 1.3
scipy >= 1.7
mne >= 1.0
mne-bids >= 0.10
PyYAML >= 5.4
dask (optional, for parallel caching)
```

## Troubleshooting

**No data found**: Verify `raw_root` in `config.yaml` points to a directory containing `.edf` files.

**Tests failing**: Run the pipeline first (option 1 or 2 in `main.py`) to populate `results/`, then re-run tests.

**Out of memory**: Reduce `caching.max_memory_mb` in `config.yaml`.

**Imbalanced splits**: The pipeline automatically oversamples the seizure class to `balance.seizure_ratio = 0.3`. Adjust in `config.yaml` if needed.