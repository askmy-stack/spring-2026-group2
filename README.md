# EEG Seizure Detection — Data Pipeline

End-to-end data pipeline for EEG seizure detection research. Downloads raw EEG recordings from PhysioNet, preprocesses signals, builds labeled windows, and serves pre-saved PyTorch tensors for instant model training.

## Supported Datasets

| Dataset | Population | Subjects | Native Hz | Channels | Source |
|---------|-----------|----------|-----------|----------|--------|
| **CHB-MIT** | Pediatric epilepsy | 24 | 256 Hz | 23 EEG | [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) |
| **Siena Scalp EEG** | Adult epilepsy | 14 | 512 Hz | 31 EEG | [PhysioNet](https://physionet.org/content/siena-scalp-eeg/1.0.0/) |

## Quick Start

### Install dependencies

```bash
cd src/
pip install -r requirements.txt
```

### Interactive mode

```bash
python main_loader.py
```

This launches a menu:

```
============================================================
  EEG DATA PIPELINE
============================================================
  1. Generate CHB-MIT dataset      (download → process → tensors)
  2. Generate Siena dataset         (download → process → tensors)
  3. Generate from local EDF folder (your own seizure data)
  4. Show dataset statistics
  5. Get PyTorch DataLoaders
  6. Test DataLoader output
  7. Clear generated data
  0. Exit
============================================================
```

### CLI mode

```bash
# Generate CHB-MIT with 5 subjects
python main_loader.py --dataset chbmit --subjects chb01 chb02 chb03 chb04 chb05 --generate

# Generate Siena with 5 subjects
python main_loader.py --dataset siena --subjects PN00 PN01 PN03 PN05 PN06 --generate

# Check dataset status
python main_loader.py --dataset chbmit --info

# Verify DataLoader output
python main_loader.py --dataset chbmit --test-load

# Clear generated data
python main_loader.py --dataset chbmit --clear
```

### From Python (for training scripts)

```python
from main_loader import get_dataloaders

train_dl, val_dl, test_dl = get_dataloaders("chbmit", batch_size=64)

for batch, labels in train_dl:
    # batch:  (64, 16, 256) — 64 windows, 16 channels, 256 samples
    # labels: (64,)         — 0 = background, 1 = seizure
    pass
```

## Pipeline Stages

The `generate()` function runs 5 stages in sequence:

### Stage 1 — Download
Downloads raw EDF files and seizure annotation files from PhysioNet. Files are cached in `data/raw/{dataset}/` so subsequent runs skip downloading.

### Stage 2 — Preprocess
For each EDF file:
- Pick EEG channels only
- Resample to 256 Hz (Siena: 512 → 256)
- Bandpass filter 1–50 Hz
- Notch filter (60 Hz for CHB-MIT, 50 Hz for Siena)
- Average reference
- Standardize to 16 common channels: Fp1, Fp2, F3, F4, F7, F8, Fz, Cz, T7, T8, P7, P8, C3, C4, O1, O2
- Save processed EDF in BIDS format

### Stage 3 — Windowing
- Slide 1-second windows across each recording
- Label: seizure (1) if ≥50% overlap with annotated seizure interval
- Exclude background windows within 300 seconds of any seizure (reduces noise)
- **Cap background at 150 windows per file** — seizure windows are never capped

### Stage 4 — Split & Balance
- **Subject-independent splits**: no subject appears in more than one split (prevents data leakage)
- Default ratio: 70% train / 15% val / 15% test
- Stratified: seizure subjects distributed across all splits
- Oversample seizure windows to 30% ratio in each split

### Stage 5 — Tensorize
- Convert CSV window indices → PyTorch tensors
- Z-score normalize per channel
- Save as `data.pt` (N, 16, 256) and `labels.pt` (N,)
- Subsequent runs load tensors instantly — no EDF I/O at training time

## Project Structure

```
src/
├── main_loader.py                 # Entry point (interactive menu + CLI)
├── requirements.txt
├── config.yaml
├── dataloaders/
│   ├── __init__.py                # Exports: generate(), get_dataloaders()
│   ├── chbmit/
│   │   ├── __init__.py
│   │   ├── config.py              # CHB-MIT metadata, URLs, 24 subjects
│   │   └── download.py            # PhysioNet downloader + summary.txt parser
│   ├── siena/
│   │   ├── __init__.py
│   │   ├── config.py              # Siena metadata, URLs, 14 subjects
│   │   └── download.py            # PhysioNet downloader + seizure CSV parser
│   └── common/
│       ├── __init__.py
│       ├── loader.py              # generate() + get_dataloaders()
│       ├── windowing.py           # Window building + labeling + balancing
│       ├── splits.py              # Subject-independent splitting
│       └── tensor_writer.py       # CSV → .pt tensors + TensorDataset
├── core/                          # Signal processing (unchanged)
│   ├── augment.py
│   ├── bids.py
│   ├── cache.py
│   ├── channels.py
│   ├── download.py
│   ├── io.py
│   ├── labels.py
│   ├── signal.py
│   └── stratify.py
├── features/                      # Feature engineering (WIP)
└── tests/
```

## Generated Output

After running `generate()`, these directories are created:

```
results/
├── bids_dataset/{dataset}/        # Processed EDFs in BIDS format
├── dataloader/{dataset}/          # Window index CSVs
│   ├── window_index_train.csv
│   ├── window_index_val.csv
│   └── window_index_test.csv
└── tensors/{dataset}/             # Pre-saved PyTorch tensors
    ├── train/
    │   ├── data.pt                # (N, 16, 256) float32
    │   ├── labels.pt              # (N,) long
    │   └── metadata.pt            # dict with stats
    ├── val/
    └── test/
```

## Key Design Decisions

**Background capping (150/file):** A 1-hour EDF produces ~3,600 one-second windows. Most are redundant background. We keep 150 randomly sampled background windows per file while preserving ALL seizure windows. This reduces chb01 from 197K to ~10K windows with zero seizure data loss.

**Subject-independent splits:** Subjects never appear in more than one split. This prevents the model from memorizing patient-specific patterns and ensures valid evaluation on unseen patients.

**Pre-saved tensors:** The first run processes EDFs and saves tensors. All subsequent runs load tensors directly — no MNE, no EDF I/O. Training startup drops from minutes to seconds.

**Three-level caching in `get_dataloaders()`:**
1. Tensors exist → load directly (instant)
2. CSVs exist but no tensors → tensorize, then load
3. Nothing exists → error, run `generate()` first

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | `chbmit` or `siena` | required |
| `--subjects` | Space-separated subject IDs | all |
| `--generate` | Run full pipeline | — |
| `--info` | Show dataset status | — |
| `--test-load` | Test DataLoader output | — |
| `--clear` | Delete generated CSVs + tensors | — |
| `--force` | Re-download and re-process | — |
| `--seizure-ratio` | Target seizure ratio | 0.3 |
| `--seed` | Random seed | 42 |
| `--batch-size` | Batch size for test-load | 64 |

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.24
- pandas ≥ 2.0
- torch ≥ 2.0
- mne ≥ 1.5
- tqdm ≥ 4.65
- scikit-learn ≥ 1.3
- matplotlib ≥ 3.7