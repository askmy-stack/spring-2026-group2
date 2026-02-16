# EEG Seizure Detection Dataloader

## Core Files (4 Files Only)

```
project/
├── dataloader.py     # Main dataloader with all functionality
├── main.py           # CLI interface for operations
├── verify.py         # Test suite
└── config.yaml       # Configuration
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy pandas scipy mne mne-bids PyYAML
```

### 2. Configure Paths

Edit `config.yaml`:
```yaml
dataset:
  raw_root: "path/to/your/raw/eeg/data"
  results_root: "results"
```

### 3. Run Pipeline

```bash
python main.py
# Select option 1: Run Full Pipeline
```

### 4. Check Balance

```bash
python main.py
# Select option 4: Show Dataset Statistics
```

### 5. Verify System

```bash
python verify.py
```

All tests should pass.

## Testing with Other EEG Datasets

### Available Public Datasets

#### 1. CHB-MIT Scalp EEG Database (PhysioNet)
- **Link**: https://physionet.org/content/chbmit/1.0.0/
- **Description**: 24 pediatric subjects, 916 hours of EEG, 198 seizures
- **Format**: EDF
- **Channels**: 23 channels (standardized to 16 by our loader)
- **Sampling Rate**: 256 Hz

**Download:**
```bash
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/
```

**Configure:**
```yaml
dataset:
  raw_root: "chbmit/1.0.0"
```

#### 2. TUH EEG Seizure Corpus
- **Link**: https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
- **Description**: Largest publicly available seizure corpus
- **Format**: EDF
- **Note**: Requires registration

**After download, configure:**
```yaml
dataset:
  raw_root: "tuh_eeg_seizure/v2.0.0/edf"
```

#### 3. Siena Scalp EEG Database
- **Link**: https://physionet.org/content/siena-scalp-eeg/1.0.0/
- **Description**: 14 subjects with absence seizures
- **Format**: EDF
- **Channels**: 29 channels (standardized to 16)

**Download:**
```bash
wget -r -N -c -np https://physionet.org/files/siena-scalp-eeg/1.0.0/
```

#### 4. OpenNeuro EEG Datasets
- **Link**: https://openneuro.org
- **Search**: Filter by "EEG" modality
- **Format**: BIDS-compliant
- **Examples**:
  - ds003190: EEG data (TUH subset)
  - ds002778: Motor imagery
  - ds003645: Epilepsy recordings

**Download (using AWS CLI):**
```bash
pip install awscli
aws s3 sync --no-sign-request s3://openneuro.org/ds003190 data/raw/ds003190/
```

### Using Downloaded Datasets

#### Step 1: Place Data
```bash
# Your directory structure:
data/raw/your_dataset/
├── sub-01/
│   └── eeg/
│       └── sub-01_task-rest_eeg.edf
├── sub-02/
└── ...
```

#### Step 2: Update Config
```yaml
dataset:
  raw_root: "data/raw/your_dataset"
  results_root: "results"
```

Update config:
```yaml
dataset:
  raw_root: "data/raw/your_dataset"
```

#### Step 4: Run Pipeline
```bash
python main.py
# Select option 1: Run Full Pipeline
```

#### Step 5: Verify
```bash
python main.py
# Select option 4: Show Dataset Statistics
```

### Expected Output Format

All datasets will be standardized to:
- 16 channels (standard 10-20 system)
- 256 Hz sampling rate
- 1-second windows
- Binary labels (0=background, 1=seizure)

## Main Menu Options

```
1. Run Full Pipeline    - Process raw EEG data
2. Generate BIDS Report - Validate BIDS structure
3. Quick Verification   - Test data integrity
4. Show Statistics      - View dataset distribution
5. Export Metadata      - Save dataset info
6. Clear Cache          - Remove cached data
```

## Configuration

Key settings in `config.yaml`:

```yaml
signal:
  target_sfreq: 256
  bandpass: [1.0, 50.0]
  notch: 60.0

windowing:
  window_sec: 1.0
  stride_sec: 1.0

balance:
  enable: true
  seizure_ratio: 0.3

split:
  train: 0.70
  val: 0.15
  test: 0.15
```

## Troubleshooting

### No Data Found
```bash
# Check paths in config.yaml
# Ensure raw data exists
python main.py  # Run option 1
```

### Imbalanced Splits
```python
from dataloader import create_loader
loader = create_loader('enhanced', mode='train')
loader.check_balance()
loader.rebalance_splits()  # If needed
```

### Out of Memory
```python
# Reduce cache size
loader = create_loader('cached', mode='train', cache_memory_mb=500)
```

### Tests Failing
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Run pipeline first
python main.py  # Option 1

# Then test
python verify.py
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
```

## Output Structure

After running pipeline:

```
results/
├── bids_dataset/           # BIDS-formatted data
│   ├── sub-01/
│   ├── sub-02/
│   └── ...
└── dataloader/
    ├── window_index_train.csv
    ├── window_index_val.csv
    ├── window_index_test.csv
    └── cache/              # Cached data
```

## Example Datasets and Expected Results

### CHB-MIT (24 subjects)
```
Expected output:
- Train: ~150K windows, 20-30% seizures
- Val: ~40K windows, 20-30% seizures
- Test: ~30K windows, 20-30% seizures
```

### TUH EEG (Larger corpus)
```
Expected output:
- Train: ~500K+ windows, 15-25% seizures
- Val: ~100K+ windows, 15-25% seizures
- Test: ~100K+ windows, 15-25% seizures
```

### Siena (14 subjects, absence seizures)
```
Expected output:
- Train: ~50K windows, 10-20% seizures
- Val: ~15K windows, 10-20% seizures
- Test: ~15K windows, 10-20% seizures
```

## Quick Reference

```python
# Import
from dataloader import create_loader

# Load data
train = create_loader('enhanced', mode='train', augment=True)

# Check balance
balance_info = train.check_balance()

# Fix if needed
if balance_info['needs_rebalancing']:
    train.rebalance_splits()

# Get weights
weights = train.get_class_weights()

# Use in training
from torch.utils.data import DataLoader
loader = DataLoader(train, batch_size=64, shuffle=True)
```
