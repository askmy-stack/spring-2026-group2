# EEG Seizure Detection

This repository contains the main code for:

- CHB-MIT data ingestion and preprocessing
- exploratory data analysis (EDA)
- deep-learning models for seizure detection
- a Streamlit app for interactive inference and visualization

The main code lives under [src](./src).

## Project Layout

- [src/data_loader](./src/data_loader): dataset download, preprocessing, BIDS conversion, window generation, cached dataloaders
- [src/EDA](./src/EDA): dataset inspection and visualization scripts
- [src/model](./src/model): core training, inference, pretrained model integrations
- [src/models](./src/models): wrapper structure and launcher scripts used for packaged model workflows
- [src/feature](./src/feature): engineered features and TabNet training artifacts
- [src/streamlit](./src/streamlit): interactive app

## Environment

Typical setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install huggingface_hub safetensors pytorch-tabnet PyWavelets numba
```

Depending on the machine, some packages may already be installed in the environment.

## Common Workflows

### 1. Run the data pipeline

```bash
cd src
PYTHONPATH=. python3 -m data_loader.main
```

This downloads and processes CHB-MIT, writes BIDS-style outputs, and prepares window indices and cache files.

### 2. Train a model

Example:

```bash
cd src
PYTHONPATH=. python3 -m model.train \
  --model enhanced_cnn_1d \
  --config-path data_loader/config.yaml
```

### 3. Run the Streamlit app

```bash
streamlit run src/streamlit/app.py --server.address 127.0.0.1 --server.port 8502
```

If the app runs on a remote Ubuntu instance, use SSH port forwarding from your local machine.

## Outputs

This project commonly writes outputs under:

- `outputs/models`
- `outputs/results`
- `outputs/logs`
- `results`
- `src/results`

Large data files, EDFs, checkpoints, and generated caches should not be committed to Git unless explicitly intended.

## Additional READMEs

- [Data Loader README](./src/data_loader/README.md)
- [EDA README](./src/EDA/README.md)
- [Model README](./src/model/README.md)
- [Streamlit README](./src/streamlit/README.md)
