# Data Loader

This module handles the end-to-end CHB-MIT ingestion pipeline:

- download EDF files
- parse seizure annotations
- preprocess signals
- standardize channels
- resample and reference EEG
- generate fixed windows and labels
- create BIDS-style outputs
- prepare cached datasets for training

## Main Entry Point

Run from `src`:

```bash
PYTHONPATH=. python3 -m data_loader.main
```

## Configuration

Primary config:

- [config.yaml](./config.yaml)

Important settings in that file:

- `signal.target_sfreq`: target sampling frequency
- `channels.target_count`: number of channels expected by downstream models
- `channels.standard_set`: canonical channel list
- `windowing.window_sec`: window length
- `windowing.stride_sec`: window stride
- `split.policy`: train/val/test split strategy
- `balance`: oversampling and seizure/background balancing

There is also a BIOT-specific config:

- [config_biot.yaml](./config_biot.yaml)

Use that only when working with BIOT-compatible input assumptions.

## Typical Outputs

Depending on config, the pipeline writes:

- `results/bids_dataset`
- `results/dataloader/window_index_train.csv`
- `results/dataloader/window_index_val.csv`
- `results/dataloader/window_index_test.csv`
- `results/cache`

## Useful Modules

- [main.py](./main.py): top-level pipeline entry
- [load_cache.py](./load_cache.py): cached dataloader access
- [pipeline/ingest.py](./pipeline/ingest.py): orchestration
- [core/io.py](./core/io.py): EDF/raw reading
- [core/signal.py](./core/signal.py): preprocessing and normalization
- [dataset/factory.py](./dataset/factory.py): dataset construction

## Notes

- The main model pipeline in this repository uses 16 channels and 256 Hz for the CNN workflows.
- If you change preprocessing assumptions here, make sure the model configs and checkpoints remain compatible.
