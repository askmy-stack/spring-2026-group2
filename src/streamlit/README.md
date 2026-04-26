# Streamlit App

This directory contains the interactive EEG seizure detection app.

Current app features:

- local EDF upload
- server-side EDF selection from `~/EEG_SEIZURE_DETECTION/uploads`
- CNN inference with:
  - `enhanced_cnn_1d`
  - `multiscale_attention_cnn`
- engineered-feature inference with:
  - `tabnet_features`
- optional visual diagnostics:
  - trace view
  - band-power view
  - spectrogram
  - channel correlation

## Main File

- [app.py](./app.py)

## Run Locally

From project root:

```bash
source .venv/bin/activate
streamlit run src/streamlit/app.py --server.address 127.0.0.1 --server.port 8502
```

Then open:

```text
http://127.0.0.1:8502
```

Do not run the app with `PYTHONPATH=src` unless you have verified there is no namespace conflict in your environment.

## Run On Ubuntu / Remote Server

On the server:

```bash
cd ~/EEG_SEIZURE_DETECTION
source .venv/bin/activate
streamlit run src/streamlit/app.py --server.address 127.0.0.1 --server.port 8502
```

On your local machine:

```bash
ssh -i ~/.ssh/capstone_ritu.pem -L 8502:127.0.0.1:8502 ubuntu@YOUR-EC2-HOST
```

Then open locally:

```text
http://127.0.0.1:8502
```

## Server File Mode

If you want to avoid repeated browser uploads for large EDFs:

1. copy EDF files to the server:

```bash
mkdir -p ~/EEG_SEIZURE_DETECTION/uploads
```

2. place `.edf` files in that directory
3. in the app, select:
   - `Use server file`

The app will read directly from the Ubuntu filesystem instead of uploading through the browser tunnel.

## Dependencies

The app relies on:

- `streamlit`
- `mne`
- `torch`
- `pytorch-tabnet` for TabNet inference
- `PyWavelets`
- `numba`

If TabNet dependencies are missing, the CNN paths can still run, but the TabNet model will fail when selected.
