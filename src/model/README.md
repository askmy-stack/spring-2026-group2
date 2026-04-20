# Model

This directory contains the core training and inference code for the seizure detection models.

It includes:

- custom CNN architectures
- pretrained model integrations
- training loop
- evaluation loop
- sanity-check inference scripts

## Main Files

- [train.py](./train.py): model training entry point
- [run.py](./run.py): quick inference / sanity-check entry point
- [factory.py](./factory.py): model registry
- [architectures.py](./architectures.py): local CNN and wrapper definitions
- [pretrained](./pretrained): pretrained model adapters
- [hugging_face](./hugging_face): Hugging Face ST-EEGFormer integration

## List Available Models

Run from `src`:

```bash
PYTHONPATH=. python3 - <<'PY'
from model.factory import list_models
print(list_models())
PY
```

## Train a Model

Example:

```bash
cd src
PYTHONPATH=. python3 -m model.train \
  --model enhanced_cnn_1d \
  --config-path data_loader/config.yaml \
  --epochs 10 \
  --batch-size 32
```

Another example:

```bash
cd src
PYTHONPATH=. python3 -m model.train \
  --model multiscale_attention_cnn \
  --config-path data_loader/config.yaml
```

## Sanity-Check One Batch

```bash
cd src
PYTHONPATH=. python3 -m model.run \
  --model enhanced_cnn_1d \
  --config-path data_loader/config.yaml \
  --split train
```

## Outputs

Training commonly writes:

- `src/results/model/...` in the original training workflow
- `outputs/models/...`
- `outputs/results/...`
- `outputs/logs/...`

Check the specific training script and artifact helper for the active output layout in your branch.

## Important Compatibility Notes

- CNN workflows in this project typically use 16 channels, 256 Hz, and 1-second windows.
- ST-EEGFormer uses different assumptions and should be matched carefully to its checkpoint requirements.
- BIOT, BENDR, and EEGPT require compatible preprocessing assumptions and additional dependencies.
