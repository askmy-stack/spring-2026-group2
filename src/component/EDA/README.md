# EDA

This directory contains scripts for exploratory data analysis on the seizure dataset.

Typical EDA tasks include:

- dataset inventory
- subject and file counts
- seizure duration distribution
- channel usage frequency
- raw signal inspection
- amplitude statistics
- PSD and band-power plots

## Main Script

- [eda_chbmit.py](./eda_chbmit.py)

Run from the **project root**:

```bash
python -m src.component.EDA.eda_chbmit
```

## Expected Inputs

The EDA scripts assume the data loader pipeline has already produced dataset outputs and that EDF files or processed metadata are available.

## Typical Outputs

Common outputs are written under:

- `src/results/eda`
- `results/eda`

Typical saved figures include:

- dataset overview
- channel frequency
- seizure duration distribution
- per-channel amplitude statistics
- PSD per channel
- band-power summaries

## When To Use

Use EDA before model training to:

- validate dataset integrity
- understand class imbalance
- inspect channel coverage
- choose visualization examples for reports or presentations
