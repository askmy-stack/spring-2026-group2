# Streamlit EEG Demo

This directory contains the Streamlit app shell for interactive EEG upload and input validation.

Run it from the project root:

```bash
cd ~/EEG_SEIZURE_DETECTION
source .venv/bin/activate
streamlit run src/streamlit/app.py
```

Current scope:

- upload one EEG window
- validate shape against a selected model profile
- adapt channels and sample length with pad/trim
- preview the signal
- inspect saved artifacts under `outputs/models`, `outputs/results`, and `outputs/logs`
- verify that the shared dataloader indices are available

Not wired yet:

- checkpoint loading
- model inference
- EDF parsing
