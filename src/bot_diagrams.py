from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from yaml_utils import get


class DiagramBuilder:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.out_dir = Path(get(cfg, "outputs.diagrams_root", "results/preprocess/diagrams"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def preprocessing_diagram(self) -> str:
        return """flowchart TD
  A[BIDS EEG Input] --> B[BIDSLoader.load_raw()]
  B --> C[EEGPreprocessor (optional methods)]
  C --> C1[ensure_loaded()]
  C --> C2[resample() (optional)]
  C --> C3[rereference() (optional)]
  C --> C4[filtering (notch/bandpass) + wavelet (optional)]
  C --> D[TimeDomainModule: QC -> mark bads -> interpolate (optional)]
  D --> E[BIDSCleanedWriter.write_cleaned_raw()]
  E --> F[Cleaned BIDS Output (.fif)]
"""

    def eda_diagram(self) -> str:
        return """flowchart TD
  A[Raw BEFORE + Raw AFTER] --> B[EDAEngine]
  B --> T[TimeDomainModule]
  T --> T1[QC JSON]
  T --> T2[Epoching + drop_bad]
  T --> T3[Epoch stats + epoch plots]
  B --> F[FrequencyDomainAnalyzer]
  F --> F1[PSD Welch/Multitaper]
  F --> F2[Bandpower]
  B --> TF[TimeFrequencyAnalyzer]
  TF --> TF1[STFT spectrogram (explicit FFT)]
  TF --> TF2[Morlet TFR]
  B --> O[ArtifactWriter: CSV + plots]
"""

    def save_all(self) -> dict[str, str]:
        p1 = self.out_dir / "preprocessing.mmd"
        p2 = self.out_dir / "eda.mmd"
        p1.write_text(self.preprocessing_diagram())
        p2.write_text(self.eda_diagram())
        return {"preprocessing": str(p1), "eda": str(p2)}
