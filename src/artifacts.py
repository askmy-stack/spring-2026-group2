from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt
import mne


class ArtifactWriter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, rel_path: str, payload: Dict[str, Any]) -> str:
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2))
        return str(p)

    def write_csv_rows(self, rel_path: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [",".join(header)]
        for r in rows:
            lines.append(",".join(str(x) for x in r))
        p.write_text("\n".join(lines))
        return str(p)

    def save_raw_plot(self, rel_path: str, raw: mne.io.BaseRaw, *, seconds: float, max_channels: int, title: str) -> str:
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)

        dur = min(float(seconds), raw.times[-1] if raw.n_times else float(seconds))
        fig = raw.plot(duration=dur, n_channels=min(max_channels, len(raw.ch_names)), show=False, title=title)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(p)

    def save_line_plot(self, rel_path: str, x: np.ndarray, y: np.ndarray, *, title: str, xlabel: str, ylabel: str) -> str:
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure()
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return str(p)

    def save_spectrogram(self, rel_path: str, freqs: np.ndarray, times: np.ndarray, S: np.ndarray, *, title: str) -> str:
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure()
        plt.imshow(
            S,
            aspect="auto",
            origin="lower",
            extent=[float(times[0]), float(times[-1]), float(freqs[0]), float(freqs[-1])],
        )
        plt.colorbar(label="|STFT| (a.u.)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(title)
        plt.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return str(p)
