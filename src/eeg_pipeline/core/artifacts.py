from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence, Optional

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

    def save_raw_plot(
        self,
        rel_path: str,
        raw: mne.io.BaseRaw,
        *,
        seconds: float,
        max_channels: int,
        title: str,
    ) -> str:
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)

        dur = min(float(seconds), raw.times[-1] if raw.n_times else float(seconds))
        browser = raw.plot(
            duration=dur,
            n_channels=min(max_channels, len(raw.ch_names)),
            show=False,
            title=title,
        )

        # MNE returns a Browser-like object; get the underlying figure safely
        fig = None
        if hasattr(browser, "figure"):
            fig = browser.figure
        elif hasattr(browser, "fig"):
            fig = browser.fig
        elif hasattr(browser, "_fig"):
            fig = browser._fig

        if fig is None:
            raise RuntimeError("Could not access matplotlib figure from raw.plot() return object.")

        fig.savefig(p, dpi=150, bbox_inches="tight")

        # close browser/fig safely
        try:
            browser.close()
        except Exception:
            pass
        plt.close(fig)
        return str(p)

    def save_line_plot(
        self,
        rel_path: str,
        x: np.ndarray,
        y: np.ndarray,
        *,
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> str:
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

    def save_spectrogram(
        self,
        rel_path: str,
        freqs: np.ndarray,
        times: np.ndarray,
        S: np.ndarray,
        *,
        title: str,
        cbar_label: str = "Magnitude",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: Optional[str] = None,
    ) -> str:
        """
        Save a time-frequency image.

        Parameters
        ----------
        freqs : (n_freq,) array
        times : (n_time,) array
        S     : (n_freq, n_time) array
        cbar_label : label shown on colorbar (e.g., 'Magnitude (µV)' or 'Power (µV²)')
        vmin/vmax  : optional fixed scale across plots
        cmap       : optional matplotlib cmap name (leave None for default)
        """
        p = self.root / rel_path
        p.parent.mkdir(parents=True, exist_ok=True)

        freqs = np.asarray(freqs, dtype=float)
        times = np.asarray(times, dtype=float)
        S = np.asarray(S)

        # Safe extents even if only 1 sample exists
        t0 = float(times[0]) if times.size else 0.0
        t1 = float(times[-1]) if times.size else 1.0
        f0 = float(freqs[0]) if freqs.size else 0.0
        f1 = float(freqs[-1]) if freqs.size else 1.0

        if times.size == 1:
            t1 = t0 + 1.0
        if freqs.size == 1:
            f1 = f0 + 1.0

        fig = plt.figure()
        im = plt.imshow(
            S,
            aspect="auto",
            origin="lower",
            extent=[t0, t1, f0, f1],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        cb = plt.colorbar(im)
        cb.set_label(cbar_label)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(title)
        plt.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        return str(p)
