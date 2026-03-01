from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from eeg_pipeline.core.yaml_utils import get


def _true(cfg: Dict[str, Any], path: str) -> bool:
    """STRICT: True only if cfg[path] exists and is truthy; missing => False."""
    return bool(get(cfg, path, False))


def _clean_label(s: str) -> str:
    """Mermaid-safe labels."""
    bad = ["(", ")", ".", "/", "+", "->", ":", ",", "[", "]"]
    out = s
    for b in bad:
        out = out.replace(b, " ")
    return " ".join(out.split())


class DiagramBuilder:
    """
    Writes ONLY TWO mermaid diagrams:
      1) eda.mmd      -> EDA-focused tree (time-domain + frequency-domain + time-frequency + outputs)
      2) modules.mmd  -> Hierarchical module tree including PREPROCESS + time/freq/timefreq driven by config flags

    It DOES NOT write pipeline.mmd or preprocessing.mmd.
    Only nodes explicitly enabled:true in config are included (strict).
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.out_dir = Path(get(cfg, "outputs.diagrams_root", "results/preprocess/diagrams"))
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Modules diagram (includes PREPROCESS + time/freq/timefreq + optional outputs)
    # -----------------------
    def modules_diagram(self) -> str:
        cfg = self.cfg
        run_pre = _true(cfg, "run.preprocess")
        run_eda = _true(cfg, "run.eda")

        lines: List[str] = ["flowchart TD"]
        lines.append("  A[EEG window] --> ROOT[Pipeline modules]")

        # -------- Preprocessing --------
        filt_on = run_pre and _true(cfg, "preprocess.filtering.enabled")
        wav_on = run_pre and _true(cfg, "preprocess.wavelet_denoise.enabled")
        reref_on = run_pre and _true(cfg, "analysis.time_domain.reref.enabled")

        pre_on = run_pre and (filt_on or wav_on or reref_on)
        if pre_on:
            lines.append("  ROOT --> PRE[Preprocessing]")
            if reref_on:
                lines.append("  PRE --> PRE1[Re-reference]")
            # Filtering label with details from config
            # -------- Filtering (split into sub-boxes) --------
            if filt_on:
                lines.append("  PRE --> PRE2[Filtering]")

                l_freq = get(cfg, "preprocess.filtering.l_freq", None)
                h_freq = get(cfg, "preprocess.filtering.h_freq", None)
                notch = get(cfg, "preprocess.filtering.notch_freqs", None)
                method = str(get(cfg, "preprocess.filtering.method", "fir")).lower()
                order = get(cfg, "preprocess.filtering.iir_params.order", None)
                ftype = get(cfg, "preprocess.filtering.iir_params.ftype", None)

                # Bandpass/highpass/lowpass text
                if l_freq is None and h_freq is None:
                    bp_txt = "Bandpass: none"
                elif l_freq is None:
                    bp_txt = f"Lowpass: {h_freq} Hz"
                elif h_freq is None:
                    bp_txt = f"Highpass: {l_freq} Hz"
                else:
                    bp_txt = f"Bandpass: {l_freq}-{h_freq} Hz"

                # Notch text
                if notch:
                    notch_txt = "Notch: " + ", ".join(str(x) for x in notch) + " Hz"
                else:
                    notch_txt = "Notch: none"

                # Method text
                if method == "iir":
                    meth_txt = "Method: IIR"
                    if ftype:
                        meth_txt += f" ({ftype})"
                    if order is not None:
                        meth_txt += f", order {order}"
                else:
                    meth_txt = "Method: FIR"

                # Create child nodes under Filtering
                lines.append(f"  PRE2 --> PRE2a[{_clean_label(bp_txt)}]")
                lines.append(f"  PRE2 --> PRE2b[{_clean_label(notch_txt)}]")
                lines.append(f"  PRE2 --> PRE2c[{_clean_label(meth_txt)}]")
            if wav_on:
                lines.append("  PRE --> PRE3[Wavelet denoise]")

        # -------- Time-domain --------
        td_on = (
            _true(cfg, "analysis.time_domain.qc.enabled")
            or _true(cfg, "analysis.time_domain.bad_channels.enabled")
            or _true(cfg, "analysis.time_domain.epoching.enabled")
            or _true(cfg, "analysis.time_domain.artifact_rejection.enabled")
            or _true(cfg, "analysis.time_domain.ica.enabled")
        )
        if td_on:
            lines.append("  ROOT --> TD[Time-domain]")
            if _true(cfg, "analysis.time_domain.qc.enabled"):
                lines.append("  TD --> TD1[QC]")
            if _true(cfg, "analysis.time_domain.bad_channels.enabled"):
                lines.append("  TD --> TD2[Bad channels]")
                if _true(cfg, "analysis.time_domain.bad_channels.use_qc_rules"):
                    lines.append("  TD2 --> TD2a[Mark bads from QC]")
                if _true(cfg, "analysis.time_domain.bad_channels.interpolate"):
                    lines.append("  TD2 --> TD2b[Interpolate bads]")
            if _true(cfg, "analysis.time_domain.epoching.enabled"):
                lines.append("  TD --> TD3[Epoching]")
                if _true(cfg, "analysis.time_domain.artifact_rejection.enabled"):
                    lines.append("  TD3 --> TD3a[Artifact rejection]")
            if _true(cfg, "analysis.time_domain.ica.enabled"):
                lines.append("  TD --> TD4[ICA]")

        # -------- Frequency-domain --------
        fd_on = (
            _true(cfg, "analysis.frequency_domain.psd.enabled")
            or _true(cfg, "analysis.frequency_domain.bandpower.enabled")
            or _true(cfg, "analysis.frequency_domain.spectrogram.enabled")
            or _true(cfg, "analysis.frequency_domain.fft.enabled")
        )
        if fd_on:
            lines.append("  ROOT --> FD[Frequency-domain]")
            if _true(cfg, "analysis.frequency_domain.psd.enabled"):
                lines.append("  FD --> FD1[PSD]")
            if _true(cfg, "analysis.frequency_domain.bandpower.enabled"):
                lines.append("  FD --> FD2[Bandpower]")
            if _true(cfg, "analysis.frequency_domain.spectrogram.enabled"):
                lines.append("  FD --> FD3[Spectrogram]")
            if _true(cfg, "analysis.frequency_domain.fft.enabled"):
                lines.append("  FD --> FD4[FFT]")  # ✅ FFT included when enabled

        # -------- Time-frequency --------
        tf_root = _true(cfg, "analysis.time_frequency.enabled")
        if tf_root:
            lines.append("  ROOT --> TF[Time-frequency]")
            if _true(cfg, "analysis.time_frequency.stft.enabled"):
                lines.append("  TF --> TF1[STFT]")
            if _true(cfg, "analysis.time_frequency.morlet_tfr.enabled"):
                lines.append("  TF --> TF2[Morlet TFR]")

        # -------- Outputs (optional) --------
        if run_eda and (_true(cfg, "eda.save_csv") or _true(cfg, "eda.save_plots")):
            lines.append("  ROOT --> OUT[Outputs]")
            if _true(cfg, "eda.save_csv"):
                lines.append("  OUT --> OUT1[CSV artifacts]")
            if _true(cfg, "eda.save_plots"):
                lines.append("  OUT --> OUT2[Plot artifacts]")

        return "\n".join(lines) + "\n"

    # -----------------------
    # Save ONLY these two
    # -----------------------
    def save_all(self) -> dict[str, str]:
        p = self.out_dir / "modules.mmd"
        p.write_text(self.modules_diagram(), encoding="utf-8")
        return {"modules": str(p)}