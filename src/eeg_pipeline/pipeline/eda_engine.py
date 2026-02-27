from __future__ import annotations

from typing import Any, Dict, Optional, List

import numpy as np
import mne
import matplotlib.pyplot as plt

from eeg_pipeline.core.yaml_utils import get
from eeg_pipeline.core.artifacts import ArtifactWriter
from eeg_pipeline.analysis.time_domain import TimeDomainModule, Interval
from eeg_pipeline.analysis.freq_domain import FrequencyDomainAnalyzer


V_TO_uV = 1e6
V2_TO_uV2 = 1e12


def _to_uV(x):
    return np.asarray(x) * V_TO_uV


def _to_uV2(x):
    return np.asarray(x) * V2_TO_uV2


class EDAEngine:
    """
    EDA layer outputs: plots + CSVs for visual inspection and numerical inspection.

    Recording-level EDA:
      - raw_after.png (+ raw_before.png optional)
      - qc.json (optional)
      - psd_mean_uV2_per_hz.csv
      - psd_per_channel_uV2_per_hz.png (optional)
      - psd_per_channel_uV2_per_hz.csv (optional)
      - bandpower_uV2.csv
      - epoch_stats.csv (variance, kurtosis) (if epoching enabled)
      - seizure_vs_nonseizure.csv (recording-level comparisons if seizure intervals exist)
      - stft.png / tfr_morlet_uV2.png / psd_spectrogram_*.png (optional)

    Window-level EDA (optional):
      - raw_after.png (+ raw_before.png optional)
      - qc.json optional
      - psd_mean_uV2_per_hz.csv optional
      - bandpower_uV2.csv optional
    """

    def __init__(self, cfg: Dict[str, Any], td: Optional[TimeDomainModule] = None, fd: Optional[FrequencyDomainAnalyzer] = None):
        self.cfg = cfg
        self.writer = ArtifactWriter(get(cfg, "outputs.eda_root", "results/preprocess/eda"))
        self.td = td or TimeDomainModule(cfg)
        self.fd = fd or FrequencyDomainAnalyzer(cfg)

    # -------------------------
    # Recording-level EDA
    # -------------------------
    def run_recording(
        self,
        *,
        recording_id: str,
        raw_after: mne.io.BaseRaw,
        raw_before: Optional[mne.io.BaseRaw] = None,
        seizure_intervals: Optional[List[Interval]] = None,
    ) -> Dict[str, Optional[str]]:
        base = f"{recording_id}/recording"
        return self._run_common(
            out_prefix=base,
            raw_after=raw_after,
            raw_before=raw_before,
            seizure_intervals=seizure_intervals,
            label=None,
            include_epoch_stats=True,
            include_seizure_compare=True,
        )

    # -------------------------
    # Window-level EDA (lightweight)
    # -------------------------
    def run_window(
            self,
            *,
            window_id: str,
            raw_after: mne.io.BaseRaw,
            raw_before: Optional[mne.io.BaseRaw] = None,
            label: Optional[int] = None,
            qc: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Optional[str]]:
        base = f"{window_id}/window"
        return self._run_common(
            out_prefix=base,
            raw_after=raw_after,
            raw_before=raw_before,
            seizure_intervals=None,
            label=label,
            qc=qc,
            include_epoch_stats=bool(get(self.cfg, "eda.window_level.epoch_stats", False)),
            include_seizure_compare=False,
        )

    # -------------------------
    # Shared implementation
    # -------------------------
    def _run_common(
            self,
            *,
            out_prefix: str,
            raw_after: mne.io.BaseRaw,
            raw_before: Optional[mne.io.BaseRaw],
            seizure_intervals: Optional[List[Interval]],
            label: Optional[int],
            qc: Optional[Dict[str, Any]] = None,
            include_epoch_stats: bool,
            include_seizure_compare: bool,
    ) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {
            "qc_json": None,
            "raw_after_plot": None,
            "raw_before_plot": None,
            "psd_csv": None,
            "psd_per_channel_plot": None,
            "psd_per_channel_csv": None,
            "bandpower_uV2_csv": None,
            "epoch_stats_csv": None,
            "seizure_vs_nonseizure_csv": None,
            "stft_plot": None,
            "tfr_plot": None,
            "psd_spectrogram_plot": None,
        }

        save_csv = bool(get(self.cfg, "eda.save_csv", True))
        save_plots = bool(get(self.cfg, "eda.save_plots", True))
        sec_plot = float(get(self.cfg, "eda.seconds_to_plot", 10.0))

        # ------------- Raw plots -------------
        # ------------- Raw plots -------------
        if save_plots:
            max_ch_cfg = get(self.cfg, "eda.max_channels_plot", None)
            max_ch = len(raw_after.ch_names) if max_ch_cfg is None else min(len(raw_after.ch_names), int(max_ch_cfg))

            out["raw_after_plot"] = self.writer.save_raw_plot(
                f"{out_prefix}/raw_after.png",
                raw_after,
                seconds=sec_plot,
                max_channels=max_ch,
                title=f"{out_prefix} raw AFTER",
            )

            if raw_before is not None:
                max_chb = len(raw_before.ch_names) if max_ch_cfg is None else min(len(raw_before.ch_names),
                                                                                  int(max_ch_cfg))
                out["raw_before_plot"] = self.writer.save_raw_plot(
                    f"{out_prefix}/raw_before.png",
                    raw_before,
                    seconds=sec_plot,
                    max_channels=max_chb,
                    title=f"{out_prefix} raw BEFORE",
                )

        # ------------- QC JSON -------------
        qc_enabled = bool(get(self.cfg, "analysis.time_domain.qc.enabled", True))
        qc_final = qc if qc is not None else (self.td.qc(raw_after) if qc_enabled else {})

        if qc_enabled and save_csv:
            out["qc_json"] = self.writer.write_json(f"{out_prefix}/qc.json", qc_final)

        # ------------- PSD + Bandpower -------------
        psd_enabled = bool(get(self.cfg, "analysis.frequency_domain.psd.enabled", True))
        band_enabled = bool(get(self.cfg, "analysis.frequency_domain.bandpower.enabled", True))

        psd_res = self.fd.psd(raw_after) if psd_enabled else None
        if psd_res is not None:
            # mean PSD CSV (µV²/Hz)
            if save_csv:
                psd_mean_uV2_per_hz = _to_uV2(psd_res.psd_mean)
                rows = [(float(f), float(p)) for f, p in zip(psd_res.freqs, psd_mean_uV2_per_hz)]
                out["psd_csv"] = self.writer.write_csv_rows(
                    f"{out_prefix}/psd_mean_uV2_per_hz.csv",
                    header=["freq_hz", "psd_mean_uV2_per_hz"],
                    rows=rows,
                )

            # PSD per-channel plot
            if save_plots and bool(get(self.cfg, "eda.plot_psd_per_channel", True)):
                psd_per_ch_uV2 = _to_uV2(psd_res.psd_per_channel)  # (n_ch, n_freq)

                fig = plt.figure()
                for i in range(psd_per_ch_uV2.shape[0]):
                    plt.plot(psd_res.freqs, psd_per_ch_uV2[i], label=psd_res.ch_names[i])  # ✅ label added

                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Power (µV²/Hz)")
                plt.title(f"{out_prefix} PSD per channel (µV²/Hz)")

                #legend (works, but can be huge)
                plt.legend(fontsize=7, ncol=2, loc="best")

                plt.tight_layout()

                p = (self.writer.root / f"{out_prefix}/psd_per_channel_uV2_per_hz.png")
                p.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(p, dpi=150)
                plt.close(fig)

                out["psd_per_channel_plot"] = str(p)

            # Optional PSD per-channel CSV (long format)
            # Optional PSD per-channel CSV (long format) — SAFE indexing
            if save_csv and bool(get(self.cfg, "eda.save_psd_per_channel_csv", False)):
                psd_per_ch_uV2 = _to_uV2(psd_res.psd_per_channel)  # (n_ch_psd, n_freq)
                n_ch_psd, n_f = psd_per_ch_uV2.shape

                # Safe channel name list
                ch_names = list(psd_res.ch_names) if isinstance(psd_res.ch_names, list) else []
                if len(ch_names) < n_ch_psd:
                    # pad names if missing
                    ch_names = ch_names + [f"ch_{i}" for i in range(len(ch_names), n_ch_psd)]

                rows = []
                for ci in range(n_ch_psd):
                    ch = ch_names[ci]  # guaranteed exists
                    for fi in range(len(psd_res.freqs)):
                        rows.append((ch, float(psd_res.freqs[fi]), float(psd_per_ch_uV2[ci, fi])))

                out["psd_per_channel_csv"] = self.writer.write_csv_rows(
                    f"{out_prefix}/psd_per_channel_uV2_per_hz.csv",
                    header=["channel", "freq_hz", "psd_uV2_per_hz"],
                    rows=rows,
                )

            # Bandpower CSV (µV²)
            if band_enabled and save_csv:
                bp_V2 = self.fd.bandpowers(psd_res)
                bp_uV2 = {k: float(v * V2_TO_uV2) for k, v in bp_V2.items()}
                rows = [(k, v) for k, v in bp_uV2.items()]
                out["bandpower_uV2_csv"] = self.writer.write_csv_rows(
                    f"{out_prefix}/bandpower_uV2.csv",
                    header=["band", "power_uV2"],
                    rows=rows,
                )

            # PSD spectrogram plot (per window)
            # PSD Spectrogram (better scaling)
            # ---- PSD Spectrogram (robust, visible) ----
            spec_enabled = bool(get(self.cfg, "analysis.frequency_domain.spectrogram.enabled", False))
            if spec_enabled and save_plots:
                pick_ch = get(self.cfg, "analysis.frequency_domain.spectrogram.pick_channel", None)
                freqs, times, P_v2_per_hz = self.fd.psd_spectrogram(raw_after, pick_channel=pick_ch)

                # Old style scaling (simple)
                scale = str(get(self.cfg, "analysis.frequency_domain.spectrogram.scale", "db")).lower()
                if scale == "db":
                    eps = 1e-12
                    S = 10.0 * np.log10(P_v2_per_hz + eps)
                    cbar = "Power (dB)"
                else:
                    S = P_v2_per_hz
                    cbar = "Power (V²/Hz)"

                vmin = float(np.percentile(S, 5))
                vmax = float(np.percentile(S, 95))

                out["psd_spectrogram_plot"] = self.writer.save_spectrogram(
                    f"{out_prefix}/psd_spectrogram_{scale}.png",
                    freqs=freqs,
                    times=times,
                    S=S,
                    title=f"{out_prefix} PSD Spectrogram ({get(self.cfg, 'analysis.frequency_domain.psd.method', 'welch')})",
                    cbar_label=cbar,
                    vmin=vmin,
                    vmax=vmax,
                )
            # Morlet spectrogram plot (per window)
            morlet_enabled = bool(get(self.cfg, "analysis.frequency_domain.morlet.enabled", False))
            if morlet_enabled and save_plots:
                pick_ch = get(self.cfg, "analysis.frequency_domain.morlet.pick_channel", None)
                mres = self.fd.morlet_spectrogram(raw_after, pick_channel=pick_ch)
                if mres is not None:
                    scale = str(get(self.cfg, "analysis.frequency_domain.morlet.scale", "db")).lower()
                    P_uV2 = mres.P * 1e12  # V^2 -> µV^2
                    if scale == "db":
                        eps = 1e-20
                        S = 10.0 * np.log10(P_uV2 + eps)
                        cbar = "Power (dB µV²)"
                    else:
                        S = P_uV2
                        cbar = "Power (µV²)"

                    vmin = float(np.percentile(S, 5))
                    vmax = float(np.percentile(S, 95))

                    out["morlet_spectrogram_plot"] = self.writer.save_spectrogram(
                        f"{out_prefix}/morlet_spectrogram_{scale}.png",
                        freqs=mres.freqs,
                        times=mres.times,
                        S=S,
                        title=f"{out_prefix} Morlet Spectrogram",
                        cbar_label=cbar,
                        vmin=vmin,
                        vmax=vmax,
                    )

            # FFT plot (per window)
            fft_res = self.fd.fft(raw_after)
            if fft_res is not None and save_plots:
                amp_uV = fft_res.amp * 1e6
                out["fft_plot"] = self.writer.save_line_plot(
                    f"{out_prefix}/fft_{fft_res.channel}_t{fft_res.tmin:.1f}-{fft_res.tmax:.1f}.png",
                    x=fft_res.freqs,
                    y=amp_uV,
                    title=f"{out_prefix} FFT amplitude ({fft_res.channel})",
                    xlabel="Frequency (Hz)",
                    ylabel="Amplitude (µV)",
                )

            if fft_res is not None and save_csv:
                amp_uV = fft_res.amp * 1e6
                pow_uV2 = fft_res.power * 1e12
                rows = [(float(f), float(a), float(p)) for f, a, p in zip(fft_res.freqs, amp_uV, pow_uV2)]
                out["fft_csv"] = self.writer.write_csv_rows(
                    f"{out_prefix}/fft_{fft_res.channel}_t{fft_res.tmin:.1f}-{fft_res.tmax:.1f}.csv",
                    header=["freq_hz", "amplitude_uV", "power_uV2"],
                    rows=rows,
                )

        # ------------- Epoch stats (variance/kurtosis) -------------
        if include_epoch_stats and bool(get(self.cfg, "analysis.time_domain.epoching.enabled", False)) and save_csv:
            epochs_before = self.td.make_epochs(raw_after)
            if epochs_before is not None and len(epochs_before) > 0:
                epochs_after = epochs_before
                if bool(get(self.cfg, "analysis.time_domain.artifact_rejection.enabled", False)):
                    epochs_after = self.td.reject_artifact_epochs(epochs_before)
                stats = self.td.epoch_stats(epochs_before, epochs_after)
                out["epoch_stats_csv"] = self.writer.write_csv_rows(
                    f"{out_prefix}/epoch_stats.csv",
                    header=["n_epochs_before", "n_epochs_after", "epoch_len_sec", "mean_var_uV2", "mean_kurtosis"],
                    rows=[[
                        stats.n_epochs_before,
                        stats.n_epochs_after,
                        stats.epoch_len_sec,
                        stats.mean_var_uV2,
                        stats.mean_kurtosis,
                    ]],
                )

        # ------------- Seizure vs non-seizure comparisons (recording-level) -------------
        if include_seizure_compare and save_csv and seizure_intervals:
            # sample matched non-seizure intervals and compute simple summary stats
            seed = int(get(self.cfg, "analysis.comparisons.seed", 42))
            non = self.td.sample_nonseizure_intervals(raw=raw_after, seizure_intervals=seizure_intervals, seed=seed)

            def _summarize(intervals: List[Interval], tag: str) -> Dict[str, float]:
                if not intervals:
                    return {"label": tag, "n_segments": 0, "mean_abs_uv": np.nan, "rms_uv": np.nan, "ptp_uv": np.nan}
                vals_abs = []
                vals_rms = []
                vals_ptp = []
                sf = float(raw_after.info["sfreq"])
                for itv in intervals:
                    s0 = int(max(0, round(itv.start * sf)))
                    s1 = int(min(raw_after.n_times, round(itv.end * sf)))
                    if s1 <= s0:
                        continue
                    seg = raw_after.get_data(start=s0, stop=s1)  # (n_ch, n_times) volts
                    seg_uv = seg * 1e6
                    vals_abs.append(float(np.mean(np.abs(seg_uv))))
                    vals_rms.append(float(np.sqrt(np.mean(seg_uv ** 2))))
                    vals_ptp.append(float(np.mean(np.ptp(seg_uv, axis=1))))
                return {
                    "label": tag,
                    "n_segments": int(len(vals_abs)),
                    "mean_abs_uv": float(np.nanmean(vals_abs)) if vals_abs else np.nan,
                    "rms_uv": float(np.nanmean(vals_rms)) if vals_rms else np.nan,
                    "ptp_uv": float(np.nanmean(vals_ptp)) if vals_ptp else np.nan,
                }

            seiz_sum = _summarize(seizure_intervals, "seizure")
            non_sum = _summarize(non, "nonseizure")

            out["seizure_vs_nonseizure_csv"] = self.writer.write_csv_rows(
                f"{out_prefix}/seizure_vs_nonseizure.csv",
                header=["label", "n_segments", "mean_abs_uv", "rms_uv", "ptp_uv"],
                rows=[
                    [seiz_sum["label"], seiz_sum["n_segments"], seiz_sum["mean_abs_uv"], seiz_sum["rms_uv"],
                     seiz_sum["ptp_uv"]],
                    [non_sum["label"], non_sum["n_segments"], non_sum["mean_abs_uv"], non_sum["rms_uv"],
                     non_sum["ptp_uv"]],
                ],
            )

        # ------------- Optional STFT plot -------------
        tf_enabled = bool(get(self.cfg, "analysis.time_frequency.enabled", False))
        stft_enabled = tf_enabled and bool(get(self.cfg, "analysis.time_frequency.stft.enabled", False))
        if stft_enabled and save_plots and bool(get(self.cfg, "eda.plot_stft", False)):
            pick_ch = get(self.cfg, "analysis.time_frequency.stft.pick_channel", None)
            freqs, times, S_V = self.fd.stft_spectrogram_mean(raw_after, pick_channel=pick_ch)
            S_uV = _to_uV(S_V)
            out["stft_plot"] = self.writer.save_spectrogram(
                f"{out_prefix}/stft.png",
                freqs=freqs,
                times=times,
                S=S_uV,
                title=f"{out_prefix} STFT spectrogram (mean, µV)",
                cbar_label="Magnitude (µV)",
            )

        # ------------- Optional Morlet TFR plot -------------
        morlet_enabled = tf_enabled and bool(get(self.cfg, "analysis.time_frequency.morlet_tfr.enabled", False))
        if morlet_enabled and save_plots and bool(get(self.cfg, "eda.plot_tfr", False)):
            epochs = self.td.make_epochs(raw_after)
            if epochs is not None and len(epochs) > 0:
                power = self.fd.morlet_tfr(epochs)
                try:
                    power.data = power.data * V2_TO_uV2
                except Exception:
                    pass
                figs = power.plot(show=False)
                fig0 = figs[0] if isinstance(figs, list) and len(figs) > 0 else figs
                p = (self.writer.root / f"{out_prefix}/tfr_morlet_uV2.png")
                p.parent.mkdir(parents=True, exist_ok=True)
                fig0.savefig(p, dpi=150, bbox_inches="tight")
                out["tfr_plot"] = str(p)
                try:
                    if isinstance(figs, list):
                        for f in figs:
                            plt.close(f)
                    else:
                        plt.close(figs)
                except Exception:
                    pass

        # ------------- Optional PSD spectrogram plot -------------

        return out