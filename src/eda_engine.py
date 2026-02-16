from __future__ import annotations

from typing import Any, Dict, Optional
from fft_analysis import FFTAnalyzer
import mne
import matplotlib.pyplot as plt

from yaml_utils import get
from artifacts import ArtifactWriter
from time_domain import TimeDomainModule
from freq_analysis import FrequencyDomainAnalyzer
from timefreq_analysis import TimeFrequencyAnalyzer


class EDAEngine:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.writer = ArtifactWriter(get(cfg, "outputs.eda_root", "results/preprocess/eda"))

        self.td = TimeDomainModule(cfg)
        self.fd = FrequencyDomainAnalyzer(cfg)
        self.tf = TimeFrequencyAnalyzer(cfg)
        self.fft = FFTAnalyzer(cfg)

    def run(
        self,
        raw_after: mne.io.BaseRaw,
        recording_id: str,
        *,
        raw_before: Optional[mne.io.BaseRaw] = None,
    ) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {
            "qc_json": None,
            "psd_csv": None,
            "psd_plot": None,
            "psd_per_channel_plot": None,
            "bandpower_csv": None,
            "stft_csv": None,
            "stft_plot": None,
            "raw_plot": None,
            "raw_vs_filtered": None,
            "epochs_plot": None,
            "epoch_stats_json": None,
            "tfr_plot": None,
        }

        save_csv = bool(get(self.cfg, "eda.save_csv", True))
        save_plots = bool(get(self.cfg, "eda.save_plots", True))
        sec_plot = float(get(self.cfg, "eda.seconds_to_plot", 10.0))

        # Plot ALL channels (your DataLoader will provide 16 refined channels)
        n_ch = len(raw_after.ch_names)

        # --- Raw plot (after)
        if save_plots:
            out["raw_plot"] = self.writer.save_raw_plot(
                f"{recording_id}/raw_after.png",
                raw_after,
                seconds=sec_plot,
                max_channels=n_ch,  # <- no cap
                title=f"{recording_id} raw AFTER",
            )

        # --- Raw vs filtered
        if bool(get(self.cfg, "eda.plot_raw_vs_filtered", False)) and (raw_before is not None) and save_plots:
            p1 = self.writer.save_raw_plot(
                f"{recording_id}/raw_before.png",
                raw_before,
                seconds=sec_plot,
                max_channels=len(raw_before.ch_names),  # <- no cap
                title=f"{recording_id} raw BEFORE",
            )
            p2 = out["raw_plot"]
            out["raw_vs_filtered"] = f"{p1} | {p2}"

        # --- QC JSON
        qc_enabled = bool(get(self.cfg, "analysis.time_domain.qc.enabled", True))
        qc = self.td.qc(raw_after) if qc_enabled else {}
        if qc_enabled and save_csv:
            out["qc_json"] = self.writer.write_json(f"{recording_id}/qc.json", qc)

        # --- Explicit FFT (single-window) optional
        fft_res = self.fft.compute(raw_after)
        if fft_res is not None:
            if bool(get(self.cfg, "analysis.frequency_domain.fft.save_csv", True)) and save_csv:
                rows = [(float(f), float(a), float(p)) for f, a, p in zip(fft_res.freqs, fft_res.amp, fft_res.power)]
                out["fft_csv"] = self.writer.write_csv_rows(
                    f"{recording_id}/fft_{fft_res.channel}_t{fft_res.tmin:.1f}-{fft_res.tmax:.1f}.csv",
                    header=["freq_hz", "amplitude", "power"],
                    rows=rows,
                )

            if bool(get(self.cfg, "analysis.frequency_domain.fft.save_plot", True)) and save_plots:
                out["fft_plot"] = self.writer.save_line_plot(
                    f"{recording_id}/fft_{fft_res.channel}_power.png",
                    fft_res.freqs,
                    fft_res.power,
                    title=f"{recording_id} FFT Power ({fft_res.channel}) [{fft_res.tmin:.1f}-{fft_res.tmax:.1f}s]",
                    xlabel="Frequency (Hz)",
                    ylabel="Power (a.u.)",
                )


        # --- PSD + bandpower
        psd_enabled = bool(get(self.cfg, "analysis.frequency_domain.psd.enabled", True))
        bp_enabled = bool(get(self.cfg, "analysis.frequency_domain.bandpower.enabled", True))
        psd_res = self.fd.psd(raw_after) if psd_enabled else None

        if psd_res is not None and save_csv:
            rows = [(float(f), float(p)) for f, p in zip(psd_res.freqs, psd_res.psd_mean)]
            out["psd_csv"] = self.writer.write_csv_rows(
                f"{recording_id}/psd_mean.csv",
                header=["freq_hz", "psd_mean"],
                rows=rows,
            )

        if psd_res is not None and bool(get(self.cfg, "eda.plot_psd", False)) and save_plots:
            out["psd_plot"] = self.writer.save_line_plot(
                f"{recording_id}/psd_mean.png",
                psd_res.freqs,
                psd_res.psd_mean,
                title=f"{recording_id} PSD mean",
                xlabel="Frequency (Hz)",
                ylabel="Power (V^2/Hz)",
            )

        if psd_res is not None and bool(get(self.cfg, "eda.plot_psd_per_channel", False)) and save_plots:
            # If you truly want ALL 16 channels, set n_show = psd_res.psd_per_channel.shape[0]
            n_show = psd_res.psd_per_channel.shape[0]
            fig = plt.figure()
            for i in range(n_show):
                plt.plot(psd_res.freqs, psd_res.psd_per_channel[i], label=psd_res.ch_names[i])
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power (V^2/Hz)")
            plt.title(f"{recording_id} PSD per channel (all {n_show})")
            # Legend can get crowded; keep it but small
            plt.legend(fontsize=7, ncol=2, loc="best")
            plt.tight_layout()

            p = (self.writer.root / f"{recording_id}/psd_per_channel.png")
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=150)
            plt.close(fig)
            out["psd_per_channel_plot"] = str(p)

        if (psd_res is not None) and bp_enabled and save_csv:
            bp = self.fd.bandpowers(psd_res)
            rows = [(k, float(v)) for k, v in bp.items()]
            out["bandpower_csv"] = self.writer.write_csv_rows(
                f"{recording_id}/bandpower.csv",
                header=["band", "power"],
                rows=rows,
            )

        # --- Epochs + rejection + plots + epoch stats
        epochs = self.td.make_epochs(raw_after)
        if epochs is not None:
            epochs2 = self.td.reject_artifact_epochs(epochs)

            if save_csv:
                stats = self.td.epoch_stats(epochs, epochs2)
                out["epoch_stats_json"] = self.writer.write_json(
                    f"{recording_id}/epoch_stats.json",
                    {
                        "n_epochs_before": stats.n_epochs_before,
                        "n_epochs_after": stats.n_epochs_after,
                        "epoch_len_sec": stats.epoch_len_sec,
                        "mean_var_uV2": stats.mean_var_uV2,
                        "mean_kurtosis": stats.mean_kurtosis,
                    },
                )

            if bool(get(self.cfg, "eda.plot_epochs", False)) and save_plots and len(epochs2) > 0:
                figs = epochs2.plot(
                    n_epochs=min(5, len(epochs2)),
                    n_channels=len(epochs2.ch_names),
                    show=False,
                )

                # MNE can return a Figure OR a list of Figures depending on version/settings
                if isinstance(figs, list):
                    fig0 = figs[0] if len(figs) > 0 else None
                else:
                    fig0 = figs

                if fig0 is not None:
                    p = (self.writer.root / f"{recording_id}/epochs.png")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    fig0.savefig(p, dpi=150, bbox_inches="tight")
                    out["epochs_plot"] = str(p)

                    # Close all opened figs to avoid memory leak
                    try:
                        if isinstance(figs, list):
                            for f in figs:
                                plt.close(f)
                        else:
                            plt.close(figs)
                    except Exception:
                        pass

        # --- STFT spectrogram
        tf_enabled = bool(get(self.cfg, "analysis.time_frequency.enabled", False))
        stft_enabled = tf_enabled and bool(get(self.cfg, "analysis.time_frequency.stft.enabled", False))
        if stft_enabled:
            pick_ch = get(self.cfg, "analysis.time_frequency.stft.pick_channel", None)
            freqs, times, S = self.tf.stft_spectrogram_mean(raw_after, pick_channel=pick_ch)

            if save_csv:
                mean_over_freq = S.mean(axis=0)
                mean_over_time = S.mean(axis=1)
                out["stft_csv"] = self.writer.write_csv_rows(
                    f"{recording_id}/stft_summary.csv",
                    header=["type", "x", "mean_mag"],
                    rows=(
                        [("time_sec", float(t), float(m)) for t, m in zip(times, mean_over_freq)]
                        + [("freq_hz", float(f), float(m)) for f, m in zip(freqs, mean_over_time)]
                    ),
                )

            if bool(get(self.cfg, "eda.plot_stft", False)) and save_plots:
                out["stft_plot"] = self.writer.save_spectrogram(
                    f"{recording_id}/stft.png",
                    freqs=freqs,
                    times=times,
                    S=S,
                    title=f"{recording_id} STFT spectrogram (mean)",
                )

        # --- Morlet TFR plot (optional; best on epochs)
        morlet_enabled = tf_enabled and bool(get(self.cfg, "analysis.time_frequency.morlet_tfr.enabled", False))
        if morlet_enabled and bool(get(self.cfg, "eda.plot_tfr", False)) and save_plots:
            if epochs is None:
                epochs = self.td.make_epochs(raw_after)
            if epochs is not None and len(epochs) > 0:
                if len(epochs.ch_names) > 0 and epochs.get_data().shape[-1] >= 64:
                    power = self.tf.morlet_tfr(epochs)
                    figs = power.plot(show=False)

                    fig0 = figs[0] if isinstance(figs, list) and len(figs) > 0 else figs
                    if fig0 is not None:
                        p = (self.writer.root / f"{recording_id}/tfr_morlet.png")
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

        return out
