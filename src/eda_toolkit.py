import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.signal import welch, spectrogram
from scipy.stats import skew, kurtosis
import os

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({'figure.titlesize': 14, 'figure.figsize': (10, 6)})


class AdvancedEDA:
    def __init__(self, raw_mne, sfreq=256):
        self.raw = raw_mne
        self.sfreq = sfreq
        self.data = raw_mne.get_data() * 1e6  # Convert to uV
        self.ch_names = raw_mne.ch_names

        self.variance = np.var(self.data, axis=1)
        self.flat_data = self.data.flatten()

        print(f">>> EDA INITIALIZED: {self.data.shape[0]} Channels, {self.data.shape[1] / sfreq:.1f} Seconds <<<")

    def _print_inference(self, title, message):
        print(f"\n[STATISTICAL INFERENCE] {title}")
        print(f"  -> {message}")

    # --- PLOT 1: AMPLITUDE DISTRIBUTION ---
    def plot_1_amplitude_histogram(self):
        clean = self.flat_data[np.abs(self.flat_data) < 200]
        skewness = skew(clean)

        plt.figure()
        sns.histplot(clean, bins=100, color='#6c5ce7', kde=True, stat="density")
        plt.title(f"1. Amplitude Distribution (Skew: {skewness:.2f})")
        plt.xlabel("Voltage (µV)")
        plt.xlim(-100, 100)
        plt.show()

        self._print_inference("Signal Normality",
                              f"Skewness is {skewness:.2f}. Values near 0 imply normal brain activity. High skew (>1) implies artifacts or seizure spikes.")

    # --- PLOT 2: CHANNEL VARIANCE RANKING ---
    def plot_2_channel_variance(self):
        df = pd.DataFrame({'Channel': self.ch_names, 'Variance': self.variance})
        df = df.sort_values('Variance', ascending=False)
        max_ch = df.iloc[0]

        plt.figure()
        sns.barplot(data=df, x='Channel', y='Variance', palette='viridis', hue='Channel', legend=False)
        plt.title("2. Channel Variance Ranking")
        plt.xticks(rotation=45)
        plt.ylabel("Variance (µV²)")
        plt.show()

        self._print_inference("Focal Point",
                              f"Highest activity found in {max_ch['Channel']} ({max_ch['Variance']:.1f} µV²). This is the likely seizure focus or artifact source.")

    # --- PLOT 3: ARTIFACT SCATTER (SKEW vs KURTOSIS) ---
    def plot_3_artifact_scatter(self):
        skews = skew(self.data, axis=1)
        kurts = kurtosis(self.data, axis=1)

        plt.figure()
        sns.scatterplot(x=skews, y=kurts, s=100, hue=self.ch_names, palette='deep')
        plt.axhline(3, color='r', linestyle='--', label='Gaussian Normal')
        plt.title("3. Artifact Detection (Skew vs Kurtosis)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    # --- PLOT 4: GLOBAL PSD SPECTRUM ---
    def plot_4_psd_spectrum(self):
        freqs, psd = welch(self.data, fs=self.sfreq, nperseg=self.sfreq * 2)
        mean_psd = np.mean(psd, axis=0)

        peak_freq = freqs[np.argmax(mean_psd)]

        plt.figure()
        plt.semilogy(freqs, mean_psd, color='#2c3e50', linewidth=2)
        plt.axvline(peak_freq, color='r', linestyle='--', alpha=0.5, label=f'Peak: {peak_freq:.1f}Hz')

        bands = [(0.5, 4, 'Delta', '#e74c3c'), (4, 8, 'Theta', '#f1c40f'),
                 (8, 12, 'Alpha', '#2ecc71'), (12, 30, 'Beta', '#3498db')]
        for l, h, n, c in bands: plt.axvspan(l, h, color=c, alpha=0.1, label=n)

        plt.title("4. Global Power Spectral Density")
        plt.xlim(0, 40)
        plt.legend()
        plt.show()

        self._print_inference("Dominant Rhythm",
                              f"Peak Frequency is {peak_freq:.1f} Hz. (Delta <4Hz often indicates sleep or pathology; Alpha 8-12Hz is normal resting).")

    # --- PLOT 5: BAND POWER BOXPLOT ---
    def plot_5_band_powers(self):
        bands = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30)}
        freqs, psd = welch(self.data, fs=self.sfreq, nperseg=self.sfreq)
        freq_res = freqs[1] - freqs[0]

        band_powers = {}
        for band, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_powers[band] = np.sum(psd[:, idx], axis=1) * freq_res

        df = pd.DataFrame(band_powers)
        delta_alpha_ratio = np.mean(df['Delta']) / np.mean(df['Alpha'])

        plt.figure()
        sns.boxplot(data=df, palette='Set2')
        plt.title("5. Frequency Band Power Distribution")
        plt.yscale('log')
        plt.show()

        self._print_inference("Slowing Index",
                              f"Delta/Alpha Ratio is {delta_alpha_ratio:.2f}. Higher values (>1.5) indicate 'slowing' common in inter-ictal seizure states.")

    # --- PLOT 6: CORRELATION HEATMAP ---
    def plot_6_connectivity(self):
        corr = np.corrcoef(self.data[:, :int(self.sfreq * 60)])
        avg_corr = np.mean(np.abs(corr)) - (1 / corr.shape[0])  # Subtract diagonal roughly

        plt.figure(figsize=(8, 8))
        sns.heatmap(corr, xticklabels=self.ch_names, yticklabels=self.ch_names,
                    cmap='coolwarm', center=0, square=True, cbar_kws={"shrink": .8})
        plt.title("6. Spatial Connectivity (Correlation)")
        plt.show()

        self._print_inference("Synchronization",
                              f"Average connectivity is {avg_corr:.2f}. Seizures often cause hypersynchrony (values approaching 1.0 across focal areas).")

    # --- PLOT 7: SPATIAL VARIANCE (HEAD MAP PROXY) ---
    def plot_7_spatial_variance(self):
        regions = {'Frontal': [], 'Temporal': [], 'Parietal': [], 'Occipital': []}
        for i, name in enumerate(self.ch_names):
            if 'F' in name:
                regions['Frontal'].append(self.variance[i])
            elif 'T' in name:
                regions['Temporal'].append(self.variance[i])
            elif 'P' in name:
                regions['Parietal'].append(self.variance[i])
            elif 'O' in name:
                regions['Occipital'].append(self.variance[i])

        avg_vars = {k: np.mean(v) if v else 0 for k, v in regions.items()}
        max_region = max(avg_vars, key=avg_vars.get)

        plt.figure()
        plt.bar(avg_vars.keys(), avg_vars.values(), color=['#3498db', '#e74c3c', '#9b59b6', '#f1c40f'])
        plt.title("7. Regional Brain Energy")
        plt.ylabel("Avg Variance")
        plt.show()

        self._print_inference("Lobe Localization",
                              f"The {max_region} Lobe shows the highest energy. This is the primary suspect region.")

    # --- PLOT 8: SPECTROGRAM ---
    def plot_8_spectrogram(self):
        f, t, Sxx = spectrogram(self.data[0], fs=self.sfreq, nperseg=512)
        plt.figure(figsize=(12, 5))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma')
        plt.title(f"8. Spectrogram (Channel {self.ch_names[0]})")
        plt.ylabel("Freq (Hz)")
        plt.ylim(0, 40)
        plt.colorbar(label='dB')
        plt.show()

    # --- PLOT 9: LINE LENGTH TREND ---
    def plot_9_line_length(self):
        n_sec = self.data.shape[1] // self.sfreq
        ll_trend = [np.mean(np.sum(np.abs(np.diff(self.data[:, i * self.sfreq:(i + 1) * self.sfreq], axis=1)), axis=1))
                    for i in range(n_sec)]

        threshold = np.mean(ll_trend) + 3 * np.std(ll_trend)
        spikes = np.sum(np.array(ll_trend) > threshold)

        plt.figure(figsize=(12, 4))
        plt.plot(ll_trend, color='#e67e22', linewidth=1)
        plt.axhline(threshold, color='red', linestyle='--', label='Seizure Threshold')
        plt.title("9. Global Signal Complexity (Line Length)")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.show()

        self._print_inference("Event Detection",
                              f"Found {spikes} time windows exceeding the 3-sigma anomaly threshold. These are high-probability seizure candidates.")

    # --- PLOT 10: CIRCADIAN TREND ---
    def plot_10_circadian(self):
        mins = self.data.shape[1] // (self.sfreq * 60)
        if mins < 2: return  # Too short

        energy = [np.mean(self.data[:, i * 60 * self.sfreq:(i + 1) * 60 * self.sfreq] ** 2) for i in range(mins)]

        plt.figure()
        plt.plot(energy, marker='o', color='#8e44ad')
        plt.title("10. Circadian Trend (Energy/Min)")
        plt.xlabel("Minute")
        plt.show()

    def run_full_dashboard(self):
        print("\n>>> GENERATING 10-PLOT DASHBOARD IN PYCHARM... <<<")
        self.plot_1_amplitude_histogram()
        self.plot_2_channel_variance()
        self.plot_3_artifact_scatter()
        self.plot_4_psd_spectrum()
        self.plot_5_band_powers()
        self.plot_6_connectivity()
        self.plot_7_spatial_variance()
        self.plot_8_spectrogram()
        self.plot_9_line_length()
        self.plot_10_circadian()
        print("\n>>> DASHBOARD COMPLETE <<<")


if __name__ == "__main__":
    from dataset_pipeline import EEGJanitor

    # SETUP
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(base_dir, "seizure_system/data/raw_edf/chb_mit/chb01_03.edf")

    if os.path.exists(test_file):
        print(f"Loading {test_file}...")
        janitor = EEGJanitor()
        clean_raw = janitor.process(test_file)

        if clean_raw:
            eda = AdvancedEDA(clean_raw, sfreq=256)
            eda.run_full_dashboard()
    else:
        print("File not found. Please check path.")