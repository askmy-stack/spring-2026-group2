import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy


class AdvancedFeatureExtractor:
    def __init__(self, sfreq=256):
        self.sfreq = sfreq
        self.bands = {
            'Delta': (0.5, 4), 'Theta': (4, 8),
            'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 45)
        }

    def _hjorth_params(self, epoch):
        """ Calculate Hjorth Activity, Mobility, and Complexity """
        # Activity = Variance
        activity = np.var(epoch, axis=1)

        # Mobility = sqrt(var(deriv) / var(signal))
        diff1 = np.diff(epoch, axis=1)
        # Avoid division by zero
        var_diff1 = np.var(diff1, axis=1)
        mobility = np.sqrt(var_diff1 / (activity + 1e-10))

        # Complexity = Mobility(deriv) / Mobility(signal)
        diff2 = np.diff(diff1, axis=1)
        mobility_deriv = np.sqrt(np.var(diff2, axis=1) / (var_diff1 + 1e-10))
        complexity = mobility_deriv / (mobility + 1e-10)

        return activity, mobility, complexity

    def _petrosian_fd(self, epoch):
        """ Petrosian Fractal Dimension """
        diff = np.diff(epoch, axis=1)
        # Count sign changes
        n_delta = np.sum(np.diff(np.sign(diff), axis=1) != 0, axis=1)
        n = epoch.shape[1]
        return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta + 1e-10)))

    def extract(self, epoch):
        """
        Extracts features for one epoch.
        Input: (Channels, Time)
        Output: Flattened 1D Array of features
        """
        feats_dict = {}

        # 1. Statistical Moments
        feats_dict['mean'] = np.mean(epoch, axis=1)
        feats_dict['std'] = np.std(epoch, axis=1)
        feats_dict['skew'] = skew(epoch, axis=1)
        feats_dict['kurt'] = kurtosis(epoch, axis=1)

        # 2. Hjorth Parameters
        act, mob, comp = self._hjorth_params(epoch)
        feats_dict['hjorth_act'] = act
        feats_dict['hjorth_mob'] = mob
        feats_dict['hjorth_comp'] = comp

        # 3. Fractal Dimension
        feats_dict['fractal_dim'] = self._petrosian_fd(epoch)

        # 4. Spectral Features
        freqs, psd = welch(epoch, fs=self.sfreq, nperseg=min(256, epoch.shape[1]))
        freq_res = freqs[1] - freqs[0]
        total_power = np.sum(psd, axis=1)

        for band, (low, high) in self.bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            power = np.sum(psd[:, idx], axis=1) * freq_res
            feats_dict[f'pow_{band}'] = power
            feats_dict[f'rel_{band}'] = power / (total_power + 1e-10)

        # Spectral Entropy
        psd_norm = psd / np.sum(psd, axis=1, keepdims=True)
        feats_dict['spec_ent'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12), axis=1)

        # Flatten Dictionary to 1D Array
        # We average across channels for the 'Global' features,
        # or we can concatenate. For XGBoost, flattening is common.
        # Here we take MEAN across channels to keep feature count manageable for baseline.
        flattened = []
        for k, v in feats_dict.items():
            flattened.append(np.mean(v))  # Global Average
            flattened.append(np.std(v))  # Global Spread

        return np.array(flattened)