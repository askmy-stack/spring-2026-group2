import mne
import numpy as np
import scipy.io
from pathlib import Path
import os


class UniversalReader:
    """
    A unified interface for loading EEG data from various formats
    (EDF, CNT, BIDS, MAT) into a standardized MNE Raw object.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def read(self, file_path, **kwargs):
        path_obj = Path(file_path)

        # 1. MATLAB (.mat) Support
        if path_obj.suffix.lower() == '.mat':
            if self.verbose: print(f"[Reader] Detected MATLAB file: {file_path}")
            return self._load_mat(file_path, **kwargs)

        # 2. BIDS Support
        elif self._is_bids(path_obj):
            if self.verbose: print(f"[Reader] Detected BIDS format: {file_path}")
            return self._load_bids(file_path, **kwargs)

        # 3. Neuroscan CNT
        elif path_obj.suffix.lower() == '.cnt':
            if self.verbose: print(f"[Reader] Detected Neuroscan CNT: {file_path}")
            return self._load_cnt(file_path, **kwargs)

        # 4. EDF/BDF
        elif path_obj.suffix.lower() in ['.edf', '.bdf', '.gdf']:
            if self.verbose: print(f"[Reader] Detected EDF/BDF: {file_path}")
            return self._load_edf(file_path, **kwargs)

        # 5. Generic Fallback
        else:
            if self.verbose: print(f"[Reader] Attempting Generic MNE Load: {file_path}")
            return self._load_generic(file_path, **kwargs)

    def _load_mat(self, file_path, sfreq=256, **kwargs):
        """
        Heuristic loader for .mat files. Finds the largest array and treats it as EEG.
        """
        try:
            mat = scipy.io.loadmat(file_path)
        except NotImplementedError:
            # Handle MATLAB v7.3 (HDF5 based) if needed, requires h5py
            raise ValueError("Old .mat format supported. For v7.3+, install h5py.")

        # Heuristic: Find the data variable (largest array)
        data_key = None
        max_size = 0
        for key, val in mat.items():
            if isinstance(val, np.ndarray) and key not in ['__header__', '__version__', '__globals__']:
                if val.size > max_size:
                    max_size = val.size
                    data_key = key

        if data_key is None:
            raise ValueError(f"Could not find EEG data array inside {file_path}")

        data = mat[data_key]

        # Ensure shape is (Channels, Time)
        if data.shape[0] > data.shape[1] and data.shape[1] < 128:
            # Assume strict channel limit to avoid flipping long recordings
            data = data.T

            # Create Dummy Info (MAT files often lack headers)
        ch_names = [f"Ch{i + 1}" for i in range(data.shape[0])]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info, verbose=self.verbose)
        return raw

    def _is_bids(self, path_obj):
        if path_obj.is_dir() and (path_obj / "dataset_description.json").exists(): return True
        return False

    def _load_cnt(self, fp, **kwargs):
        return mne.io.read_raw_cnt(fp, preload=kwargs.get('preload', True), verbose=self.verbose)

    def _load_edf(self, fp, **kwargs):
        return mne.io.read_raw_edf(fp, preload=kwargs.get('preload', False), verbose=self.verbose)

    def _load_bids(self, bids_root, task=None, subject=None, **kwargs):
        try:
            from mne_bids import find_matching_paths, read_raw_bids
            bids_paths = find_matching_paths(bids_root, tasks=task, subjects=subject)
            if not bids_paths: raise FileNotFoundError("No matching BIDS data found.")
            return read_raw_bids(bids_paths[0], verbose=self.verbose)
        except ImportError:
            raise ImportError("Install 'mne-bids' for BIDS support.")

    def _load_generic(self, fp, **kwargs):
        return mne.io.read_raw(fp, preload=kwargs.get('preload', False), verbose=self.verbose)
