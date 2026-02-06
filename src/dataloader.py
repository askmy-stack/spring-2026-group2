import os
import mne
from mne_bids import BIDSPath, read_raw_bids
from pathlib import Path


class UniversalReader:

    def __init__(self, verbose=False):
        self.verbose = verbose

    def read(self, file_path, **kwargs):
        """
        Main entry point. Automatically detects format and routes to the correct loader.

        Args:
            file_path (str | Path): Path to file (.edf, .cnt) or BIDS root directory.
            **kwargs: Extra arguments for specific MNE loaders (e.g., preload=True).

        Returns:
            mne.io.Raw: The loaded raw EEG data.
        """
        path_obj = Path(file_path)

        # 1. Check for BIDS (Directory or BIDSPath)
        if self._is_bids(path_obj):
            print(f"[Reader] Detected BIDS format: {file_path}")
            return self._load_bids(file_path, **kwargs)

        # 2. Check for CNT (Neuroscan)
        elif path_obj.suffix.lower() == '.cnt':
            print(f"[Reader] Detected Neuroscan CNT: {file_path}")
            return self._load_cnt(file_path, **kwargs)

        # 3. Check for EDF/EDF+ (European Data Format)
        elif path_obj.suffix.lower() in ['.edf', '.bdf', '.gdf']:
            print(f"[Reader] Detected EDF/BDF: {file_path}")
            return self._load_edf(file_path, **kwargs)

        # 4. Fallback (MNE Auto-detection for .fif, .vhdr, etc.)
        else:
            print(f"[Reader] Attempting Generic MNE Load: {file_path}")
            return self._load_generic(file_path, **kwargs)

    def _is_bids(self, path_obj):
        """Simple heuristic to detect BIDS structure."""
        # If it's a directory containing 'dataset_description.json', it's a BIDS root
        if path_obj.is_dir():
            if (path_obj / "dataset_description.json").exists():
                return True
        return False

    def _load_cnt(self, file_path, **kwargs):
        """
        Loads Neuroscan .CNT files.
        Crucial: CNT often requires 'preload=True' to handle large headers correctly.
        """
        # CNT files often have date issues, 'auto' usually fixes them.
        return mne.io.read_raw_cnt(
            file_path,
            preload=kwargs.get('preload', True),  # CNT often needs preloading
            data_format='auto',
            date_format='mm/dd/yy',
            verbose=self.verbose
        )

    def _load_edf(self, file_path, **kwargs):
        """Loads EDF/EDF+ files."""
        return mne.io.read_raw_edf(
            file_path,
            preload=kwargs.get('preload', False),
            verbose=self.verbose
        )

    def _load_bids(self, bids_root, task=None, subject=None, **kwargs):
        """
        Loads BIDS datasets.
        Note: BIDS is a structure, not a single file. You typically load a specific run.
        """
        # Construct a BIDS Path query
        # This is a simplified loader that grabs the first matching run found
        # In a real app, you would iterate over subjects/tasks.
        try:
            from mne_bids import find_matching_paths

            # Find all .edf/.vhdr/.cnt files in the BIDS folder
            bids_paths = find_matching_paths(bids_root, tasks=task, subjects=subject)

            if not bids_paths:
                raise FileNotFoundError("No matching BIDS data found.")

            # Load the first match (Example logic)
            target_path = bids_paths[0]
            print(f"[Reader] Loading BIDS Subject: {target_path.subject}, Task: {target_path.task}")

            return read_raw_bids(target_path, verbose=self.verbose)

        except ImportError:
            raise ImportError("Please install 'mne-bids' to load BIDS datasets: pip install mne-bids")

    def _load_generic(self, file_path, **kwargs):
        """Fallback to MNE's smart generic loader."""
        return mne.io.read_raw(file_path, preload=kwargs.get('preload', False), verbose=self.verbose)


if __name__ == "__main__":
    reader = UniversalReader(verbose=True)