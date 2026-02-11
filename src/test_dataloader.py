import unittest
import shutil
import os
import numpy as np
import pandas as pd
import yaml
import mne
# Ensure this points to your actual dataloader file location
from src.dataloader import EEGDataLoader


class TestEEGDataLoader(unittest.TestCase):

    def setUp(self):
        # 1. Define Test Config
        self.test_config_path = "test_config.yaml"
        self.cfg = {
            "dataset": {
                "raw_root": "./test_raw",
                "results_root": "./test_results",
                "bids_root": "./test_results/bids"  # Added specifically for BIDS tests
            },
            "split": {"train": 0.6, "val": 0.2, "test": 0.2, "seed": 42},
            "windowing": {"window_sec": 4, "stride_sec": 4},
            "signal": {"target_sfreq": 256, "bandpass": [1, 50]},  # Added required signal params
            "system": {"n_jobs": 1}  # Force single core for testing
        }

        # 2. Write Config to YAML file (Fixes TypeError)
        with open(self.test_config_path, 'w') as f:
            yaml.dump(self.cfg, f)

        # 3. Create Dummy Data
        os.makedirs(self.cfg["dataset"]["raw_root"], exist_ok=True)
        info = mne.create_info(['C3', 'C4'], 256, 'eeg')
        raw = mne.io.RawArray(np.random.randn(2, 1000), info)

        # FIX: Added overwrite=True (Fixes FileExistsError)
        mne.export.export_raw("./test_raw/test_sub.edf", raw, fmt='edf', overwrite=True)

        # 4. Initialize Loader with PATH, not DICT
        self.loader = EEGDataLoader(self.test_config_path)

    def tearDown(self):
        # Cleanup folders
        if os.path.exists("./test_raw"):
            shutil.rmtree("./test_raw")
        if os.path.exists("./test_results"):
            shutil.rmtree("./test_results")
        # Cleanup config file
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    def test_metadata_injection_logic(self):
        """Test the internal helper function for metadata"""
        # We test the helper function _inject_metadata directly if possible,
        # or we verify the loader's ability to run without crashing.
        # Since _inject_metadata is a helper, we test the flow:

        # Manually load the dummy file
        raw = mne.io.read_raw_edf("./test_raw/test_sub.edf", preload=True, verbose='ERROR')

        # Import the helper (it's defined in dataloader.py)
        from src.dataloader import _inject_metadata

        raw = _inject_metadata(raw, "001", age=30, sex="M")

        # Assertions
        self.assertIn('subject_info', raw.info)
        self.assertEqual(raw.info['subject_info']['sex'], 1)  # 1=Male

    def test_stratification_fallback(self):
        """Test Actions 1 & 2 (Split)"""
        # Create fake file list
        df = pd.DataFrame({'subject_id': ['s1', 's2', 's3'], 'path': ['p1', 'p2', 'p3']})
        # Only 3 subjects -> Stratification impossible -> Should fallback to random without crash
        # Note: 'strat_key' creation happens inside split_data now, so we pass raw parts

        # We mock the dataframe to have 'age' and 'sex' columns if we want to test that logic,
        # but split_data expects a merged dataframe or handles it internally.
        # Let's mock a dataframe that looks like the result of discover_files + merge
        df_mock = pd.DataFrame({
            'subject_id': ['s1', 's2', 's3'],
            'path': ['p1', 'p2', 'p3'],
            'age': [20, 20, 90],
            'sex': ['M', 'M', 'F']
        })

        splits = self.loader.split_data(df_mock)

        self.assertIn("train", splits)
        self.assertIn("test", splits)
        # Check total count matches
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        self.assertEqual(total, 3)


if __name__ == '__main__':
    unittest.main()