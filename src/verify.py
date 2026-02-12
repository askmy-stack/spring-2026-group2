import unittest
import pandas as pd
import torch
import shutil
import time
from pathlib import Path
from pipeline import SeizurePipeline, EEGDataset

# --- SMART PATH SETUP ---
# Automatically find config.yaml relative to this script
BASE_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = BASE_DIR / "config.yaml"


class TestPipeline(unittest.TestCase):
    def test_pipeline_flow(self):
        """Mini-test to ensure logic holds."""
        print("\n Running Logic Checks...")

        if not CONFIG_PATH.exists():
            self.fail(f"Config file not found at {CONFIG_PATH}")

        # Mock Dataframe
        df = pd.DataFrame({'subject_id': ['s1', 's2'], 'recording_id': ['r1', 'r2'], 'path': ['p1', 'p2']})

        # Initialize with Absolute Path
        pipe = SeizurePipeline(str(CONFIG_PATH))
        splits = pipe.split_data(df)
        self.assertIn('train', splits)
        print(" Logic Check Passed.")


def run_speed_test():
    print("\n⚡ RUNNING SPEED DEMO ⚡")

    # Try finding the 'train' index relative to src/
    idx_path = BASE_DIR / "results" / "dataloader" / "window_index_train.csv"

    if not idx_path.exists():
        print(f" Index not found at {idx_path}")
        print(" Run 'python src/main.py' (Option 1) first.")
        return

    # Initialize Dataset with Absolute Config Path
    ds = EEGDataset(str(idx_path), str(CONFIG_PATH))
    print(f" Dataset Size: {len(ds)} windows")

    if len(ds) > 0:
        start = time.time()
        # Test 10 random loads
        for _ in range(10):
            idx = torch.randint(0, len(ds), (1,)).item()
            _ = ds[idx]
        latency = ((time.time() - start) / 10) * 1000
        print(f" Latency: {latency:.2f} ms")
        if latency < 100: print(" Status: FAST (Ready for GPU)")
    else:
        print(" Dataset empty.")


if __name__ == "__main__":
    # Run Unit Tests
    unittest.main(exit=False)
    # Run Speed Demo
    run_speed_test()