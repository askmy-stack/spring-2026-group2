import os
import numpy as np
import mne
# We import the class we built in dataloader.py
from src.dataloader import EEGDataLoader


def create_dummy_data():
    """
    Creates dummy EDF files to test the pipeline.
    It generates random noise data and saves it as .edf files
    in the folder specified by 'data_raw'.
    """
    # Define where to save dummy data (must match config.yaml 'raw_root')
    raw_root = "./data_raw"
    os.makedirs(raw_root, exist_ok=True)

    # Create a dummy MNE Raw object (2 channels, 1 minute duration)
    info = mne.create_info(['Ch1', 'Ch2'], 256, 'eeg')
    data = np.random.randn(2, 256 * 60)
    raw = mne.io.RawArray(data, info)

    print(f"Generatng 5 dummy EDF files in {raw_root}...")
    # Create 5 subjects (sub00 to sub04)
    for i in range(5):
        # We save them as EDF files
        fname = f"{raw_root}/sub{i:02d}_task.edf"
        mne.export.export_raw(fname, raw, fmt='edf', overwrite=True)
    print("Dummy data generation complete.")


def main():
    # 1. Generate Fake Data (Optional: Comment this out if you have real data)
    # This checks if the folder is empty/missing and fills it
    if not os.path.exists("data_raw") or not os.listdir("data_raw"):
        create_dummy_data()

    # 2. Initialize Loader
    # CRITICAL FIX: We now pass the PATH to the config file ("config.yaml"),
    # not a dictionary object.
    loader = EEGDataLoader(config_path="config.yaml")

    # 3. Run Production Pipeline
    print("\n>>> Running Universal Pipeline...")
    # This executes: Discovery -> Parallel Preprocessing -> Splitting -> Indexing
    loader.run()


if __name__ == "__main__":
    main()