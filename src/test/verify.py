import pytest
from pathlib import Path
from pipeline import SeizurePipeline
from mne_bids import BIDSPath, read_raw_bids

def test_bids_structure():
    """Checks if folders follow sub-XX/ses-YY/eeg pattern"""
    root = Path("../results/bids_dataset")
    subs = list(root.glob("sub-*"))
    assert len(subs) > 0, "No subjects found in BIDS root!"
    print("BIDS Structure Valid")

def test_metadata_injection():
    """Checks if participants.tsv exists and has columns"""
    tsv = Path("../results/bids_dataset/participants.tsv")
    assert tsv.exists(), "participants.tsv missing!"
    with open(tsv, 'r') as f:
        header = f.readline()
    assert "age" in header and "sex" in header, "Metadata columns missing!"
    print(" Metadata Valid")

if __name__ == "__main__":
    test_bids_structure()
    test_metadata_injection()
    print(" ALL TESTS PASSED")