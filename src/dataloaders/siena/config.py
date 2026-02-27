"""Siena Scalp EEG dataset-specific configuration and metadata."""

SIENA_CONFIG = {
    "name": "siena",
    "base_url": "https://physionet.org/files/siena-scalp-eeg/1.0.0",
    "native_sfreq": 512.0,
    "target_sfreq": 256,
    "notch_freq": 50.0,       # EU power line
    "raw_cache": "data/raw/siena",
    "population": "adult",
}

SUBJECT_META = {
    "PN00": {"age": 50, "sex": "F"}, "PN01": {"age": 46, "sex": "M"},
    "PN03": {"age": 42, "sex": "F"}, "PN05": {"age": 76, "sex": "M"},
    "PN06": {"age": 56, "sex": "F"}, "PN07": {"age": 25, "sex": "M"},
    "PN09": {"age": 46, "sex": "F"}, "PN10": {"age": 63, "sex": "M"},
    "PN11": {"age": 48, "sex": "M"}, "PN12": {"age": 29, "sex": "F"},
    "PN13": {"age": 55, "sex": "F"}, "PN14": {"age": 58, "sex": "M"},
    "PN16": {"age": 69, "sex": "F"}, "PN17": {"age": 37, "sex": "M"},
}

ALL_SUBJECTS = sorted(SUBJECT_META.keys())