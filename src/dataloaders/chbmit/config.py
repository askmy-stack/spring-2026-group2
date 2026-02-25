"""CHB-MIT dataset-specific configuration and metadata."""

CHBMIT_CONFIG = {
    "name": "chbmit",
    "base_url": "https://physionet.org/files/chbmit/1.0.0",
    "native_sfreq": 256.0,
    "target_sfreq": 256,
    "notch_freq": 60.0,       # US power line
    "raw_cache": "data/raw/chbmit",
    "population": "pediatric",
}

# Age/sex from CHB-MIT SUBJECT-INFO documentation
SUBJECT_META = {
    "chb01": {"age": 11, "sex": "F"}, "chb02": {"age": 11, "sex": "M"},
    "chb03": {"age": 14, "sex": "F"}, "chb04": {"age": 22, "sex": "M"},
    "chb05": {"age": 7,  "sex": "F"}, "chb06": {"age": 1,  "sex": "F"},
    "chb07": {"age": 14, "sex": "F"}, "chb08": {"age": 3,  "sex": "M"},
    "chb09": {"age": 10, "sex": "F"}, "chb10": {"age": 3,  "sex": "M"},
    "chb11": {"age": 12, "sex": "F"}, "chb12": {"age": 2,  "sex": "F"},
    "chb13": {"age": 3,  "sex": "F"}, "chb14": {"age": 9,  "sex": "F"},
    "chb15": {"age": 16, "sex": "M"}, "chb16": {"age": 7,  "sex": "F"},
    "chb17": {"age": 12, "sex": "F"}, "chb18": {"age": 18, "sex": "F"},
    "chb19": {"age": 19, "sex": "F"}, "chb20": {"age": 6,  "sex": "F"},
    "chb21": {"age": 13, "sex": "F"}, "chb22": {"age": 9,  "sex": "F"},
    "chb23": {"age": 6,  "sex": "F"}, "chb24": {"age": 16, "sex": "NA"},
}

ALL_SUBJECTS = sorted(SUBJECT_META.keys())