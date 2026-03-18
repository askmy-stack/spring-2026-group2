from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.io import download_sample_edf, read_raw, scan_raw_dir
from core.signal import preprocess, normalize_signal
from core.channels import standardize_channels, STANDARD_16
from core.cache import PickleCacher, make_cache_key
from core.labels import build_window_index, balance_index, extract_seizure_intervals
from core.stratify import stratify_subjects, assign_split_column
from core.augment import augment, time_warp, magnitude_scale, add_noise, time_shift
from core.bids import convert_to_bids, load_participants
from dataset.factory import create_loader
from dataset.loaders import StandardEEGLoader, CachedEEGLoader, EnhancedEEGLoader
from dataset.base import BaseEEGDataset
import pandas as pd
import tempfile
import yaml


CFG_PATH = Path(__file__).parent.parent / "config.yaml"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


def _load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def _get_sample_edf() -> Path:
    return download_sample_edf(DATA_DIR)


def test_download_sample_edf():
    p = _get_sample_edf()
    assert p.exists()
    assert p.suffix == ".edf"


def test_read_raw():
    p = _get_sample_edf()
    raw = read_raw(p, preload=True)
    assert len(raw.ch_names) > 0
    assert raw.info["sfreq"] > 0


def test_preprocess():
    cfg = _load_cfg()
    p = _get_sample_edf()
    raw = read_raw(p, preload=True)
    raw = preprocess(raw, cfg)
    assert int(raw.info["sfreq"]) == cfg["signal"]["target_sfreq"]


def test_channel_standardization_reduce():
    cfg = _load_cfg()
    p = _get_sample_edf()
    raw = read_raw(p, preload=True)
    raw = preprocess(raw, cfg)
    raw = standardize_channels(raw, cfg)
    assert len(raw.ch_names) == 16


def test_channel_standardization_expand():
    cfg = _load_cfg()
    import mne
    ch_names = ["Fp1", "Fp2", "Fz"]
    data = np.random.randn(3, 256)
    info = mne.create_info(ch_names=ch_names, sfreq=256, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    raw = standardize_channels(raw, cfg)
    assert len(raw.ch_names) == 16


def test_normalize_signal():
    data = np.random.randn(16, 256)
    normed = normalize_signal(data, method="zscore")
    assert normed.shape == (16, 256)
    assert abs(normed.mean()) < 0.1


def test_augmentation():
    cfg = _load_cfg()
    cfg["augmentation"]["enable"] = True
    data = np.random.randn(16, 256)
    aug = augment(data, label=1, cfg=cfg)
    assert aug.shape == (16, 256)


def test_augmentation_functions():
    data = np.random.randn(16, 256)
    assert time_warp(data).shape == (16, 256)
    assert magnitude_scale(data).shape == (16, 256)
    assert add_noise(data).shape == (16, 256)
    assert time_shift(data).shape == (16, 256)


def test_cache_put_get():
    with tempfile.TemporaryDirectory() as tmp:
        cacher = PickleCacher(Path(tmp), memory_limit_mb=100)
        key = make_cache_key("file.edf", 0.0, 1.0)
        payload = np.random.randn(16, 256)
        cacher.put(key, payload, ns="raw")
        result = cacher.get(key, ns="raw")
        assert result is not None
        assert np.allclose(result, payload)


def test_cache_miss():
    with tempfile.TemporaryDirectory() as tmp:
        cacher = PickleCacher(Path(tmp))
        result = cacher.get("nonexistent_key", ns="raw")
        assert result is None


def test_cache_stats():
    with tempfile.TemporaryDirectory() as tmp:
        cacher = PickleCacher(Path(tmp))
        key = "k1"
        cacher.get(key, ns="raw")
        cacher.put(key, [1, 2, 3], ns="raw")
        cacher.get(key, ns="raw")
        assert cacher.stats.hits == 1
        assert cacher.stats.misses == 1


def test_build_window_index():
    cfg = _load_cfg()
    cfg_no_exclude = {**cfg, "windowing": {**cfg.get("windowing", {}), "exclude_negatives_within_sec": 0}}

    df = build_window_index(
        eeg_path="/fake/file.edf",
        duration_sec=10.0,
        subject_id="0001",
        seizure_intervals=[],
        cfg=cfg_no_exclude,
    )
    assert len(df) == 10
    assert "label" in df.columns
    assert "start_sec" in df.columns
    assert df["label"].isin([0, 1]).all()

    df_labeled = build_window_index(
        eeg_path="/fake/file.edf",
        duration_sec=10.0,
        subject_id="0001",
        seizure_intervals=[(3.0, 5.0)],
        cfg=cfg_no_exclude,
    )
    assert (df_labeled["label"] == 1).sum() >= 1


def test_balance_index():
    cfg = _load_cfg()
    rows = [{"path": "f.edf", "subject_id": "s1", "start_sec": i,
              "end_sec": i + 1, "label": 0, "age": "NA", "sex": "NA"}
            for i in range(90)]
    rows += [{"path": "f.edf", "subject_id": "s1", "start_sec": i,
               "end_sec": i + 1, "label": 1, "age": "NA", "sex": "NA"}
             for i in range(10)]
    df = pd.DataFrame(rows)
    balanced = balance_index(df, cfg)
    pos = (balanced["label"] == 1).sum()
    neg = (balanced["label"] == 0).sum()
    assert pos > 10


def test_stratify_subjects():
    cfg = _load_cfg()
    subjects = pd.DataFrame([
        {"subject_id": f"s{i:03d}", "age": 10 + i * 2, "sex": "M" if i % 2 == 0 else "F"}
        for i in range(10)
    ])
    train, val, test = stratify_subjects(subjects, cfg)

    train_ids = set(train["subject_id"])
    val_ids = set(val["subject_id"])
    test_ids = set(test["subject_id"])

    all_ids = train_ids | val_ids | test_ids
    assert all_ids == set(subjects["subject_id"]), "Some subjects lost during stratification"

    assert len(train_ids & val_ids) == 0, "Train/val overlap detected"
    assert len(train_ids & test_ids) == 0, "Train/test overlap detected"
    assert len(val_ids & test_ids) == 0, "Val/test overlap detected"

    assert len(train_ids) > 0, "Train set is empty"
    assert len(val_ids) > 0, "Val set is empty"
    assert len(test_ids) > 0, "Test set is empty"


def test_stratify_subjects_small_cohort():
    """Reproduce the 6-subject CHB-MIT scenario that previously gave empty val."""
    cfg = _load_cfg()
    subjects = pd.DataFrame([
        {"subject_id": "chb01", "age": 11, "sex": "F"},
        {"subject_id": "chb02", "age": 11, "sex": "M"},
        {"subject_id": "chb03", "age": 14, "sex": "F"},
        {"subject_id": "chb05", "age": 7,  "sex": "F"},
        {"subject_id": "chb08", "age": 3,  "sex": "M"},
        {"subject_id": "chb10", "age": 3,  "sex": "M"},
    ])
    train, val, test = stratify_subjects(subjects, cfg)

    train_ids = set(train["subject_id"])
    val_ids = set(val["subject_id"])
    test_ids = set(test["subject_id"])

    assert train_ids | val_ids | test_ids == set(subjects["subject_id"])

    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0

    assert len(val_ids) >= 1, f"Val set is empty! train={train_ids}, val={val_ids}, test={test_ids}"
    assert len(test_ids) >= 1, f"Test set is empty! train={train_ids}, val={val_ids}, test={test_ids}"
    assert len(train_ids) >= len(val_ids)
    assert len(train_ids) >= len(test_ids)


def test_stratify_subjects_minimum():
    """With exactly 3 subjects, each split must get exactly 1."""
    cfg = _load_cfg()
    subjects = pd.DataFrame([
        {"subject_id": "s1", "age": 5,  "sex": "M"},
        {"subject_id": "s2", "age": 15, "sex": "F"},
        {"subject_id": "s3", "age": 25, "sex": "M"},
    ])
    train, val, test = stratify_subjects(subjects, cfg)

    assert len(train) == 1
    assert len(val) == 1
    assert len(test) == 1

    all_ids = set(train["subject_id"]) | set(val["subject_id"]) | set(test["subject_id"])
    assert all_ids == {"s1", "s2", "s3"}


def test_no_window_level_leakage():
    """Verify that assign_split_column never puts the same subject in multiple splits."""
    rows = []
    for sid in ["s1", "s2", "s3"]:
        for i in range(5):
            rows.append({"subject_id": sid, "start_sec": i, "end_sec": i + 1,
                         "label": 0, "path": "fake.edf"})
    windows_df = pd.DataFrame(rows)

    result = assign_split_column(windows_df, ["s1"], ["s2"], ["s3"])

    for sid in ["s1", "s2", "s3"]:
        splits_for_subject = result[result["subject_id"] == sid]["split"].unique()
        assert len(splits_for_subject) == 1, (
            f"Subject {sid} appears in multiple splits: {splits_for_subject}"
        )

    assert (result[result["subject_id"] == "s1"]["split"] == "train").all()
    assert (result[result["subject_id"] == "s2"]["split"] == "val").all()
    assert (result[result["subject_id"] == "s3"]["split"] == "test").all()


def test_assign_split_column_unmapped_raises():
    """Verify that unmapped subject IDs raise ValueError instead of silently defaulting."""
    rows = [
        {"subject_id": "s1", "start_sec": 0, "end_sec": 1, "label": 0, "path": "f.edf"},
        {"subject_id": "s_unknown", "start_sec": 0, "end_sec": 1, "label": 0, "path": "f.edf"},
    ]
    windows_df = pd.DataFrame(rows)

    try:
        assign_split_column(windows_df, ["s1"], [], [])
        assert False, "Should have raised ValueError for unmapped subject 's_unknown'"
    except ValueError:
        pass


def test_bids_conversion():
    cfg = _load_cfg()
    p = _get_sample_edf()
    raw = read_raw(p, preload=True)
    raw = preprocess(raw, cfg)
    raw = standardize_channels(raw, cfg)
    meta = {"age": 25, "sex": "M"}
    with tempfile.TemporaryDirectory() as tmp:
        bids_out = Path(tmp) / "bids"
        bids_path = convert_to_bids(raw, bids_out, "0001", meta, cfg)
        assert bids_path.exists()
        assert (bids_out / "sub-0001" / "eeg").exists()
        participants = load_participants(bids_out)
        assert "sub-0001" in participants["participant_id"].values


def test_loader_class_hierarchy():
    assert issubclass(StandardEEGLoader, BaseEEGDataset)
    assert issubclass(CachedEEGLoader, BaseEEGDataset)
    assert issubclass(EnhancedEEGLoader, BaseEEGDataset)


def test_loader_empty():
    loader = create_loader("standard", config_path=str(CFG_PATH), mode="train")
    assert isinstance(loader, StandardEEGLoader)
    assert len(loader) >= 0


def test_loader_shape():
    loader = create_loader("cached", config_path=str(CFG_PATH), mode="train")
    if len(loader) == 0:
        return
    data, label = loader[0]
    assert isinstance(data, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert data.shape[0] == loader.n_channels
    assert data.shape[1] == loader.window_samples


def test_class_weights():
    loader = create_loader("standard", config_path=str(CFG_PATH), mode="train")
    w = loader.get_class_weights()
    assert isinstance(w, torch.Tensor)
    assert w.shape[0] == 2


def test_eda_helpers_empty():
    loader = create_loader("standard", config_path=str(CFG_PATH), mode="train")
    labels = loader.get_labels()
    subjects = loader.get_subject_ids()
    df = loader.get_data_index()
    assert isinstance(labels, np.ndarray)
    assert isinstance(subjects, np.ndarray)
    assert isinstance(df, pd.DataFrame)


def test_num_classes_empty():
    loader = create_loader("standard", config_path=str(CFG_PATH), mode="train")
    nc = loader.num_classes
    assert isinstance(nc, int)
    assert nc >= 2


def test_class_weights_multiclass():
    """Verify get_class_weights works with more than 2 classes."""
    loader = create_loader("standard", config_path=str(CFG_PATH), mode="train")
    loader.data_index = pd.DataFrame({
        "label": [0, 0, 0, 1, 1, 2],
        "subject_id": ["s1"] * 6,
        "path": ["f.edf"] * 6,
        "start_sec": list(range(6)),
        "end_sec": list(range(1, 7)),
    })
    w = loader.get_class_weights()
    assert w.shape[0] == 3
    assert loader.num_classes == 3


def run_all():
    tests = [
        test_download_sample_edf,
        test_read_raw,
        test_preprocess,
        test_channel_standardization_reduce,
        test_channel_standardization_expand,
        test_normalize_signal,
        test_augmentation,
        test_augmentation_functions,
        test_cache_put_get,
        test_cache_miss,
        test_cache_stats,
        test_build_window_index,
        test_balance_index,
        test_stratify_subjects,
        test_stratify_subjects_small_cohort,
        test_stratify_subjects_minimum,
        test_no_window_level_leakage,
        test_assign_split_column_unmapped_raises,
        test_bids_conversion,
        test_loader_class_hierarchy,
        test_loader_empty,
        test_loader_shape,
        test_class_weights,
        test_eda_helpers_empty,
        test_num_classes_empty,
        test_class_weights_multiclass,
    ]
    passed = 0
    failed = []
    for t in tests:
        name = t.__name__
        result = "PASS"
        err = ""
        try:
            t()
            passed += 1
        except Exception as e:
            result = "FAIL"
            err = f"{type(e).__name__}: {e}"
            failed.append((name, err))
        print(f"  [{result}] {name}" + (f"  -> {err}" if err else ""))
    print(f"\n{passed}/{len(tests)} tests passed")
    if failed:
        print("\nFailed:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    run_all()