from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from matplotlib import pyplot as plt
from scipy.signal import spectrogram as scipy_spectrogram, welch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch


APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
SERVER_UPLOAD_DIR = PROJECT_ROOT / "uploads"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.utils.artifacts import artifact_paths, list_available_artifacts
from models.utils.registry import create_model
from data_loader.core.channels import standardize_channels
from data_loader.core.io import read_raw
from data_loader.core.signal import normalize_signal, preprocess
from feature.feature_engineering import AdvancedFeatureExtractor


def log_event(message: str) -> None:
    print(f"[streamlit] {message}", flush=True)


def list_server_edf_files() -> list[Path]:
    if not SERVER_UPLOAD_DIR.exists():
        return []
    return sorted(
        path for path in SERVER_UPLOAD_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".edf"
    )


@dataclass(frozen=True)
class ModelProfile:
    key: str
    label: str
    channels: int
    samples: int
    sfreq: int
    window_sec: float
    notes: str


def _find_window_index_dir() -> Path | None:
    candidates = [
        PROJECT_ROOT / "results" / "dataloader",
        SRC_DIR / "results" / "dataloader",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_project_config() -> dict:
    config_path = SRC_DIR / "data_loader" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_feature_config() -> dict:
    config_path = SRC_DIR / "feature" / "fe.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_standard_channel_names(cfg: dict, fallback_count: int) -> list[str]:
    standard = list(cfg.get("channels", {}).get("standard_set", []))
    if standard:
        return standard[:fallback_count]
    return [f"ch_{idx:02d}" for idx in range(fallback_count)]


def make_eeg_preview_figure(data: np.ndarray, channel_names: list[str], sfreq: float) -> plt.Figure:
    n_channels, n_samples = data.shape
    time_axis = np.arange(n_samples, dtype=float) / float(sfreq)
    channel_scale = np.max(np.std(data, axis=1)) if n_channels > 0 else 1.0
    if not np.isfinite(channel_scale) or channel_scale <= 0:
        channel_scale = 1.0
    spacing = channel_scale * 6.0
    offsets = np.arange(n_channels - 1, -1, -1, dtype=float) * spacing

    fig_height = max(6.0, n_channels * 0.44)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    for idx in range(n_channels):
        ax.plot(time_axis, data[idx] + offsets[idx], linewidth=0.85, color="#32d1ff")

    ax.set_xlim(time_axis[0], time_axis[-1] if len(time_axis) > 1 else 1.0 / float(sfreq))
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")
    ax.set_title("Signal window")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def make_window_bandpower_figure(data: np.ndarray, channel_names: list[str], sfreq: float) -> plt.Figure:
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }
    band_matrix = np.zeros((len(channel_names), len(bands)), dtype=float)
    for idx in range(data.shape[0]):
        freqs, psd = welch(data[idx], fs=sfreq, nperseg=min(data.shape[1], int(sfreq)))
        for jdx, (_band, (lo, hi)) in enumerate(bands.items()):
            mask = (freqs >= lo) & (freqs <= hi)
            band_matrix[idx, jdx] = float(np.trapezoid(psd[mask], freqs[mask])) if np.any(mask) else 0.0

    fig, ax = plt.subplots(figsize=(8.6, max(5.5, len(channel_names) * 0.42)))
    im = ax.imshow(band_matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(bands)))
    ax.set_xticklabels(list(bands.keys()))
    ax.set_yticks(np.arange(len(channel_names)))
    ax.set_yticklabels(channel_names)
    ax.set_title("Band power")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


def make_channel_stats_figure(data: np.ndarray, channel_names: list[str]) -> plt.Figure:
    std_uv = np.std(data, axis=1) * 1e6
    rng_uv = (np.max(data, axis=1) - np.min(data, axis=1)) * 1e6
    x = np.arange(len(channel_names))

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.8, len(channel_names) * 0.26)))
    axes[0].bar(x, std_uv, color="#32d1ff")
    axes[0].set_title("Std (uV)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channel_names, rotation=90)

    axes[1].bar(x, rng_uv, color="#7d7cff")
    axes[1].set_title("Peak-to-peak (uV)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channel_names, rotation=90)

    fig.tight_layout()
    return fig


def make_correlation_figure(data: np.ndarray, channel_names: list[str]) -> plt.Figure:
    corr = np.corrcoef(data)
    fig, ax = plt.subplots(figsize=(8.4, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(channel_names)))
    ax.set_yticks(np.arange(len(channel_names)))
    ax.set_xticklabels(channel_names, rotation=90, fontsize=8)
    ax.set_yticklabels(channel_names, fontsize=8)
    ax.set_title("Channel correlation")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    return fig


def make_spectrogram_figure(data: np.ndarray, channel_names: list[str], sfreq: float, max_channels: int = 4) -> plt.Figure:
    show_n = min(max_channels, data.shape[0])
    fig, axes = plt.subplots(show_n, 1, figsize=(10.5, max(6.0, show_n * 2.2)), sharex=True)
    if show_n == 1:
        axes = [axes]
    for idx in range(show_n):
        # A 1-second EEG window needs shorter STFT segments; using the full
        # window collapses the time axis into one coarse bin and hides structure.
        n_times = data.shape[1]
        nperseg = min(max(32, int(sfreq // 4)), n_times)
        noverlap = min(int(nperseg * 0.75), max(0, nperseg - 1))
        freqs, times, spec = scipy_spectrogram(
            data[idx],
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            detrend=False,
            mode="psd",
        )
        mask = freqs <= 50
        spec_db = 10 * np.log10(spec[mask] + 1e-12)
        vmin = float(np.percentile(spec_db, 5))
        vmax = float(np.percentile(spec_db, 95))
        mesh = axes[idx].pcolormesh(
            times,
            freqs[mask],
            spec_db,
            shading="gouraud",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        axes[idx].set_ylabel(channel_names[idx], rotation=0, labelpad=30)
        axes[idx].set_ylim(0, 50)
        fig.colorbar(mesh, ax=axes[idx], fraction=0.025, pad=0.015)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Spectrogram")
    fig.tight_layout()
    return fig


def apply_app_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 20%, rgba(0, 218, 222, 0.16), transparent 22%),
                radial-gradient(circle at 85% 15%, rgba(45, 97, 255, 0.16), transparent 18%),
                linear-gradient(180deg, #07111f 0%, #0d1a2b 48%, #0a1421 100%);
            color: #eaf7ff;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        h1, h2, h3, label, p, div, span {
            color: #eaf7ff;
        }
        div[data-testid="stMetric"] {
            background: rgba(8, 20, 33, 0.72);
            border: 1px solid rgba(75, 204, 255, 0.16);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            box-shadow: 0 14px 30px rgba(0, 0, 0, 0.18);
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-testid="stFileUploader"] section,
        div[data-testid="stMultiSelect"] > div {
            background: rgba(8, 20, 33, 0.72) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(75, 204, 255, 0.16) !important;
        }
        button[kind="primary"] {
            border-radius: 999px !important;
            background: linear-gradient(90deg, #00d7d6 0%, #2f8cff 100%) !important;
            border: none !important;
            color: #07111f !important;
            font-weight: 700 !important;
        }
        .eeg-shell {
            background: rgba(8, 20, 33, 0.68);
            border: 1px solid rgba(75, 204, 255, 0.14);
            border-radius: 28px;
            padding: 1.25rem 1.4rem 0.9rem 1.4rem;
            box-shadow: 0 22px 44px rgba(0, 0, 0, 0.24);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.05;
            letter-spacing: -0.03em;
            margin-bottom: 0.35rem;
        }
        .hero-sub {
            color: #91c8dd;
            font-size: 0.98rem;
            margin-bottom: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


MODEL_PROFILES: dict[str, ModelProfile] = {
    "tabnet_features": ModelProfile(
        key="tabnet_features",
        label="TabNet",
        channels=16,
        samples=256,
        sfreq=256,
        window_sec=1.0,
        notes="Engineered features extracted from the selected EEG window and scored with the saved TabNet checkpoint.",
    ),
    "deepconvnet": ModelProfile(
        key="deepconvnet",
        label="DeepConvNet",
        channels=16,
        samples=256,
        sfreq=256,
        window_sec=1.0,
        notes="Uses the project CNN pipeline profile copied from saved training config.",
    ),
    "enhanced_cnn_1d": ModelProfile(
        key="enhanced_cnn_1d",
        label="Enhanced CNN 1D",
        channels=16,
        samples=256,
        sfreq=256,
        window_sec=1.0,
        notes="Uses the project CNN pipeline profile copied from saved training config.",
    ),
    "multiscale_attention_cnn": ModelProfile(
        key="multiscale_attention_cnn",
        label="Multiscale Attention CNN",
        channels=16,
        samples=256,
        sfreq=256,
        window_sec=1.0,
        notes="Uses the project CNN pipeline profile copied from saved training config.",
    ),
    "custom": ModelProfile(
        key="custom",
        label="Custom",
        channels=16,
        samples=256,
        sfreq=256,
        window_sec=1.0,
        notes="Set your own expected shape in the sidebar.",
    ),
}

APP_MODEL_KEYS = ("enhanced_cnn_1d", "multiscale_attention_cnn", "tabnet_features", "custom")


def adapt_window(array: np.ndarray, target_channels: int, target_samples: int) -> tuple[np.ndarray, dict[str, str]]:
    info: dict[str, str] = {}
    channels, samples = array.shape

    if channels > target_channels:
        array = array[:target_channels, :]
        info["channels"] = f"Trimmed channels from {channels} to {target_channels}."
    elif channels < target_channels:
        pad = np.zeros((target_channels - channels, samples), dtype=array.dtype)
        array = np.vstack([array, pad])
        info["channels"] = f"Padded channels from {channels} to {target_channels} with zeros."

    channels, samples = array.shape
    if samples > target_samples:
        array = array[:, :target_samples]
        info["samples"] = f"Trimmed samples from {samples} to {target_samples}."
    elif samples < target_samples:
        pad = np.zeros((channels, target_samples - samples), dtype=array.dtype)
        array = np.hstack([array, pad])
        info["samples"] = f"Padded samples from {samples} to {target_samples} with zeros."

    return array.astype(np.float32), info


def zscore_per_channel(array: np.ndarray) -> np.ndarray:
    mean = array.mean(axis=1, keepdims=True)
    std = array.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (array - mean) / std


def process_edf_upload(
    file_bytes: bytes,
    file_name: str,
    cfg: dict,
    target_channels: int,
    target_samples: int,
    window_start_sec: float,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, str], dict[str, object]]:
    started_at = time.perf_counter()
    suffix = Path(file_name).suffix.lower()
    if suffix != ".edf":
        raise ValueError("EDF processing was requested for a non-EDF upload.")

    log_event(f"process_edf_upload start file={file_name} window_start={window_start_sec:.2f}s")
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        step_started = time.perf_counter()
        raw = read_raw(tmp_path, preload=True)
        log_event(f"read_raw preload=True took {time.perf_counter() - step_started:.2f}s")
        original_channels = len(raw.ch_names)
        original_sfreq = float(raw.info["sfreq"])
        original_duration = float(raw.n_times / raw.info["sfreq"])

        step_started = time.perf_counter()
        raw = preprocess(raw, cfg)
        log_event(f"preprocess took {time.perf_counter() - step_started:.2f}s")
        step_started = time.perf_counter()
        raw = standardize_channels(raw, cfg)
        log_event(f"standardize_channels took {time.perf_counter() - step_started:.2f}s")
        step_started = time.perf_counter()
        data = raw.get_data()
        log_event(f"raw.get_data took {time.perf_counter() - step_started:.2f}s")
        sfreq = float(raw.info["sfreq"])
        duration = float(data.shape[1] / sfreq)
        window_samples = int(target_samples)
        start_sample = int(window_start_sec * sfreq)

        if data.shape[1] < window_samples:
            raise ValueError(
                f"EDF is too short after preprocessing: {duration:.2f}s available, "
                f"but {window_samples / sfreq:.2f}s is required."
            )
        if start_sample + window_samples > data.shape[1]:
            raise ValueError(
                f"Requested window exceeds EDF length. Max valid start is "
                f"{max(0.0, duration - (window_samples / sfreq)):.2f}s."
            )

        window = data[:, start_sample:start_sample + window_samples].astype(np.float32)
        if normalize:
            norm_method = cfg.get("signal", {}).get("normalize", "zscore")
            step_started = time.perf_counter()
            window = normalize_signal(window, method=norm_method).astype(np.float32)
            log_event(f"normalize_signal took {time.perf_counter() - step_started:.2f}s")

        adaptation_info = {
            "edf_original": (
                f"EDF upload preprocessed from {original_channels} channels at "
                f"{original_sfreq:.2f} Hz to {window.shape[0]} channels at {sfreq:.2f} Hz."
            ),
            "edf_window": (
                f"Selected EDF window from {window_start_sec:.2f}s to "
                f"{window_start_sec + (window_samples / sfreq):.2f}s."
            ),
        }
        metadata = {
            "source_type": "edf",
            "original_channels": original_channels,
            "original_sfreq": original_sfreq,
            "original_duration_sec": original_duration,
            "processed_channels": int(window.shape[0]),
            "processed_sfreq": sfreq,
            "processed_duration_sec": duration,
            "window_start_sec": float(window_start_sec),
            "channel_names": list(raw.ch_names),
        }
        log_event(
            f"process_edf_upload done in {time.perf_counter() - started_at:.2f}s "
            f"shape={window.shape} sfreq={sfreq:.2f}"
        )
        return window, adaptation_info, metadata
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def process_edf_path(
    edf_path: Path,
    cfg: dict,
    target_channels: int,
    target_samples: int,
    window_start_sec: float,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, str], dict[str, object]]:
    started_at = time.perf_counter()
    log_event(f"process_edf_path start file={edf_path} window_start={window_start_sec:.2f}s")
    if edf_path.suffix.lower() != ".edf":
        raise ValueError("Server file must be an EDF.")

    step_started = time.perf_counter()
    raw = read_raw(edf_path, preload=True)
    log_event(f"read_raw server preload=True took {time.perf_counter() - step_started:.2f}s")
    original_channels = len(raw.ch_names)
    original_sfreq = float(raw.info["sfreq"])
    original_duration = float(raw.n_times / raw.info["sfreq"])

    step_started = time.perf_counter()
    raw = preprocess(raw, cfg)
    log_event(f"preprocess server file took {time.perf_counter() - step_started:.2f}s")
    step_started = time.perf_counter()
    raw = standardize_channels(raw, cfg)
    log_event(f"standardize_channels server file took {time.perf_counter() - step_started:.2f}s")
    step_started = time.perf_counter()
    data = raw.get_data()
    log_event(f"raw.get_data server file took {time.perf_counter() - step_started:.2f}s")

    sfreq = float(raw.info["sfreq"])
    duration = float(data.shape[1] / sfreq)
    window_samples = int(target_samples)
    start_sample = int(window_start_sec * sfreq)

    if data.shape[1] < window_samples:
        raise ValueError(
            f"EDF is too short after preprocessing: {duration:.2f}s available, "
            f"but {window_samples / sfreq:.2f}s is required."
        )
    if start_sample + window_samples > data.shape[1]:
        raise ValueError(
            f"Requested window exceeds EDF length. Max valid start is "
            f"{max(0.0, duration - (window_samples / sfreq)):.2f}s."
        )

    window = data[:, start_sample:start_sample + window_samples].astype(np.float32)
    if normalize:
        norm_method = cfg.get("signal", {}).get("normalize", "zscore")
        step_started = time.perf_counter()
        window = normalize_signal(window, method=norm_method).astype(np.float32)
        log_event(f"normalize_signal server file took {time.perf_counter() - step_started:.2f}s")

    adaptation_info = {
        "edf_original": (
            f"EDF preprocessed from {original_channels} channels at "
            f"{original_sfreq:.2f} Hz to {window.shape[0]} channels at {sfreq:.2f} Hz."
        ),
        "edf_window": (
            f"Selected EDF window from {window_start_sec:.2f}s to "
            f"{window_start_sec + (window_samples / sfreq):.2f}s."
        ),
    }
    metadata = {
        "source_type": "server_file",
        "source_path": str(edf_path),
        "original_channels": original_channels,
        "original_sfreq": original_sfreq,
        "original_duration_sec": original_duration,
        "processed_channels": int(window.shape[0]),
        "processed_sfreq": sfreq,
        "processed_duration_sec": duration,
        "window_start_sec": float(window_start_sec),
        "channel_names": list(raw.ch_names),
    }
    log_event(
        f"process_edf_path done in {time.perf_counter() - started_at:.2f}s "
        f"shape={window.shape} sfreq={sfreq:.2f}"
    )
    return window, adaptation_info, metadata


@st.cache_data(show_spinner=False)
def get_edf_duration(file_bytes: bytes, file_name: str) -> float:
    started_at = time.perf_counter()
    suffix = Path(file_name).suffix.lower()
    if suffix != ".edf":
        raise ValueError("Only EDF files are supported.")
    log_event(f"get_edf_duration start file={file_name}")
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)
    try:
        raw_preview = read_raw(tmp_path, preload=False)
        duration = float(raw_preview.n_times / raw_preview.info["sfreq"])
        log_event(f"get_edf_duration done in {time.perf_counter() - started_at:.2f}s duration={duration:.2f}s")
        return duration
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@st.cache_data(show_spinner=False)
def get_edf_duration_from_path(edf_path: str) -> float:
    started_at = time.perf_counter()
    path = Path(edf_path)
    if path.suffix.lower() != ".edf":
        raise ValueError("Only EDF files are supported.")
    log_event(f"get_edf_duration_from_path start file={path}")
    raw_preview = read_raw(path, preload=False)
    duration = float(raw_preview.n_times / raw_preview.info["sfreq"])
    log_event(f"get_edf_duration_from_path done in {time.perf_counter() - started_at:.2f}s duration={duration:.2f}s")
    return duration


@st.cache_resource(show_spinner=False)
def load_inference_bundle(model_name: str) -> dict[str, object]:
    started_at = time.perf_counter()
    log_event(f"load_inference_bundle start model={model_name}")
    paths = artifact_paths(model_name)
    checkpoint = torch.load(paths.checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    threshold = float(checkpoint.get("best_threshold", 0.5))

    model_kwargs = {
        "in_channels": int(config.get("channels", 16)),
        "num_classes": 2,
        "dropout": float(config.get("dropout", 0.3)),
    }
    model = create_model(model_name, **model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log_event(f"load_inference_bundle done model={model_name} in {time.perf_counter() - started_at:.2f}s")
    return {
        "model": model,
        "threshold": threshold,
        "config": config,
    }


@st.cache_resource(show_spinner=False)
def load_tabnet_bundle() -> dict[str, object]:
    started_at = time.perf_counter()
    log_event("load_tabnet_bundle start")
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
    except ImportError as exc:
        raise ImportError(
            "TabNet inference requires `pytorch-tabnet`. Install it in the same environment as Streamlit."
        ) from exc

    checkpoint_path = SRC_DIR / "feature" / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TabNet checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint.get("args", {})
    state_dict = checkpoint["network_state_dict"]
    input_dim = int(state_dict["tabnet.initial_bn.weight"].shape[0])

    clf = TabNetClassifier(
        input_dim=input_dim,
        output_dim=2,
        n_d=int(args.get("n_d", 64)),
        n_a=int(args.get("n_a", 64)),
        n_steps=int(args.get("n_steps", 5)),
        gamma=float(args.get("gamma", 1.2)),
        lambda_sparse=float(args.get("lambda_sparse", 1.0e-3)),
        mask_type="entmax",
        device_name="cpu",
        verbose=0,
        seed=int(args.get("seed", 42)),
    )
    clf.classes_ = np.array([0, 1])
    clf.target_mapper = {0: 0, 1: 1}
    clf.preds_mapper = {0: 0, 1: 1}
    clf._default_loss = torch.nn.functional.cross_entropy
    clf.loss_fn = clf._default_loss
    clf._set_network()
    clf.network.load_state_dict(state_dict)
    clf.network.eval()
    log_event(f"load_tabnet_bundle done in {time.perf_counter() - started_at:.2f}s input_dim={input_dim}")
    return {
        "classifier": clf,
        "threshold": float(args.get("default_threshold", 0.5)),
        "input_dim": input_dim,
        "args": args,
    }


def extract_tabnet_features(window_data: np.ndarray, sfreq: float, cfg: dict) -> tuple[np.ndarray, list[str]]:
    started_at = time.perf_counter()
    log_event("extract_tabnet_features start")
    extractor = AdvancedFeatureExtractor(sfreq=sfreq, cfg=cfg)
    feature_dict = extractor.extract(window_data)
    feature_names = list(feature_dict.keys())
    feature_vector = np.asarray([feature_dict[name] for name in feature_names], dtype=np.float32)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    log_event(f"extract_tabnet_features done in {time.perf_counter() - started_at:.2f}s count={feature_vector.shape[0]}")
    return feature_vector, feature_names


def predict_tabnet_window(window_data: np.ndarray, sfreq: float, cfg: dict) -> dict[str, object]:
    started_at = time.perf_counter()
    log_event("predict_tabnet_window start")
    bundle = load_tabnet_bundle()
    feature_vector, feature_names = extract_tabnet_features(window_data, sfreq=sfreq, cfg=cfg)
    if feature_vector.shape[0] != int(bundle["input_dim"]):
        raise ValueError(
            f"TabNet feature length mismatch: extracted {feature_vector.shape[0]} features, "
            f"checkpoint expects {bundle['input_dim']}."
        )

    clf = bundle["classifier"]
    with torch.no_grad():
        logits, _ = clf.network(torch.from_numpy(feature_vector).unsqueeze(0).float())
        probs = torch.softmax(logits, dim=1)[0]
    seizure_prob = float(probs[1].item())
    background_prob = float(probs[0].item())
    threshold = float(bundle["threshold"])
    prediction = "seizure" if seizure_prob >= threshold else "background"
    log_event(
        f"predict_tabnet_window done in {time.perf_counter() - started_at:.2f}s "
        f"seizure_prob={seizure_prob:.4f}"
    )
    return {
        "model_name": "tabnet_features",
        "background_prob": background_prob,
        "seizure_prob": seizure_prob,
        "threshold": threshold,
        "prediction": prediction,
        "feature_count": int(feature_vector.shape[0]),
        "feature_preview": feature_names[:12],
    }


def predict_single_window(model_name: str, prepared_array: np.ndarray) -> dict[str, float | int | str]:
    started_at = time.perf_counter()
    log_event(f"predict_single_window start model={model_name}")
    bundle = load_inference_bundle(model_name)
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    x = torch.from_numpy(prepared_array).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    seizure_prob = float(probs[1].item())
    background_prob = float(probs[0].item())
    pred_label = "seizure" if seizure_prob >= threshold else "background"
    log_event(
        f"predict_single_window done model={model_name} in {time.perf_counter() - started_at:.2f}s "
        f"seizure_prob={seizure_prob:.4f}"
    )
    return {
        "model_name": model_name,
        "background_prob": background_prob,
        "seizure_prob": seizure_prob,
        "threshold": threshold,
        "prediction": pred_label,
    }


def main():
    log_event("app rerun started")
    available_artifacts = list_available_artifacts()
    available_artifact_map = {item["model_name"]: item for item in available_artifacts}
    project_cfg = load_project_config()
    feature_cfg = load_feature_config()
    standard_channel_names = get_standard_channel_names(project_cfg, fallback_count=16)

    st.set_page_config(page_title="EEG Seizure Model Demo", layout="wide")
    apply_app_theme()
    st.markdown(
        """
        <div class="eeg-shell">
          <div class="hero-title">Neural Event Viewer</div>
          <p class="hero-sub">Upload an EDF. Inspect the signal. Run seizure inference.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    expected_channels = 16
    expected_samples = 256
    expected_sfreq = 256
    window_sec = expected_samples / expected_sfreq
    normalize = True

    input_mode = st.radio(
        "Input source",
        options=["Upload from local", "Use server file"],
        horizontal=True,
    )

    uploaded = None
    server_file_path: Path | None = None
    if input_mode == "Upload from local":
        uploaded = st.file_uploader(
            "Upload one EDF file",
            type=["edf"],
            help="Upload a full EDF file. The app will extract one 1-second inference window.",
        )
    else:
        server_files = list_server_edf_files()
        if server_files:
            server_name = st.selectbox(
                "Select a server EDF file",
                options=[path.name for path in server_files],
                help=f"Files are read directly from {SERVER_UPLOAD_DIR}",
            )
            server_file_path = next((path for path in server_files if path.name == server_name), None)
        else:
            st.warning(f"No EDF files found in {SERVER_UPLOAD_DIR}")

    if "app_processed" not in st.session_state:
        st.session_state["app_processed"] = None
    if "app_results" not in st.session_state:
        st.session_state["app_results"] = None
    if "app_file_key" not in st.session_state:
        st.session_state["app_file_key"] = None

    if uploaded is None and server_file_path is None:
        log_event("no file uploaded")
        st.session_state["app_processed"] = None
        st.session_state["app_results"] = None
        st.session_state["app_file_key"] = None
        st.info("Upload one EDF file or select a server file to inspect and prepare it.")
    else:
        file_bytes = b""
        active_name = ""
        active_source_label = ""
        if uploaded is not None:
            active_name = uploaded.name
            active_source_label = "local_upload"
            log_event(f"file uploaded name={uploaded.name}")
            file_bytes = uploaded.getvalue()
            file_key = f"upload:{uploaded.name}:{len(file_bytes)}"
        else:
            assert server_file_path is not None
            active_name = server_file_path.name
            active_source_label = "server_file"
            stat = server_file_path.stat()
            log_event(f"server file selected name={server_file_path.name}")
            file_key = f"server:{server_file_path}:{stat.st_mtime_ns}:{stat.st_size}"
        if st.session_state["app_file_key"] != file_key:
            log_event("new uploaded file detected, clearing cached session state")
            st.session_state["app_processed"] = None
            st.session_state["app_results"] = None
            st.session_state["app_file_key"] = file_key

        try:
            if uploaded is not None:
                raw_duration = get_edf_duration(file_bytes, uploaded.name)
            else:
                raw_duration = get_edf_duration_from_path(str(server_file_path))
            edf_window_max_start = max(0.0, raw_duration - window_sec)
            window_input_cols = st.columns([1.3, 1.0])
            with window_input_cols[0]:
                window_start_sec = st.slider(
                    "EDF window start (seconds)",
                    min_value=0.0,
                    max_value=float(edf_window_max_start),
                    value=0.0,
                    step=float(window_sec),
                )
            with window_input_cols[1]:
                manual_window_start_sec = st.number_input(
                    "Manual start (s)",
                    min_value=0.0,
                    max_value=float(edf_window_max_start),
                    value=float(window_start_sec),
                    step=float(window_sec),
                    help="Type an exact start time in seconds for the selected inference window.",
                )
            window_start_sec = float(manual_window_start_sec)
        except Exception as exc:
            st.error(f"Could not parse the uploaded file: {exc}")
            return

        input_cols = st.columns([1.4, 1.0, 0.8])
        with input_cols[0]:
            st.metric("Recording", active_name)
        with input_cols[1]:
            st.metric("Duration", f"{raw_duration:.1f} s")
        with input_cols[2]:
            st.metric("Window", f"{window_sec:.1f} s")
        st.caption(f"Source: {active_source_label}")

        st.subheader("Run Inference")
        selected_models = st.multiselect(
            "Models to run",
            options=["enhanced_cnn_1d", "multiscale_attention_cnn", "tabnet_features"],
            default=["enhanced_cnn_1d", "multiscale_attention_cnn"],
            format_func=lambda key: MODEL_PROFILES[key].label,
        )
        action_cols = st.columns([0.9, 0.9, 2.2])
        run_inference = action_cols[0].button("Predict", type="primary")
        generate_visuals = action_cols[1].button("Show visuals")

        if run_inference:
            log_event("Predict button clicked")
            if not selected_models:
                st.warning("Select at least one model to run.")
                return
            with st.spinner("Preprocessing EDF window and running inference..."):
                if uploaded is not None:
                    prepared_array, adaptation_info, input_metadata = process_edf_upload(
                        file_bytes=file_bytes,
                        file_name=uploaded.name,
                        cfg=project_cfg,
                        target_channels=expected_channels,
                        target_samples=expected_samples,
                        window_start_sec=window_start_sec,
                        normalize=normalize,
                    )
                else:
                    assert server_file_path is not None
                    prepared_array, adaptation_info, input_metadata = process_edf_path(
                        edf_path=server_file_path,
                        cfg=project_cfg,
                        target_channels=expected_channels,
                        target_samples=expected_samples,
                        window_start_sec=window_start_sec,
                        normalize=normalize,
                    )
                st.session_state["app_processed"] = {
                    "array": prepared_array,
                    "adaptation_info": adaptation_info,
                    "input_metadata": input_metadata,
                    "window_start_sec": window_start_sec,
                }

            results = []
            missing_models = [name for name in selected_models if name != "tabnet_features" and name not in available_artifact_map]
            if missing_models:
                st.error(f"Missing saved artifacts for: {', '.join(missing_models)}")
                return

            for model_name in selected_models:
                try:
                    if model_name == "tabnet_features":
                        results.append(
                            predict_tabnet_window(
                                prepared_array,
                                sfreq=float(input_metadata.get("processed_sfreq", expected_sfreq)),
                                cfg=feature_cfg,
                            )
                        )
                    else:
                        results.append(predict_single_window(model_name, prepared_array))
                except Exception as exc:
                    st.error(f"Failed to run {model_name}: {exc}")
                    return
            st.session_state["app_results"] = results

        processed = st.session_state["app_processed"]
        if processed is not None:
            prepared_array = processed["array"]
            input_metadata = processed["input_metadata"]
            final_channels, final_samples = prepared_array.shape
            preview_channel_names = input_metadata.get("channel_names")
            if not isinstance(preview_channel_names, list) or len(preview_channel_names) != prepared_array.shape[0]:
                preview_channel_names = standard_channel_names[: prepared_array.shape[0]]

            metric_cols = st.columns(4)
            metric_cols[0].metric("Channels", str(final_channels))
            metric_cols[1].metric("Freq", f"{int(input_metadata.get('processed_sfreq', expected_sfreq))} Hz")
            metric_cols[2].metric("Window Start", f"{processed['window_start_sec']:.1f} s")
            metric_cols[3].metric("Samples", str(final_samples))

            with st.expander("Processed window details", expanded=False):
                st.write(
                    {
                        "uploaded_file": active_name,
                        "source": active_source_label,
                        "source_path": input_metadata.get("source_path", ""),
                        "processed_channels": final_channels,
                        "processed_sfreq": input_metadata.get("processed_sfreq", expected_sfreq),
                        "window_start_sec": input_metadata.get("window_start_sec", 0.0),
                        "channel_names": preview_channel_names,
                    }
                )

        results = st.session_state["app_results"]
        if results is not None:
            seizure_votes = sum(1 for result in results if result["prediction"] == "seizure")
            st.subheader("Prediction Result")
            if seizure_votes >= 1:
                st.error(f"SEIZURE detected by {seizure_votes}/{len(results)} model(s)")
            else:
                st.success("BACKGROUND / non-seizure")

            prediction_cols = st.columns(len(results))
            for col, result in zip(prediction_cols, results):
                with col:
                    st.markdown(f"**{MODEL_PROFILES[result['model_name']].label}**")
                    if result["prediction"] == "seizure":
                        st.error("Prediction: SEIZURE")
                    else:
                        st.success("Prediction: BACKGROUND")
                    st.metric("Seizure probability", f"{result['seizure_prob']:.4f}")
                    if result["model_name"] == "tabnet_features":
                        st.caption(f"Features: {result['feature_count']}")

            st.subheader("Detailed Results")
            result_df = pd.DataFrame(results)
            if "threshold" in result_df.columns:
                result_df = result_df.drop(columns=["threshold"])
            st.dataframe(result_df, width="stretch")
            tabnet_rows = [row for row in results if row["model_name"] == "tabnet_features"]
            if tabnet_rows:
                with st.expander("TabNet feature preview", expanded=False):
                    st.write(tabnet_rows[0]["feature_preview"])

        if generate_visuals:
            log_event("Show visuals button clicked")
            if st.session_state["app_processed"] is None:
                with st.spinner("Preprocessing EDF window for visualization..."):
                    if uploaded is not None:
                        prepared_array, adaptation_info, input_metadata = process_edf_upload(
                            file_bytes=file_bytes,
                            file_name=uploaded.name,
                            cfg=project_cfg,
                            target_channels=expected_channels,
                            target_samples=expected_samples,
                            window_start_sec=window_start_sec,
                            normalize=normalize,
                        )
                    else:
                        assert server_file_path is not None
                        prepared_array, adaptation_info, input_metadata = process_edf_path(
                            edf_path=server_file_path,
                            cfg=project_cfg,
                            target_channels=expected_channels,
                            target_samples=expected_samples,
                            window_start_sec=window_start_sec,
                            normalize=normalize,
                        )
                    st.session_state["app_processed"] = {
                        "array": prepared_array,
                        "adaptation_info": adaptation_info,
                        "input_metadata": input_metadata,
                        "window_start_sec": window_start_sec,
                    }
            processed = st.session_state["app_processed"]
            prepared_array = processed["array"]
            input_metadata = processed["input_metadata"]
            preview_channel_names = input_metadata.get("channel_names")
            if not isinstance(preview_channel_names, list) or len(preview_channel_names) != prepared_array.shape[0]:
                preview_channel_names = standard_channel_names[: prepared_array.shape[0]]

            tab_trace, tab_bands, tab_spectral, tab_corr = st.tabs(
                ["Trace", "Bands", "Spectral", "Connectivity"]
            )
            with tab_trace:
                st.pyplot(
                    make_eeg_preview_figure(
                        prepared_array[: min(16, prepared_array.shape[0])],
                        preview_channel_names[: min(16, prepared_array.shape[0])],
                        float(input_metadata.get("processed_sfreq", expected_sfreq)),
                    ),
                    clear_figure=True,
                    width="stretch",
                )
            with tab_bands:
                left_col, right_col = st.columns([1.1, 1.0])
                with left_col:
                    st.pyplot(
                        make_window_bandpower_figure(
                            prepared_array[: min(16, prepared_array.shape[0])],
                            preview_channel_names[: min(16, prepared_array.shape[0])],
                            float(input_metadata.get("processed_sfreq", expected_sfreq)),
                        ),
                        clear_figure=True,
                        width="stretch",
                    )
                with right_col:
                    st.pyplot(
                        make_channel_stats_figure(
                            prepared_array[: min(16, prepared_array.shape[0])],
                            preview_channel_names[: min(16, prepared_array.shape[0])],
                        ),
                        clear_figure=True,
                        width="stretch",
                    )
            with tab_spectral:
                st.pyplot(
                    make_spectrogram_figure(
                        prepared_array[: min(16, prepared_array.shape[0])],
                        preview_channel_names[: min(16, prepared_array.shape[0])],
                        float(input_metadata.get("processed_sfreq", expected_sfreq)),
                    ),
                    clear_figure=True,
                    width="stretch",
                )
            with tab_corr:
                st.pyplot(
                    make_correlation_figure(
                        prepared_array[: min(16, prepared_array.shape[0])],
                        preview_channel_names[: min(16, prepared_array.shape[0])],
                    ),
                    clear_figure=True,
                    width="stretch",
                )

    if st.session_state["app_processed"] is None:
        st.info("Choose an EDF source, select a window, then click Predict.")
        return


if __name__ == "__main__":
    main()
