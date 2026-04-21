from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

THIS_DIR = Path(__file__).parent
SRC_DIR = THIS_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from data_loader.core.channels import standardize_channels
from data_loader.core.io import read_raw
from data_loader.core.signal import normalize_signal, preprocess
from model.factory import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate saliency and multi-scale activation visualizations.")
    parser.add_argument("--model", required=True, choices=("enhanced_cnn_1d", "multiscale_attention_cnn"))
    parser.add_argument("--edf-path", required=True, help="Path to EDF file.")
    parser.add_argument("--window-start-sec", type=float, default=0.0, help="Window start time in seconds.")
    parser.add_argument("--config-path", default=str(SRC_DIR / "data_loader" / "config.yaml"))
    parser.add_argument("--checkpoint-path", default="", help="Optional explicit checkpoint path.")
    parser.add_argument("--out-dir", default="", help="Optional explicit output directory.")
    parser.add_argument("--target-class", default="predicted", choices=("predicted", "seizure", "background"))
    return parser.parse_args()


@dataclass
class PreparedWindow:
    window: np.ndarray
    sfreq: float
    channel_names: list[str]
    metadata: dict[str, object]


def load_cfg(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_window(edf_path: Path, cfg: dict, window_start_sec: float) -> PreparedWindow:
    raw = read_raw(edf_path, preload=True)
    raw = preprocess(raw, cfg)
    raw = standardize_channels(raw, cfg)
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    window_sec = float(cfg.get("windowing", {}).get("window_sec", 1.0))
    target_samples = int(window_sec * sfreq)
    start_sample = int(window_start_sec * sfreq)
    if start_sample + target_samples > data.shape[1]:
        raise ValueError(
            f"Requested window exceeds available samples. Max valid start is "
            f"{max(0.0, (data.shape[1] / sfreq) - window_sec):.2f}s."
        )

    window = data[:, start_sample:start_sample + target_samples].astype(np.float32)
    window = normalize_signal(window, method=cfg.get("signal", {}).get("normalize", "zscore")).astype(np.float32)
    return PreparedWindow(
        window=window,
        sfreq=sfreq,
        channel_names=list(raw.ch_names),
        metadata={
            "edf_path": str(edf_path),
            "window_start_sec": float(window_start_sec),
            "window_sec": window_sec,
            "shape": list(window.shape),
        },
    )


def infer_checkpoint_path(model_name: str) -> Path:
    return PROJECT_ROOT / "outputs" / "models" / model_name / f"{model_name}_v1.pt"


def load_model(model_name: str, checkpoint_path: Path) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = create_model(
        model_name,
        in_channels=int(config.get("channels", 16)),
        num_classes=2,
        dropout=float(config.get("dropout", 0.3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def choose_target_class(logits: torch.Tensor, target_class: str) -> int:
    if target_class == "predicted":
        return int(logits.argmax(dim=1).item())
    if target_class == "seizure":
        return 1
    return 0


def compute_saliency(model: torch.nn.Module, window: np.ndarray, target_class: str) -> tuple[np.ndarray, np.ndarray, int]:
    x = torch.from_numpy(window).unsqueeze(0).requires_grad_(True)
    logits = model(x)
    class_idx = choose_target_class(logits, target_class)
    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()
    grad = x.grad.detach().cpu().numpy()[0]
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    saliency = np.abs(grad)
    saliency = saliency / (saliency.max() + 1e-8)
    return saliency, probs, class_idx


def make_saliency_figure(window: np.ndarray, saliency: np.ndarray, channel_names: list[str], sfreq: float, title: str) -> plt.Figure:
    n_channels, n_samples = window.shape
    t = np.arange(n_samples, dtype=float) / sfreq
    channel_scale = np.max(np.std(window, axis=1)) if n_channels > 0 else 1.0
    if not np.isfinite(channel_scale) or channel_scale <= 0:
        channel_scale = 1.0
    spacing = channel_scale * 6.0
    offsets = np.arange(n_channels - 1, -1, -1, dtype=float) * spacing

    fig, ax = plt.subplots(figsize=(14, max(7, n_channels * 0.5)))
    for idx in range(n_channels):
        trace = window[idx] + offsets[idx]
        ax.plot(t, trace, color="#1c2d4d", linewidth=0.9, zorder=2)
        ax.scatter(
            t,
            trace,
            c=saliency[idx],
            cmap="inferno",
            s=8,
            vmin=0.0,
            vmax=1.0,
            linewidths=0,
            zorder=3,
        )

    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channels")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)
    mappable = plt.cm.ScalarMappable(cmap="inferno")
    mappable.set_clim(0.0, 1.0)
    fig.colorbar(mappable, ax=ax, fraction=0.025, pad=0.02, label="Normalized saliency")
    fig.tight_layout()
    return fig


def capture_multiscale_branch_outputs(model: torch.nn.Module, window: np.ndarray) -> list[np.ndarray]:
    if not hasattr(model, "branches"):
        return []

    outputs: list[np.ndarray] = []
    hooks = []

    def make_hook():
        def _hook(_module, _inputs, output):
            outputs.append(output.detach().cpu().numpy()[0])
        return _hook

    for branch in model.branches:
        hooks.append(branch.register_forward_hook(make_hook()))

    x = torch.from_numpy(window).unsqueeze(0)
    with torch.no_grad():
        _ = model(x)

    for hook in hooks:
        hook.remove()
    return outputs


def make_multiscale_figure(branch_outputs: list[np.ndarray], kernel_sizes: list[int], sfreq: float, title: str) -> plt.Figure:
    fig, axes = plt.subplots(len(branch_outputs), 1, figsize=(14, max(7, len(branch_outputs) * 2.2)), sharex=True)
    if len(branch_outputs) == 1:
        axes = [axes]

    for ax, branch_out, kernel_size in zip(axes, branch_outputs, kernel_sizes):
        activation = np.mean(np.abs(branch_out), axis=0)
        t = np.arange(activation.shape[0], dtype=float) / sfreq
        ax.plot(t, activation, color="#2f8cff", linewidth=1.5)
        ax.fill_between(t, 0.0, activation, color="#32d1ff", alpha=0.25)
        ax.set_ylabel(f"k={kernel_size}")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(Path(args.config_path))
    edf_path = Path(args.edf_path).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else infer_checkpoint_path(args.model)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    run_slug = f"{edf_path.stem}_t{int(round(args.window_start_sec)):04d}"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (PROJECT_ROOT / "outputs" / "interpretability" / args.model / run_slug)
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_window(edf_path, cfg, args.window_start_sec)
    model, checkpoint = load_model(args.model, checkpoint_path)
    saliency, probs, class_idx = compute_saliency(model, prepared.window, args.target_class)

    label_map = {0: "background", 1: "seizure"}
    saliency_title = (
        f"{args.model} saliency | target={label_map.get(class_idx, class_idx)} | "
        f"p(background)={probs[0]:.3f}, p(seizure)={probs[1]:.3f}"
    )
    saliency_fig = make_saliency_figure(
        prepared.window,
        saliency,
        prepared.channel_names,
        prepared.sfreq,
        title=saliency_title,
    )
    saliency_path = out_dir / "saliency_map.png"
    saliency_fig.savefig(saliency_path, dpi=220, bbox_inches="tight")
    plt.close(saliency_fig)

    summary = {
        "model": args.model,
        "edf_path": str(edf_path),
        "checkpoint_path": str(checkpoint_path),
        "window_start_sec": args.window_start_sec,
        "shape": prepared.metadata["shape"],
        "prob_background": float(probs[0]),
        "prob_seizure": float(probs[1]),
        "target_class_index": int(class_idx),
        "target_class_name": label_map.get(class_idx, str(class_idx)),
        "saliency_path": str(saliency_path),
    }

    if args.model == "multiscale_attention_cnn":
        branch_outputs = capture_multiscale_branch_outputs(model, prepared.window)
        kernel_sizes = [3, 7, 15, 31][: len(branch_outputs)]
        if branch_outputs:
            branch_fig = make_multiscale_figure(
                branch_outputs,
                kernel_sizes,
                prepared.sfreq,
                title="Multi-scale branch activations",
            )
            branch_path = out_dir / "multiscale_branch_activations.png"
            branch_fig.savefig(branch_path, dpi=220, bbox_inches="tight")
            plt.close(branch_fig)
            summary["multiscale_branch_path"] = str(branch_path)
            summary["kernel_sizes"] = kernel_sizes

    save_json(out_dir / "summary.json", summary)

    print("=" * 88)
    print(f"Saved saliency visualization : {saliency_path}")
    if "multiscale_branch_path" in summary:
        print(f"Saved branch visualization   : {summary['multiscale_branch_path']}")
    print(f"Saved summary               : {out_dir / 'summary.json'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
