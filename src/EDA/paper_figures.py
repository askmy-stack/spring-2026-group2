from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"


# Classical model metrics supplied by the user.
CLASSICAL_MODELS = [
    {
        "model": "LightGBM (baseline)",
        "family": "classical",
        "aucpr": 0.3144,
        "auroc": 0.9254,
        "f1": 0.1751,
        "precision": 0.1095,
        "recall": 0.4369,
        "threshold": 0.69,
    },
    {
        "model": "XGBoost (baseline)",
        "family": "classical",
        "aucpr": 0.3307,
        "auroc": 0.9113,
        "f1": 0.2139,
        "precision": 0.1427,
        "recall": 0.4263,
        "threshold": 0.54,
    },
    {
        "model": "Random Forest (baseline)",
        "family": "classical",
        "aucpr": 0.3307,
        "auroc": 0.8435,
        "f1": 0.1992,
        "precision": 0.1301,
        "recall": 0.4243,
        "threshold": 0.09,
    },
    {
        "model": "LightGBM (tuned)",
        "family": "classical",
        "aucpr": 0.2299,
        "auroc": 0.9326,
        "f1": 0.1947,
        "precision": 0.1318,
        "recall": 0.3725,
        "threshold": 0.66,
    },
    {
        "model": "XGBoost (tuned)",
        "family": "classical",
        "aucpr": 0.2510,
        "auroc": 0.9270,
        "f1": 0.1921,
        "precision": 0.1294,
        "recall": 0.3725,
        "threshold": 0.73,
    },
    {
        "model": "Random Forest (tuned)",
        "family": "classical",
        "aucpr": 0.3573,
        "auroc": 0.8825,
        "f1": 0.1516,
        "precision": 0.0906,
        "recall": 0.4641,
        "threshold": 0.12,
    },
]


LSTM_MODELS = [
    {
        "model": "Attention-BiLSTM",
        "family": "lstm",
        "aucpr": None,
        "auroc": 0.6406,
        "f1": 0.3476,
        "precision": None,
        "recall": 0.2731,
        "threshold": None,
    },
    {
        "model": "CNN-LSTM",
        "family": "lstm",
        "aucpr": None,
        "auroc": 0.7118,
        "f1": 0.5177,
        "precision": None,
        "recall": 0.5692,
        "threshold": None,
    },
]


DEEP_MODEL_ORDER = [
    "deepconvnet",
    "st_eegformer",
    "multiscale_attention_cnn",
    "enhanced_cnn_1d",
]


DISPLAY_NAMES = {
    "deepconvnet": "DeepConvNet",
    "st_eegformer": "ST-EEGFormer",
    "multiscale_attention_cnn": "Multiscale Attention CNN",
    "enhanced_cnn_1d": "Enhanced 1D CNN",
}


# Final comparison set requested by the user.
MODEL_COMPARISON_ORDER = [
    "Random Forest (baseline)",
    "LightGBM (tuned)",
    "XGBoost (baseline)",
    "Attention-BiLSTM",
    "CNN-LSTM",
    "Multiscale Attention CNN",
    "Enhanced 1D CNN",
]


def load_deep_models() -> list[dict]:
    rows: list[dict] = []
    for model_name in DEEP_MODEL_ORDER:
        metrics_path = RESULTS_DIR / model_name / "test_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        rows.append(
            {
                "model": DISPLAY_NAMES.get(model_name, model_name),
                "family": "deep",
                "aucpr": None,
                "auroc": float(metrics["auroc"]),
                "f1": float(metrics["f1"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "threshold": float(metrics["threshold"]),
            }
        )
    return rows


def save_model_comparison(models: list[dict]) -> Path:
    selected = [row for row in models if row["model"] in MODEL_COMPARISON_ORDER]
    selected.sort(key=lambda row: MODEL_COMPARISON_ORDER.index(row["model"]))

    labels = [row["model"] for row in selected]
    f1_vals = np.array([row["f1"] for row in selected], dtype=float)
    auroc_vals = np.array([row["auroc"] for row in selected], dtype=float)

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, f1_vals, width, label="F1", color="#1f77b4")
    ax.bar(x + width / 2, auroc_vals, width, label="ROC-AUC", color="#ff7f0e")

    for idx, value in enumerate(f1_vals):
        ax.text(idx - width / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(auroc_vals):
        ax.text(idx + width / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison Across Selected Classical and CNN Models")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "figure5_model_comparison.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_threshold_operating_points(models: list[dict]) -> Path:
    # This is not a true threshold sweep curve for every model because only
    # selected operating points are currently available in the saved artifacts.
    point_groups = {
        "Random Forest": [
            next(row for row in models if row["model"] == "Random Forest (baseline)"),
            next(row for row in models if row["model"] == "Random Forest (tuned)"),
        ],
        "XGBoost": [
            next(row for row in models if row["model"] == "XGBoost (baseline)"),
            next(row for row in models if row["model"] == "XGBoost (tuned)"),
        ],
        "Enhanced CNN": [
            next(row for row in models if row["model"] == "Enhanced 1D CNN"),
        ],
    }

    colors = {
        "Random Forest": "#2ca02c",
        "XGBoost": "#d62728",
        "Enhanced CNN": "#1f77b4",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, rows in point_groups.items():
        thresholds = [row["threshold"] for row in rows]
        f1_scores = [row["f1"] for row in rows]
        ax.plot(thresholds, f1_scores, marker="o", linewidth=2, color=colors[name], label=name)
        for row in rows:
            ax.text(
                row["threshold"],
                row["f1"] + 0.008,
                f"{row['threshold']:.2f}, {row['f1']:.3f}",
                fontsize=8,
                ha="center",
                color=colors[name],
            )

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.6, label="Default threshold = 0.50")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("Threshold vs F1 Using Available Operating Points")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, max(row["f1"] for rows in point_groups.values() for row in rows) + 0.1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "figure4_threshold_vs_f1.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_models = CLASSICAL_MODELS + LSTM_MODELS + load_deep_models()

    fig4_path = save_threshold_operating_points(all_models)
    fig5_path = save_model_comparison(all_models)

    print(f"Saved Figure 4 -> {fig4_path}")
    print(f"Saved Figure 5 -> {fig5_path}")


if __name__ == "__main__":
    main()
