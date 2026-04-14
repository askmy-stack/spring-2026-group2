from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).parent
SRC_DIR = THIS_DIR.parent
RESULTS_ROOT = SRC_DIR / "results"
MODEL_RESULTS_DIR = RESULTS_ROOT / "model"
MODEL_VIZ_DIR = RESULTS_ROOT / "model_viz"
OUT_DIR = RESULTS_ROOT / "comparison"

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not MODEL_RESULTS_DIR.exists():
        raise FileNotFoundError(f"Model results directory not found: {MODEL_RESULTS_DIR}")

    for model_dir in sorted(MODEL_RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        summary_path = model_dir / "summary.json"
        test_metrics_path = model_dir / "logs" / "test_metrics.json"
        config_path = model_dir / "logs" / "config.json"
        viz_summary_path = MODEL_VIZ_DIR / model_dir.name / "summary.json"

        if not summary_path.exists() or not test_metrics_path.exists():
            continue

        summary = load_json(summary_path)
        test_metrics = load_json(test_metrics_path)
        config = load_json(config_path) if config_path.exists() else {}
        viz_summary = load_json(viz_summary_path) if viz_summary_path.exists() else {}

        row = {
            "model": model_dir.name,
            "best_epoch": summary.get("best_epoch"),
            "best_val_f1": summary.get("best_val_f1"),
            "test_loss": test_metrics.get("loss"),
            "test_accuracy": test_metrics.get("accuracy"),
            "test_precision": test_metrics.get("precision"),
            "test_recall": test_metrics.get("recall"),
            "test_specificity": test_metrics.get("specificity"),
            "test_f1": test_metrics.get("f1"),
            "test_auroc": test_metrics.get("auroc"),
            "train_batch_size": config.get("batch_size"),
            "epochs_configured": config.get("epochs"),
            "trainable_params": viz_summary.get("trainable_params"),
        }
        rows.append(row)

    rows.sort(key=lambda r: (r["test_f1"] if r["test_f1"] is not None else float("-inf")), reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# Model Comparison\n\n")
        if not rows:
            f.write("No completed model results found.\n")
            return

        headers = [
            "model", "best_epoch", "best_val_f1", "test_f1", "test_precision",
            "test_recall", "test_auroc", "test_accuracy", "trainable_params",
        ]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            f.write(
                "| " + " | ".join(
                    fmt(row[h], digits=4 if "params" not in h else 0) for h in headers
                ) + " |\n"
            )


def write_console_summary(rows: list[dict[str, Any]]):
    print("=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    if not rows:
        print("No completed model results found.")
        print("=" * 100)
        return

    for i, row in enumerate(rows, start=1):
        print(
            f"{i:>2}. {row['model']:<28} "
            f"test_f1={fmt(row['test_f1'])}  "
            f"precision={fmt(row['test_precision'])}  "
            f"recall={fmt(row['test_recall'])}  "
            f"auroc={fmt(row['test_auroc'])}  "
            f"best_val_f1={fmt(row['best_val_f1'])}"
        )
    print("=" * 100)


def write_plots(rows: list[dict[str, Any]]):
    if not MATPLOTLIB_AVAILABLE or not rows:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_names = [row["model"] for row in rows]
    metrics = {
        "test_f1": "Test F1",
        "test_auroc": "Test AUROC",
        "test_precision": "Test Precision",
        "test_recall": "Test Recall",
    }

    for key, label in metrics.items():
        values = [row[key] if row[key] is not None else 0.0 for row in rows]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(model_names, values, color="#3b82f6")
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        fig.savefig(OUT_DIR / f"{key}.png", dpi=180)
        plt.close(fig)

    radar_metrics = [
        ("best_val_f1", "Val F1"),
        ("test_f1", "Test F1"),
        ("test_precision", "Precision"),
        ("test_recall", "Recall"),
        ("test_specificity", "Specificity"),
        ("test_auroc", "AUROC"),
    ]
    labels = [label for _, label in radar_metrics]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    color_cycle = plt.cm.tab10(np.linspace(0, 1, max(len(rows), 3)))

    for color, row in zip(color_cycle, rows):
        values = [row[key] if row[key] is not None else 0.0 for key, _ in radar_metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["model"], color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Model Metrics Comparison", pad=24)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    plt.tight_layout()
    fig.savefig(OUT_DIR / "model_metrics_radar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = collect_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "model_comparison.csv", rows)
    write_markdown(OUT_DIR / "model_comparison.md", rows)
    write_plots(rows)
    write_console_summary(rows)
    print(f"Saved comparison outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
