from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = ["f1", "precision", "recall", "specificity", "accuracy", "auroc", "auprc"]
PLOT_METRICS = ["f1", "precision", "recall", "specificity", "accuracy"]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_records(metrics_dir: Path, dataset: str | None) -> list[dict]:
    records: list[dict] = []
    for p in sorted(metrics_dir.glob("*.json")):
        try:
            obj = _load_json(p)
            if "final_test" not in obj or "model" not in obj:
                continue
            if dataset is not None and obj.get("dataset") != dataset:
                continue
            rec = {"model": obj["model"], "dataset": obj.get("dataset", "unknown"), "path": str(p)}
            for k in METRICS:
                v = obj["final_test"].get(k, float("nan"))
                rec[k] = float(v) if v is not None else float("nan")
            records.append(rec)
        except Exception:
            continue
    return records


def _best_by_model(records: list[dict], key: str = "f1") -> list[dict]:
    by_model: dict[str, dict] = {}
    for r in records:
        m = r["model"]
        score = r.get(key, float("nan"))
        prev = by_model.get(m)
        if prev is None:
            by_model[m] = r
            continue
        prev_score = prev.get(key, float("nan"))
        if (not math.isnan(score)) and (math.isnan(prev_score) or score > prev_score):
            by_model[m] = r
    return list(by_model.values())


def _bar_plot(records: list[dict], out_path: Path) -> None:
    models = [r["model"] for r in records]
    f1 = [r["f1"] for r in records]
    precision = [r["precision"] for r in records]
    recall = [r["recall"] for r in records]
    specificity = [r["specificity"] for r in records]
    accuracy = [r["accuracy"] for r in records]

    x = np.arange(len(models))
    w = 0.16
    plt.figure(figsize=(13, 6))
    plt.bar(x - 2 * w, f1, width=w, label="F1")
    plt.bar(x - w, precision, width=w, label="Precision")
    plt.bar(x, recall, width=w, label="Recall")
    plt.bar(x + w, specificity, width=w, label="Specificity")
    plt.bar(x + 2 * w, accuracy, width=w, label="Accuracy")
    plt.xticks(x, models, rotation=25, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Model Comparison (best run per model)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _radar_plot(records: list[dict], out_path: Path) -> None:
    radar_metrics = PLOT_METRICS
    labels = [m.upper() for m in radar_metrics]
    angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)

    for r in records:
        vals = [r.get(m, float("nan")) for m in radar_metrics]
        vals = [0.0 if math.isnan(v) else max(0.0, min(1.0, v)) for v in vals]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=r["model"])
        ax.fill(angles, vals, alpha=0.08)

    ax.set_title("Model Performance Radar (best run per model)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_dir", type=str, default="model/results/benchmark_metrics")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="model/results/benchmark_plots")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _collect_records(metrics_dir=metrics_dir, dataset=args.dataset)
    if not records:
        raise SystemExit(f"No metrics JSON files found in: {metrics_dir}")

    best = _best_by_model(records, key="f1")
    best = sorted(best, key=lambda r: r["f1"], reverse=True)

    bar_path = out_dir / "model_compare_bar.png"
    radar_path = out_dir / "model_compare_radar.png"
    _bar_plot(best, bar_path)
    _radar_plot(best, radar_path)

    print("Saved plots:")
    print(f"  {bar_path}")
    print(f"  {radar_path}")
    print("Models included:")
    for r in best:
        print(
            f"  {r['model']}: f1={r['f1']:.4f} auroc={r['auroc']:.4f} "
            f"auprc={r['auprc']:.4f} from {r['path']}"
        )


if __name__ == "__main__":
    main()
