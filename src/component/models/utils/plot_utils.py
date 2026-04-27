from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: str | Path,
    title: str,
) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    aucpr = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AUCPR = {aucpr:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: str | Path,
    title: str,
) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_threshold_plot(
    threshold_rows: list[dict],
    out_path: str | Path,
    title: str,
) -> None:
    thresholds = [row["threshold"] for row in threshold_rows]
    f1_scores = [row["f1"] for row in threshold_rows]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_confusion_matrix_plot(
    cm: list[list[int]],
    out_path: str | Path,
    title: str,
) -> None:
    cm_arr = np.array(cm)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm_arr)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            plt.text(j, i, str(cm_arr[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_feature_importance_plot(
    feature_names: list[str],
    importances: np.ndarray,
    out_path: str | Path,
    title: str,
    top_k: int = 25,
) -> None:
    if importances is None or len(importances) == 0:
        return

    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_k]
    names = [p[0] for p in pairs][::-1]
    values = [p[1] for p in pairs][::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(names, values)
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_model_comparison_bar(
    labels: list[str],
    values: list[float],
    out_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
