"""
metrics.py
Computes and saves all required evaluation metrics.
Per-fold breakdown + aggregate summary.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc":       roc_auc_score(y_true, y_prob),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
        "tp": int(confusion_matrix(y_true, y_pred, labels=[0,1])[1,1]),
        "fn": int(confusion_matrix(y_true, y_pred, labels=[0,1])[1,0]),
        "fp": int(confusion_matrix(y_true, y_pred, labels=[0,1])[0,1]),
        "tn": int(confusion_matrix(y_true, y_pred, labels=[0,1])[0,0]),
    }


def save_roc_curve(y_true, y_prob, output_path: str, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0,1],[0,1],"k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred, output_path: str, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-resp", "Resp"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_metrics_csv(fold_metrics: list[dict], summary: dict,
                     output_path: str):
    rows = []
    for i, fm in enumerate(fold_metrics):
        row = {"fold": i + 1}
        row.update(fm)
        rows.append(row)
    summary_row = {"fold": "SUMMARY"}
    summary_row.update(summary)
    rows.append(summary_row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[metrics] Saved → {output_path}")
