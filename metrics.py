import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Tuple[int, int]:
    pred = logits.argmax(dim=1)
    return int((pred == y).sum().item()), int(y.numel())


def confusion_matrix_np(y_true: Iterable[int], y_pred: Iterable[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1)
    out = np.zeros(cm.shape[0], dtype=np.float64)
    valid = denom > 0
    out[valid] = np.diag(cm)[valid] / denom[valid]
    return out


def save_confusion_outputs(cm: np.ndarray, out_dir: Path, prefix: str) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_csv = out_dir / f"{prefix}_confusion_matrix.csv"
    per_class_csv = out_dir / f"{prefix}_per_class_acc.csv"
    cm_png = out_dir / f"{prefix}_confusion_matrix.png"

    with cm_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *range(cm.shape[1])])
        for i, row in enumerate(cm):
            writer.writerow([i, *row.tolist()])

    acc = per_class_accuracy(cm)
    with per_class_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "correct", "total", "accuracy"])
        for i in range(cm.shape[0]):
            writer.writerow([i, int(cm[i, i]), int(cm[i].sum()), float(acc[i])])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted transmitter")
    ax.set_ylabel("True transmitter")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(cm_png, dpi=160)
    plt.close(fig)

    return {"confusion_csv": cm_csv, "per_class_csv": per_class_csv, "confusion_png": cm_png}

