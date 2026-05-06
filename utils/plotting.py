from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_anomaly_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    output_path: Path,
    title: str = "TranAD anomaly scores",
    max_points: Optional[int] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if max_points is not None and len(scores) > max_points:
        scores = scores[:max_points]
        labels = labels[:max_points]
        predictions = predictions[:max_points]

    x = np.arange(len(scores))

    plt.figure(figsize=(18, 6))
    plt.plot(x, scores, linewidth=1, label="anomaly score")
    plt.axhline(threshold, linestyle="--", linewidth=2, label="threshold")

    if labels.sum() > 0:
        plt.fill_between(
            x,
            0,
            scores.max() if scores.max() > 0 else 1,
            where=labels.astype(bool),
            alpha=0.25,
            label="true anomaly",
        )

    if predictions.sum() > 0:
        plt.scatter(
            x[predictions.astype(bool)],
            scores[predictions.astype(bool)],
            s=8,
            label="predicted anomaly",
        )

    plt.title(title)
    plt.xlabel("Time index")
    plt.ylabel("Reconstruction error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_zoomed_event(
    scores: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    start: int,
    end: int,
    output_path: Path,
    padding: int = 300,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    left = max(0, start - padding)
    right = min(len(scores), end + padding)

    xs = np.arange(left, right)
    local_scores = scores[left:right]
    local_labels = labels[left:right]
    local_predictions = predictions[left:right]

    plt.figure(figsize=(16, 5))
    plt.plot(xs, local_scores, linewidth=1, label="anomaly score")
    plt.axhline(threshold, linestyle="--", linewidth=2, label="threshold")

    plt.fill_between(
        xs,
        0,
        local_scores.max() if local_scores.max() > 0 else 1,
        where=local_labels.astype(bool),
        alpha=0.25,
        label="true anomaly",
    )

    if local_predictions.sum() > 0:
        plt.scatter(
            xs[local_predictions.astype(bool)],
            local_scores[local_predictions.astype(bool)],
            s=10,
            label="predicted anomaly",
        )

    plt.title(f"Zoomed anomaly event: {start} to {end}")
    plt.xlabel("Time index")
    plt.ylabel("Reconstruction error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()