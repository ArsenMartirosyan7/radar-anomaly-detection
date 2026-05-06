from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.metrics import extract_events


ENAC_TYPES = [
    "THETA",
    "RHO",
    "ALL",
    "RND",
    "ROUTE",
    "FL_PLUS",
    "FL_MINUS",
    "CGS_PLUS",
    "CGS_MINUS",
]

DISPLAY_NAMES = {
    "THETA": "THETA",
    "RHO": "RHO",
    "ALL": "ALL",
    "RND": "RND",
    "ROUTE": "ROUTE",
    "FL_PLUS": "FL(+)",
    "FL_MINUS": "FL(-)",
    "CGS_PLUS": "CGS(+)",
    "CGS_MINUS": "CGS(-)",
}

ENAC_AUTOENCODER_BASELINE = {
    "precision": 0.8565,
    "recall": 0.6571,
    "f1": 0.7116,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_metrics(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "ensemble_results.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def get_mean_row(df: pd.DataFrame) -> pd.Series:
    mean_rows = df[df["type"] == "MEAN"]

    if mean_rows.empty:
        raise RuntimeError("No MEAN row found in ensemble_results.csv")

    return mean_rows.iloc[0]


def load_type_metrics(results_dir: Path, anomaly_type: str) -> Dict[str, object]:
    path = results_dir / anomaly_type / "metrics.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_enac_vs_hybrid(df: pd.DataFrame, output_dir: Path) -> None:
    mean_row = get_mean_row(df)

    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1-score"]

    enac_values = [ENAC_AUTOENCODER_BASELINE[m] for m in metrics]
    hybrid_values = [float(mean_row[m]) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, enac_values, width, label="Previous ENAC Autoencoder")
    plt.bar(x + width / 2, hybrid_values, width, label="Hybrid TranAD-AE")

    plt.xticks(x, labels)
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("Mean Performance Comparison: ENAC Baseline vs Hybrid TranAD-AE")
    plt.legend()

    for i, value in enumerate(enac_values):
        plt.text(i - width / 2, value + 0.02, f"{value:.4f}", ha="center")

    for i, value in enumerate(hybrid_values):
        plt.text(i + width / 2, value + 0.02, f"{value:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_dir / "01_enac_vs_hybrid_mean_metrics.png", dpi=150)
    plt.close()


def plot_per_attack_metrics(df: pd.DataFrame, output_dir: Path) -> None:
    data = df[df["type"] != "MEAN"].copy()

    x = np.arange(len(data))
    width = 0.25

    plt.figure(figsize=(16, 7))

    plt.bar(x - width, data["precision"], width, label="Precision")
    plt.bar(x, data["recall"], width, label="Recall")
    plt.bar(x + width, data["f1"], width, label="F1-score")

    plt.xticks(x, data["type"], rotation=45)
    plt.ylim(0, 1.1)
    plt.ylabel("Score")
    plt.title("Hybrid TranAD-AE Performance by ENAC Attack Type")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "02_per_attack_precision_recall_f1.png", dpi=150)
    plt.close()


def plot_per_attack_f1(df: pd.DataFrame, output_dir: Path) -> None:
    data = df[df["type"] != "MEAN"].copy()

    plt.figure(figsize=(14, 6))
    bars = plt.bar(data["type"], data["f1"])

    plt.ylim(0, 1.1)
    plt.ylabel("F1-score")
    plt.title("F1-score by Attack Type")
    plt.xticks(rotation=45)

    for bar, value in zip(bars, data["f1"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.02,
            f"{float(value):.4f}",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "03_per_attack_f1.png", dpi=150)
    plt.close()


def plot_number_of_attacks(results_dir: Path, output_dir: Path) -> None:
    attack_names = []
    true_event_counts = []
    predicted_event_counts = []

    for anomaly_type in ENAC_TYPES:
        metrics = load_type_metrics(results_dir, anomaly_type)

        attack_names.append(DISPLAY_NAMES[anomaly_type])
        true_event_counts.append(int(metrics.get("true_events", 12)))
        predicted_event_counts.append(int(metrics.get("predicted_events", 0)))

    x = np.arange(len(attack_names))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width / 2, true_event_counts, width, label="Injected attacks")
    plt.bar(x + width / 2, predicted_event_counts, width, label="Detected attack events")

    plt.xticks(x, attack_names, rotation=45)
    plt.ylabel("Number of events")
    plt.title("Number of Injected and Detected Attack Events")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_dir / "04_number_of_attacks.png", dpi=150)
    plt.close()


def plot_total_confusion_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    mean_row = get_mean_row(df)

    # In our mean row, tp/fp/fn/tn are actually sums across all attack types.
    tn = int(mean_row["tn"])
    fp = int(mean_row["fp"])
    fn = int(mean_row["fn"])
    tp = int(mean_row["tp"])

    matrix = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix)

    plt.title("Aggregated Confusion Matrix Across All Attack Types")
    plt.xticks([0, 1], ["Predicted Normal", "Predicted Anomaly"])
    plt.yticks([0, 1], ["True Normal", "True Anomaly"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=12)

    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(output_dir / "05_total_confusion_matrix.png", dpi=150)
    plt.close()


def plot_full_timeline(
    results_dir: Path,
    tranad_results_root: Path,
    anomaly_type: str,
    output_dir: Path,
) -> None:
    type_dir = results_dir / anomaly_type

    scores_path = type_dir / "ensemble_scores.npy"
    predictions_path = type_dir / "ensemble_predictions.npy"
    labels_path = tranad_results_root / f"enac_{anomaly_type}" / "test_labels_aligned.npy"
    metrics_path = type_dir / "metrics.json"

    if not scores_path.exists():
        raise FileNotFoundError(scores_path)

    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    scores = np.load(scores_path)
    predictions = np.load(predictions_path).astype(int)
    labels = np.load(labels_path).astype(int)

    metrics = load_type_metrics(results_dir, anomaly_type)
    threshold = float(metrics["threshold"])

    n = min(len(scores), len(predictions), len(labels))
    scores = scores[:n]
    predictions = predictions[:n]
    labels = labels[:n]

    x = np.arange(n)

    y_min = float(np.min(scores))
    y_max = float(np.max(scores))
    padding = 0.05 * max(y_max - y_min, 1.0)
    y_low = y_min - padding
    y_high = y_max + padding

    true_events = extract_events(labels)
    pred_events = extract_events(predictions)

    plt.figure(figsize=(20, 6))
    plt.plot(x, scores, linewidth=0.8, label="Ensemble anomaly score")
    plt.axhline(threshold, linestyle="--", linewidth=1.5, label="Threshold")

    for i, (start, end) in enumerate(true_events):
        label = "True attack interval" if i == 0 else None
        plt.axvspan(start, end, alpha=0.20, label=label)

    for i, (start, end) in enumerate(pred_events):
        label = "Predicted attack interval" if i == 0 else None
        plt.axvspan(start, end, alpha=0.12, label=label)

    plt.ylim(y_low, y_high)
    plt.title(f"Final Ensemble Anomaly Score Timeline - {DISPLAY_NAMES[anomaly_type]}")
    plt.xlabel("Time index")
    plt.ylabel("Normalized ensemble anomaly score")
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(output_dir / f"{anomaly_type}_full_timeline.png", dpi=150)
    plt.close()


def plot_zoomed_events(
    results_dir: Path,
    tranad_results_root: Path,
    anomaly_type: str,
    output_dir: Path,
    padding: int,
    max_events: int,
) -> None:
    type_dir = results_dir / anomaly_type

    scores = np.load(type_dir / "ensemble_scores.npy")
    predictions = np.load(type_dir / "ensemble_predictions.npy").astype(int)
    labels = np.load(
        tranad_results_root / f"enac_{anomaly_type}" / "test_labels_aligned.npy"
    ).astype(int)

    metrics = load_type_metrics(results_dir, anomaly_type)
    threshold = float(metrics["threshold"])

    n = min(len(scores), len(predictions), len(labels))
    scores = scores[:n]
    predictions = predictions[:n]
    labels = labels[:n]

    true_events = extract_events(labels)
    pred_events = extract_events(predictions)

    for event_id, (start, end) in enumerate(true_events[:max_events], start=1):
        left = max(0, start - padding)
        right = min(n, end + padding)

        xs = np.arange(left, right)
        local_scores = scores[left:right]
        local_labels = labels[left:right]
        local_predictions = predictions[left:right]

        y_min = float(np.min(local_scores))
        y_max = float(np.max(local_scores))
        pad = 0.05 * max(y_max - y_min, 1.0)

        plt.figure(figsize=(16, 5))
        plt.plot(xs, local_scores, linewidth=1.0, label="Ensemble anomaly score")
        plt.axhline(threshold, linestyle="--", linewidth=1.5, label="Threshold")

        local_true_events = extract_events(local_labels)
        for i, (s, e) in enumerate(local_true_events):
            label = "True attack interval" if i == 0 else None
            plt.axvspan(left + s, left + e, alpha=0.20, label=label)

        local_pred_events = extract_events(local_predictions)
        for i, (s, e) in enumerate(local_pred_events):
            label = "Predicted attack interval" if i == 0 else None
            plt.axvspan(left + s, left + e, alpha=0.12, label=label)

        plt.ylim(y_min - pad, y_max + pad)
        plt.title(
            f"{DISPLAY_NAMES[anomaly_type]} - Zoom Around Attack #{event_id:02d}"
        )
        plt.xlabel("Time index")
        plt.ylabel("Normalized ensemble anomaly score")
        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.savefig(
            output_dir / f"{anomaly_type}_event_{event_id:02d}_zoom.png",
            dpi=150,
        )
        plt.close()


def create_summary_markdown(df: pd.DataFrame, output_dir: Path) -> None:
    mean_row = get_mean_row(df)

    text = f"""# Final Hybrid TranAD-AE Visualization Summary

## Final Mean Results

| Method | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Previous ENAC Autoencoder | {ENAC_AUTOENCODER_BASELINE["precision"]:.4f} | {ENAC_AUTOENCODER_BASELINE["recall"]:.4f} | {ENAC_AUTOENCODER_BASELINE["f1"]:.4f} |
| Hybrid TranAD-AE | {float(mean_row["precision"]):.4f} | {float(mean_row["recall"]):.4f} | {float(mean_row["f1"]):.4f} |

## Generated Figures

- `01_enac_vs_hybrid_mean_metrics.png`: compares the final hybrid model against the previous ENAC Autoencoder.
- `02_per_attack_precision_recall_f1.png`: shows Precision, Recall, and F1 for each ENAC attack type.
- `03_per_attack_f1.png`: highlights which attack types are easiest or hardest.
- `04_number_of_attacks.png`: compares injected attacks and detected attack events.
- `05_total_confusion_matrix.png`: shows total TP, FP, FN, and TN across all attack types.
- `timelines/`: full anomaly-score timeline for each attack type.
- `zooms/`: zoomed anomaly-score plots around each injected attack interval.

## Main Interpretation

The Hybrid TranAD-AE ensemble combines temporal reconstruction errors from TranAD with point-level reconstruction errors from the Autoencoder. The final model improves over the previous ENAC Autoencoder baseline on timestamp-level Precision, Recall, and F1-score.
"""

    with (output_dir / "README_visualizations.md").open("w", encoding="utf-8") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate meaningful visualizations for the final Hybrid TranAD-AE ensemble."
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/ensemble/final_optimized_strict",
        help="Final ensemble results directory",
    )

    parser.add_argument(
        "--tranad-results-root",
        type=str,
        default="results/tranad",
        help="Root folder containing TranAD ENAC test labels",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ensemble/final_optimized_strict/plots",
        help="Where to save generated plots",
    )

    parser.add_argument(
        "--zoom-padding",
        type=int,
        default=600,
        help="Number of points before/after each attack interval in zoomed plots",
    )

    parser.add_argument(
        "--max-zoom-events",
        type=int,
        default=12,
        help="Maximum number of zoomed events per attack type",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    tranad_results_root = Path(args.tranad_results_root)
    output_dir = Path(args.output_dir)

    summary_dir = output_dir / "summary"
    timelines_dir = output_dir / "timelines"
    zooms_dir = output_dir / "zooms"

    ensure_dir(summary_dir)
    ensure_dir(timelines_dir)
    ensure_dir(zooms_dir)

    df = load_metrics(results_dir)

    print("Generating summary plots...")
    plot_enac_vs_hybrid(df, summary_dir)
    plot_per_attack_metrics(df, summary_dir)
    plot_per_attack_f1(df, summary_dir)
    plot_number_of_attacks(results_dir, summary_dir)
    plot_total_confusion_matrix(df, summary_dir)
    create_summary_markdown(df, output_dir)

    print("Generating timeline and zoomed plots...")

    for anomaly_type in ENAC_TYPES:
        print(f"  Plotting {anomaly_type}...")

        plot_full_timeline(
            results_dir=results_dir,
            tranad_results_root=tranad_results_root,
            anomaly_type=anomaly_type,
            output_dir=timelines_dir,
        )

        plot_zoomed_events(
            results_dir=results_dir,
            tranad_results_root=tranad_results_root,
            anomaly_type=anomaly_type,
            output_dir=zooms_dir,
            padding=args.zoom_padding,
            max_events=args.max_zoom_events,
        )

    print("\nDone.")
    print(f"Plots saved to: {output_dir}")
    print(f"Summary plots: {summary_dir}")
    print(f"Timelines:     {timelines_dir}")
    print(f"Zooms:         {zooms_dir}")


if __name__ == "__main__":
    main()