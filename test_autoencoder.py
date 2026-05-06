from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoder import FeedForwardAutoencoder
from utils.metrics import event_metrics, extract_events, point_metrics
from utils.plotting import plot_anomaly_scores, plot_zoomed_event
from utils.postprocessing import moving_average, postprocess_predictions


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(requested)


@torch.no_grad()
def compute_outputs(
    model: FeedForwardAutoencoder,
    data: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    dataset = TensorDataset(torch.from_numpy(data.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_scores = []
    all_feature_errors = []

    for (batch,) in loader:
        batch = batch.to(device)
        reconstruction = model(batch)

        feature_errors = torch.square(reconstruction - batch)
        scores = torch.mean(feature_errors, dim=1)

        all_scores.append(scores.detach().cpu().numpy())
        all_feature_errors.append(feature_errors.detach().cpu().numpy())

    return (
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_feature_errors, axis=0),
    )


def save_dict_csv(path: Path, data: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])

        for key, value in data.items():
            writer.writerow([key, value])


def save_top_feature_errors(
    path: Path,
    feature_errors: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if labels.sum() > 0:
        mean_errors = feature_errors[labels.astype(bool)].mean(axis=0)
    else:
        mean_errors = feature_errors.mean(axis=0)

    order = np.argsort(mean_errors)[::-1]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "feature", "mean_error"])

        for rank, idx in enumerate(order, start=1):
            writer.writerow([rank, feature_names[idx], float(mean_errors[idx])])


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Feed-Forward Autoencoder on RADAR data.")

    parser.add_argument("--data-dir", type=str, default="data/processed/RADAR")
    parser.add_argument("--model-dir", type=str, default="results/autoencoder")
    parser.add_argument("--output-dir", type=str, default="results/autoencoder/test")

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--min-event-length", type=int, default=120)
    parser.add_argument("--merge-gap", type=int, default=60)
    parser.add_argument("--threshold-quantile", type=float, default=None)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    checkpoint_path = model_dir / "checkpoint.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    test = np.load(data_dir / "test.npy").astype(np.float32)
    labels = np.load(data_dir / "test_labels.npy").astype(np.int64)

    feature_names = checkpoint["feature_names"]
    align_window_size = int(checkpoint.get("align_window_size", 30))
    align_start = align_window_size - 1

    threshold = float(checkpoint["threshold"])

    model = FeedForwardAutoencoder(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=list(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Test data: {test.shape}")
    print(f"Align window size: {align_window_size}")
    print(f"Original threshold: {threshold:.8f}")
    print(f"Smooth window: {args.smooth_window}")
    print(f"Min event length: {args.min_event_length}")
    print(f"Merge gap: {args.merge_gap}")

    start_time = time.time()

    raw_scores_full, feature_errors_full = compute_outputs(
        model=model,
        data=test,
        batch_size=args.batch_size,
        device=device,
    )

    inference_time = time.time() - start_time

    raw_scores = raw_scores_full[align_start:]
    feature_errors = feature_errors_full[align_start:]
    aligned_labels = labels[align_start:]

    processed_scores = moving_average(raw_scores, window=args.smooth_window)

    if args.threshold_quantile is not None:
        val_scores_path = model_dir / "val_scores_aligned.npy"

        if not val_scores_path.exists():
            raise FileNotFoundError(val_scores_path)

        val_scores = np.load(val_scores_path)
        val_scores = moving_average(val_scores, window=args.smooth_window)

        threshold = float(np.quantile(val_scores, args.threshold_quantile))

        print(f"Using new threshold quantile: {args.threshold_quantile}")
        print(f"New threshold: {threshold:.8f}")
    else:
        print(f"Using checkpoint threshold: {threshold:.8f}")

    raw_predictions = (processed_scores > threshold).astype(np.int64)

    predictions = postprocess_predictions(
        raw_predictions=raw_predictions,
        min_event_length=args.min_event_length,
        merge_gap=args.merge_gap,
    )

    raw_metrics = point_metrics(
        y_true=aligned_labels,
        scores=processed_scores,
        threshold=threshold,
    )

    post_metrics = point_metrics(
        y_true=aligned_labels,
        scores=predictions.astype(float),
        threshold=0.5,
    )

    event_values = event_metrics(
        y_true=aligned_labels,
        y_pred=predictions,
    )

    metrics = {
        **post_metrics,
        "raw_threshold_precision": raw_metrics["precision"],
        "raw_threshold_recall": raw_metrics["recall"],
        "raw_threshold_f1": raw_metrics["f1"],
        "raw_threshold_predicted_anomaly_points": raw_metrics["predicted_anomaly_points"],
        "roc_auc": raw_metrics["roc_auc"],
        "pr_auc": raw_metrics["pr_auc"],
        **event_values,
        "inference_time_seconds": float(inference_time),
        "points_per_second": float(len(processed_scores) / inference_time),
        "smooth_window": int(args.smooth_window),
        "min_event_length": int(args.min_event_length),
        "merge_gap": int(args.merge_gap),
    }

    np.save(output_dir / "test_scores_raw.npy", raw_scores)
    np.save(output_dir / "test_scores_processed.npy", processed_scores)
    np.save(output_dir / "test_predictions_raw.npy", raw_predictions)
    np.save(output_dir / "test_predictions.npy", predictions)
    np.save(output_dir / "test_labels_aligned.npy", aligned_labels)
    np.save(output_dir / "test_feature_errors.npy", feature_errors)

    save_dict_csv(output_dir / "metrics.csv", metrics)

    save_top_feature_errors(
        path=output_dir / "top_feature_errors.csv",
        feature_errors=feature_errors,
        labels=aligned_labels,
        feature_names=feature_names,
    )

    plot_anomaly_scores(
        scores=processed_scores,
        labels=aligned_labels,
        predictions=predictions,
        threshold=threshold,
        output_path=output_dir / "anomaly_scores_full.png",
        title="Autoencoder anomaly scores",
    )

    true_events = extract_events(aligned_labels)

    for i, (event_start, event_end) in enumerate(true_events[:12], start=1):
        plot_zoomed_event(
            scores=processed_scores,
            labels=aligned_labels,
            predictions=predictions,
            threshold=threshold,
            start=event_start,
            end=event_end,
            output_path=output_dir / f"anomaly_event_{i:02d}.png",
        )

    print("\nTesting finished.")
    print(f"Scores: {processed_scores.shape}")
    print(f"Aligned labels: {aligned_labels.shape}")
    print(f"Raw predicted anomaly points: {int(raw_predictions.sum())}")
    print(f"Post-processed predicted anomaly points: {int(predictions.sum())}")
    print(f"True anomaly points: {int(aligned_labels.sum())}")
    print(f"Inference time: {inference_time:.3f} seconds")

    print("\nPoint-level metrics after post-processing:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    print(
        f"  TP: {metrics['tp']} | FP: {metrics['fp']} | "
        f"FN: {metrics['fn']} | TN: {metrics['tn']}"
    )

    print("\nEvent-level metrics after post-processing:")
    print(f"  True events:      {metrics['true_events']}")
    print(f"  Predicted events: {metrics['predicted_events']}")
    print(f"  Event precision:  {metrics['event_precision']:.4f}")
    print(f"  Event recall:     {metrics['event_recall']:.4f}")
    print(f"  Event F1:         {metrics['event_f1']:.4f}")
    print(f"  Mean delay:       {metrics['mean_detection_delay']}")

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()