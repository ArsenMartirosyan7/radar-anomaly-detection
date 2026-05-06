from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.tranad import TranAD
from utils.dataset import SlidingWindowDataset
from utils.metrics import event_metrics, point_metrics
from utils.postprocessing import moving_average, postprocess_predictions


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


GROUP_KEYWORDS = {
    "THETA": ["theta_deg"],
    "RHO": ["rho_nm"],
    "ROUTE": ["x_nm", "y_nm"],
    "FL_PLUS": ["flight_level"],
    "FL_MINUS": ["flight_level"],
    "CGS_PLUS": ["ground_speed_nm_s"],
    "CGS_MINUS": ["ground_speed_nm_s"],
    "ALL": [
        "theta_deg",
        "rho_nm",
        "x_nm",
        "y_nm",
        "flight_level",
        "ground_speed_nm_s",
        "heading_deg",
    ],
    "RND": [
        "asterix_block_count",
        "packet_count",
        "udp_byte_count",
        "cat001_count",
        "cat002_count",
        "cat034_count",
        "cat048_count",
        "cat048_record_count",
        "unique_tracks",
        "unique_aircraft",
        "theta_deg",
        "rho_nm",
        "x_nm",
        "y_nm",
        "flight_level",
        "ground_speed_nm_s",
        "heading_deg",
    ],
}


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def get_feature_indices(feature_names: List[str], anomaly_type: str) -> List[int]:
    keywords = GROUP_KEYWORDS[anomaly_type]

    indices = [
        i
        for i, name in enumerate(feature_names)
        if any(keyword in name for keyword in keywords)
    ]

    if not indices:
        raise RuntimeError(
            f"No feature indices found for {anomaly_type}. "
            f"Keywords were: {keywords}"
        )

    return indices


@torch.no_grad()
def compute_feature_errors(
    model: TranAD,
    data: np.ndarray,
    window_size: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = SlidingWindowDataset(data=data, window_size=window_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model.eval()
    all_feature_errors = []

    for windows, targets in loader:
        windows = windows.to(device)
        targets = targets.to(device)

        _rec1, rec2 = model(windows)
        feature_errors = torch.square(rec2 - targets)

        all_feature_errors.append(feature_errors.detach().cpu().numpy())

    return np.concatenate(all_feature_errors, axis=0)


def load_model(model_dir: Path, device: torch.device) -> Tuple[TranAD, Dict[str, object]]:
    checkpoint_path = model_dir / "checkpoint.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = TranAD(
        input_dim=int(checkpoint["input_dim"]),
        window_size=int(checkpoint["window_size"]),
        d_model=int(checkpoint["d_model"]),
        nhead=int(checkpoint["nhead"]),
        num_layers=int(checkpoint["num_layers"]),
        dim_feedforward=int(checkpoint["dim_feedforward"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint


def save_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[Dict[str, object]]) -> None:
    headers = [
        "type",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "event_precision",
        "event_recall",
        "event_f1",
        "predicted_events",
    ]

    print("\nENAC feature-group results:")
    print(
        f"{'Type':<9} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'ROC-AUC':>10} "
        f"{'PR-AUC':>10} "
        f"{'Event-P':>10} "
        f"{'Event-R':>10} "
        f"{'Event-F1':>10} "
        f"{'PredEv':>8}"
    )

    for row in rows:
        print(
            f"{row['type']:<9} "
            f"{float(row['precision']):>10.4f} "
            f"{float(row['recall']):>10.4f} "
            f"{float(row['f1']):>10.4f} "
            f"{float(row['roc_auc']):>10.4f} "
            f"{float(row['pr_auc']):>10.4f} "
            f"{float(row['event_precision']):>10.4f} "
            f"{float(row['event_recall']):>10.4f} "
            f"{float(row['event_f1']):>10.4f} "
            f"{int(row['predicted_events']):>8}"
        )


def mean_row(rows: List[Dict[str, object]]) -> Dict[str, object]:
    metric_keys = [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "event_precision",
        "event_recall",
        "event_f1",
    ]

    result: Dict[str, object] = {"type": "MEAN"}

    for key in metric_keys:
        values = [float(row[key]) for row in rows]
        result[key] = float(np.mean(values))

    result["predicted_events"] = int(round(np.mean([int(row["predicted_events"]) for row in rows])))
    result["threshold"] = float("nan")
    result["tp"] = int(sum(int(row["tp"]) for row in rows))
    result["fp"] = int(sum(int(row["fp"]) for row in rows))
    result["fn"] = int(sum(int(row["fn"]) for row in rows))
    result["tn"] = int(sum(int(row["tn"]) for row in rows))

    return result


def evaluate_one_type(
    anomaly_type: str,
    feature_names: List[str],
    val_feature_errors: np.ndarray,
    test_result_dir: Path,
    output_dir: Path,
    smooth_window: int,
    min_event_length: int,
    merge_gap: int,
    threshold_quantile: float,
) -> Dict[str, object]:
    feature_error_path = test_result_dir / "test_feature_errors.npy"
    labels_path = test_result_dir / "test_labels_aligned.npy"

    if not feature_error_path.exists():
        raise FileNotFoundError(
            f"{feature_error_path} not found. "
            f"Run test.py for {anomaly_type} first."
        )

    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    test_feature_errors = np.load(feature_error_path)
    labels = np.load(labels_path).astype(np.int64)

    indices = get_feature_indices(feature_names, anomaly_type)

    val_scores = val_feature_errors[:, indices].mean(axis=1)
    test_scores = test_feature_errors[:, indices].mean(axis=1)

    val_scores = moving_average(val_scores, window=smooth_window)
    test_scores = moving_average(test_scores, window=smooth_window)

    threshold = float(np.quantile(val_scores, threshold_quantile))

    raw_predictions = (test_scores > threshold).astype(np.int64)

    predictions = postprocess_predictions(
        raw_predictions=raw_predictions,
        min_event_length=min_event_length,
        merge_gap=merge_gap,
    )

    raw_metric_values = point_metrics(
        y_true=labels,
        scores=test_scores,
        threshold=threshold,
    )

    post_metric_values = point_metrics(
        y_true=labels,
        scores=predictions.astype(float),
        threshold=0.5,
    )

    event_metric_values = event_metrics(
        y_true=labels,
        y_pred=predictions,
    )

    result = {
        "type": DISPLAY_NAMES[anomaly_type],
        "internal_type": anomaly_type,
        "threshold": threshold,
        "feature_count": len(indices),
        "features_used": ";".join(feature_names[i] for i in indices),
        "precision": post_metric_values["precision"],
        "recall": post_metric_values["recall"],
        "f1": post_metric_values["f1"],
        "roc_auc": raw_metric_values["roc_auc"],
        "pr_auc": raw_metric_values["pr_auc"],
        "tp": post_metric_values["tp"],
        "fp": post_metric_values["fp"],
        "fn": post_metric_values["fn"],
        "tn": post_metric_values["tn"],
        "predicted_anomaly_points": post_metric_values["predicted_anomaly_points"],
        "raw_predicted_anomaly_points": raw_metric_values["predicted_anomaly_points"],
        "true_events": event_metric_values["true_events"],
        "predicted_events": event_metric_values["predicted_events"],
        "detected_true_events": event_metric_values["detected_true_events"],
        "event_precision": event_metric_values["event_precision"],
        "event_recall": event_metric_values["event_recall"],
        "event_f1": event_metric_values["event_f1"],
        "mean_detection_delay": event_metric_values["mean_detection_delay"],
    }

    type_output = output_dir / anomaly_type
    type_output.mkdir(parents=True, exist_ok=True)

    np.save(type_output / "group_scores.npy", test_scores)
    np.save(type_output / "group_predictions.npy", predictions)
    np.save(type_output / "group_raw_predictions.npy", raw_predictions)

    with (type_output / "features_used.json").open("w", encoding="utf-8") as f:
        json.dump([feature_names[i] for i in indices], f, indent=2)

    with (type_output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TranAD using ENAC-style feature-group anomaly scores."
    )

    parser.add_argument("--data-dir", type=str, default="data/processed/RADAR")
    parser.add_argument("--model-dir", type=str, default="results/tranad")
    parser.add_argument("--test-results-root", type=str, default="results/tranad")
    parser.add_argument("--output-dir", type=str, default="results/tranad/enac_group_scores")

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--min-event-length", type=int, default=120)
    parser.add_argument("--merge-gap", type=int, default=60)
    parser.add_argument("--threshold-quantile", type=float, default=0.999)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    test_results_root = Path(args.test_results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    model, checkpoint = load_model(model_dir=model_dir, device=device)

    feature_names = checkpoint["feature_names"]
    window_size = int(checkpoint["window_size"])

    val_path = data_dir / "val.npy"

    if not val_path.exists():
        raise FileNotFoundError(val_path)

    val = np.load(val_path).astype(np.float32)

    print("Computing validation feature errors...")
    print(f"Val data: {val.shape}")
    print(f"Window size: {window_size}")

    start_time = time.time()

    val_feature_errors = compute_feature_errors(
        model=model,
        data=val,
        window_size=window_size,
        batch_size=args.batch_size,
        device=device,
    )

    elapsed = time.time() - start_time

    print(f"Validation feature errors: {val_feature_errors.shape}")
    print(f"Computed in {elapsed:.3f} seconds")

    rows: List[Dict[str, object]] = []

    for anomaly_type in ENAC_TYPES:
        test_result_dir = test_results_root / f"enac_{anomaly_type}"

        print(f"\nEvaluating {anomaly_type} using feature-group score...")
        print(f"Test result directory: {test_result_dir}")

        row = evaluate_one_type(
            anomaly_type=anomaly_type,
            feature_names=feature_names,
            val_feature_errors=val_feature_errors,
            test_result_dir=test_result_dir,
            output_dir=output_dir,
            smooth_window=args.smooth_window,
            min_event_length=args.min_event_length,
            merge_gap=args.merge_gap,
            threshold_quantile=args.threshold_quantile,
        )

        rows.append(row)

        print(
            f"  {DISPLAY_NAMES[anomaly_type]} | "
            f"P={row['precision']:.4f} "
            f"R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} "
            f"Event-F1={row['event_f1']:.4f}"
        )

    mean = mean_row(rows)
    rows_with_mean = rows + [mean]

    save_rows_csv(output_dir / "enac_group_score_results.csv", rows_with_mean)

    print_table(rows_with_mean)

    print(f"\nSaved table to: {output_dir / 'enac_group_score_results.csv'}")


if __name__ == "__main__":
    main()