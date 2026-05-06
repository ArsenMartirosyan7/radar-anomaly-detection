from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.tranad import TranAD
from utils.dataset import SlidingWindowDataset
from utils.metrics import event_metrics, point_metrics, strict_event_metrics
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
        raise RuntimeError(f"No features found for {anomaly_type}")

    return indices


@torch.no_grad()
def compute_tranad_feature_errors(
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
    all_errors = []

    for windows, targets in loader:
        windows = windows.to(device)
        targets = targets.to(device)

        _rec1, rec2 = model(windows)
        errors = torch.square(rec2 - targets)

        all_errors.append(errors.detach().cpu().numpy())

    return np.concatenate(all_errors, axis=0)


def load_tranad_model(model_dir: Path, device: torch.device) -> Tuple[TranAD, Dict[str, object]]:
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


def robust_normalize_from_validation(
    val_scores: np.ndarray,
    test_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    median = float(np.median(val_scores))
    q1 = float(np.quantile(val_scores, 0.25))
    q3 = float(np.quantile(val_scores, 0.75))
    scale = q3 - q1

    if scale < 1e-8:
        scale = float(np.std(val_scores))

    if scale < 1e-8:
        scale = 1.0

    val_norm = (val_scores - median) / scale
    test_norm = (test_scores - median) / scale

    return val_norm, test_norm


def quantile_for_type(
    anomaly_type: str,
    default_quantile: float,
    cgs_quantile: float,
    route_quantile: float,
    theta_quantile: float,
) -> float:
    if anomaly_type in {"CGS_PLUS", "CGS_MINUS"}:
        return cgs_quantile

    if anomaly_type == "ROUTE":
        return route_quantile

    if anomaly_type == "THETA":
        return theta_quantile

    return default_quantile

def save_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        result[key] = float(np.mean([float(row[key]) for row in rows]))

    result["predicted_events"] = int(round(np.mean([int(row["predicted_events"]) for row in rows])))
    result["threshold"] = float("nan")
    result["quantile"] = float("nan")
    result["tp"] = int(sum(int(row["tp"]) for row in rows))
    result["fp"] = int(sum(int(row["fp"]) for row in rows))
    result["fn"] = int(sum(int(row["fn"]) for row in rows))
    result["tn"] = int(sum(int(row["tn"]) for row in rows))

    return result


def print_table(rows: List[Dict[str, object]]) -> None:
    print("\nTranAD + Autoencoder ensemble results:")
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


def evaluate_one_type(
    anomaly_type: str,
    feature_names: List[str],
    tranad_val_errors: np.ndarray,
    ae_val_errors: np.ndarray,
    tranad_test_root: Path,
    ae_test_root: Path,
    output_dir: Path,
    smooth_window: int,
    min_event_length: int,
    merge_gap: int,
    default_quantile: float,
    cgs_quantile: float,
    route_quantile: float,
    theta_quantile: float,
    tranad_weight: float,
) -> Dict[str, object]:
    ae_weight = 1.0 - tranad_weight

    tranad_dir = tranad_test_root / f"enac_{anomaly_type}"
    ae_dir = ae_test_root / f"enac_{anomaly_type}"

    tranad_test_errors = np.load(tranad_dir / "test_feature_errors.npy")
    ae_test_errors = np.load(ae_dir / "test_feature_errors.npy")

    labels = np.load(tranad_dir / "test_labels_aligned.npy").astype(np.int64)
    ae_labels = np.load(ae_dir / "test_labels_aligned.npy").astype(np.int64)

    if len(labels) != len(ae_labels) or not np.array_equal(labels, ae_labels):
        raise RuntimeError(f"Label mismatch for {anomaly_type}")

    indices = get_feature_indices(feature_names, anomaly_type)

    tranad_val_scores = tranad_val_errors[:, indices].mean(axis=1)
    ae_val_scores = ae_val_errors[:, indices].mean(axis=1)

    tranad_test_scores = tranad_test_errors[:, indices].mean(axis=1)
    ae_test_scores = ae_test_errors[:, indices].mean(axis=1)

    tranad_val_scores = moving_average(tranad_val_scores, window=smooth_window)
    ae_val_scores = moving_average(ae_val_scores, window=smooth_window)
    tranad_test_scores = moving_average(tranad_test_scores, window=smooth_window)
    ae_test_scores = moving_average(ae_test_scores, window=smooth_window)

    tranad_val_norm, tranad_test_norm = robust_normalize_from_validation(
        tranad_val_scores,
        tranad_test_scores,
    )

    ae_val_norm, ae_test_norm = robust_normalize_from_validation(
        ae_val_scores,
        ae_test_scores,
    )

    ensemble_val_scores = tranad_weight * tranad_val_norm + ae_weight * ae_val_norm
    ensemble_test_scores = tranad_weight * tranad_test_norm + ae_weight * ae_test_norm

    q = quantile_for_type(
        anomaly_type=anomaly_type,
        default_quantile=default_quantile,
        cgs_quantile=cgs_quantile,
        route_quantile=route_quantile,
        theta_quantile=theta_quantile,
    )

    threshold = float(np.quantile(ensemble_val_scores, q))

    raw_predictions = (ensemble_test_scores > threshold).astype(np.int64)

    predictions = postprocess_predictions(
        raw_predictions=raw_predictions,
        min_event_length=min_event_length,
        merge_gap=merge_gap,
    )

    raw_metrics = point_metrics(
        y_true=labels,
        scores=ensemble_test_scores,
        threshold=threshold,
    )

    post_metrics = point_metrics(
        y_true=labels,
        scores=predictions.astype(float),
        threshold=0.5,
    )

    event_values = event_metrics(
        y_true=labels,
        y_pred=predictions,
    )

    strict_event_values = strict_event_metrics(
        y_true=labels,
        y_pred=predictions,
        iou_threshold=0.5,
    )

    result = {
        "type": DISPLAY_NAMES[anomaly_type],
        "internal_type": anomaly_type,
        "quantile": q,
        "threshold": threshold,
        "feature_count": len(indices),
        "features_used": ";".join(feature_names[i] for i in indices),
        "tranad_weight": tranad_weight,
        "ae_weight": ae_weight,
        "precision": post_metrics["precision"],
        "recall": post_metrics["recall"],
        "f1": post_metrics["f1"],
        "roc_auc": raw_metrics["roc_auc"],
        "pr_auc": raw_metrics["pr_auc"],
        "tp": post_metrics["tp"],
        "fp": post_metrics["fp"],
        "fn": post_metrics["fn"],
        "tn": post_metrics["tn"],
        "predicted_anomaly_points": post_metrics["predicted_anomaly_points"],
        "raw_predicted_anomaly_points": raw_metrics["predicted_anomaly_points"],
        "true_events": event_values["true_events"],
        "predicted_events": event_values["predicted_events"],
        "detected_true_events": event_values["detected_true_events"],
        "event_precision": event_values["event_precision"],
        "event_recall": event_values["event_recall"],
        "event_f1": event_values["event_f1"],
        "mean_detection_delay": event_values["mean_detection_delay"],
        "strict_event_precision": strict_event_values["strict_event_precision"],
        "strict_event_recall": strict_event_values["strict_event_recall"],
        "strict_event_f1": strict_event_values["strict_event_f1"],
        "strict_event_iou_threshold": strict_event_values["strict_event_iou_threshold"],
    }

    type_output = output_dir / anomaly_type
    type_output.mkdir(parents=True, exist_ok=True)

    np.save(type_output / "ensemble_scores.npy", ensemble_test_scores)
    np.save(type_output / "ensemble_predictions.npy", predictions)
    np.save(type_output / "ensemble_raw_predictions.npy", raw_predictions)

    with (type_output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TranAD + Autoencoder ensemble.")

    parser.add_argument("--data-dir", type=str, default="data/processed/RADAR")
    parser.add_argument("--tranad-model-dir", type=str, default="results/tranad")
    parser.add_argument("--ae-model-dir", type=str, default="results/autoencoder")
    parser.add_argument("--tranad-test-root", type=str, default="results/tranad")
    parser.add_argument("--ae-test-root", type=str, default="results/autoencoder")
    parser.add_argument("--output-dir", type=str, default="results/ensemble/tranad_ae")

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--smooth-window", type=int, default=15)
    parser.add_argument("--min-event-length", type=int, default=120)
    parser.add_argument("--merge-gap", type=int, default=60)

    parser.add_argument("--default-quantile", type=float, default=0.999)
    parser.add_argument("--cgs-quantile", type=float, default=0.995)
    parser.add_argument("--route-quantile", type=float, default=0.999)
    parser.add_argument("--theta-quantile", type=float, default=0.999)

    parser.add_argument("--tranad-weight", type=float, default=0.6)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tranad_model_dir = Path(args.tranad_model_dir)
    ae_model_dir = Path(args.ae_model_dir)
    tranad_test_root = Path(args.tranad_test_root)
    ae_test_root = Path(args.ae_test_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    tranad_model, tranad_checkpoint = load_tranad_model(
        model_dir=tranad_model_dir,
        device=device,
    )

    feature_names = tranad_checkpoint["feature_names"]
    window_size = int(tranad_checkpoint["window_size"])

    val = np.load(data_dir / "val.npy").astype(np.float32)

    print("Computing TranAD validation feature errors...")
    start = time.time()

    tranad_val_errors = compute_tranad_feature_errors(
        model=tranad_model,
        data=val,
        window_size=window_size,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"TranAD validation errors: {tranad_val_errors.shape}")
    print(f"Computed in {time.time() - start:.3f} seconds")

    ae_val_errors_path = ae_model_dir / "val_feature_errors_aligned.npy"

    if not ae_val_errors_path.exists():
        raise FileNotFoundError(
            f"{ae_val_errors_path} not found. Run train_autoencoder.py first."
        )

    ae_val_errors = np.load(ae_val_errors_path)

    print(f"AE validation errors: {ae_val_errors.shape}")

    if tranad_val_errors.shape != ae_val_errors.shape:
        raise RuntimeError(
            f"Validation shape mismatch: TranAD {tranad_val_errors.shape}, AE {ae_val_errors.shape}"
        )

    rows: List[Dict[str, object]] = []

    for anomaly_type in ENAC_TYPES:
        print(f"\nEvaluating ensemble for {anomaly_type}...")

        row = evaluate_one_type(
            anomaly_type=anomaly_type,
            feature_names=feature_names,
            tranad_val_errors=tranad_val_errors,
            ae_val_errors=ae_val_errors,
            tranad_test_root=tranad_test_root,
            ae_test_root=ae_test_root,
            output_dir=output_dir,
            smooth_window=args.smooth_window,
            min_event_length=args.min_event_length,
            merge_gap=args.merge_gap,
            default_quantile=args.default_quantile,
            cgs_quantile=args.cgs_quantile,
            route_quantile=args.route_quantile,
            theta_quantile=args.theta_quantile,
            tranad_weight=args.tranad_weight,
)

        rows.append(row)

        print(
            f"  {row['type']} | "
            f"P={row['precision']:.4f} "
            f"R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} "
            f"Event-F1={row['event_f1']:.4f}"
        )

    final_rows = rows + [mean_row(rows)]

    save_rows_csv(output_dir / "ensemble_results.csv", final_rows)
    print_table(final_rows)

    print(f"\nSaved table to: {output_dir / 'ensemble_results.csv'}")


if __name__ == "__main__":
    main()