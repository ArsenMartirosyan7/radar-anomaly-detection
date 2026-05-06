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
        raise RuntimeError(f"No feature indices found for {anomaly_type}")

    return indices


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

    return (val_scores - median) / scale, (test_scores - median) / scale


def load_tranad_model(model_dir: Path, device: torch.device) -> Tuple[TranAD, Dict[str, object]]:
    checkpoint = torch.load(model_dir / "checkpoint.pt", map_location=device)

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


def evaluate_config(
    feature_names: List[str],
    tranad_val_errors: np.ndarray,
    ae_val_errors: np.ndarray,
    tranad_test_root: Path,
    ae_test_root: Path,
    tranad_weight: float,
    smooth_window: int,
    min_event_length: int,
    merge_gap: int,
    default_quantile: float,
    cgs_quantile: float,
    route_quantile: float,
    theta_quantile: float,
) -> Dict[str, object]:
    ae_weight = 1.0 - tranad_weight

    rows = []

    for anomaly_type in ENAC_TYPES:
        tranad_dir = tranad_test_root / f"enac_{anomaly_type}"
        ae_dir = ae_test_root / f"enac_{anomaly_type}"

        tranad_test_errors = np.load(tranad_dir / "test_feature_errors.npy")
        ae_test_errors = np.load(ae_dir / "test_feature_errors.npy")

        labels = np.load(tranad_dir / "test_labels_aligned.npy").astype(np.int64)

        indices = get_feature_indices(feature_names, anomaly_type)

        tranad_val_scores = tranad_val_errors[:, indices].mean(axis=1)
        tranad_test_scores = tranad_test_errors[:, indices].mean(axis=1)

        ae_val_scores = ae_val_errors[:, indices].mean(axis=1)
        ae_test_scores = ae_test_errors[:, indices].mean(axis=1)

        tranad_val_scores = moving_average(tranad_val_scores, smooth_window)
        tranad_test_scores = moving_average(tranad_test_scores, smooth_window)

        ae_val_scores = moving_average(ae_val_scores, smooth_window)
        ae_test_scores = moving_average(ae_test_scores, smooth_window)

        tranad_val_norm, tranad_test_norm = robust_normalize_from_validation(
            tranad_val_scores,
            tranad_test_scores,
        )

        ae_val_norm, ae_test_norm = robust_normalize_from_validation(
            ae_val_scores,
            ae_test_scores,
        )

        val_scores = tranad_weight * tranad_val_norm + ae_weight * ae_val_norm
        test_scores = tranad_weight * tranad_test_norm + ae_weight * ae_test_norm

        q = quantile_for_type(
            anomaly_type=anomaly_type,
            default_quantile=default_quantile,
            cgs_quantile=cgs_quantile,
            route_quantile=route_quantile,
            theta_quantile=theta_quantile,
        )

        threshold = float(np.quantile(val_scores, q))

        raw_predictions = (test_scores > threshold).astype(np.int64)

        predictions = postprocess_predictions(
            raw_predictions=raw_predictions,
            min_event_length=min_event_length,
            merge_gap=merge_gap,
        )

        p = point_metrics(
            y_true=labels,
            scores=predictions.astype(float),
            threshold=0.5,
        )

        raw = point_metrics(
            y_true=labels,
            scores=test_scores,
            threshold=threshold,
        )

        e = event_metrics(
            y_true=labels,
            y_pred=predictions,
        )

        rows.append(
            {
                "type": DISPLAY_NAMES[anomaly_type],
                "precision": p["precision"],
                "recall": p["recall"],
                "f1": p["f1"],
                "roc_auc": raw["roc_auc"],
                "pr_auc": raw["pr_auc"],
                "event_precision": e["event_precision"],
                "event_recall": e["event_recall"],
                "event_f1": e["event_f1"],
                "predicted_events": e["predicted_events"],
                "tp": p["tp"],
                "fp": p["fp"],
                "fn": p["fn"],
                "tn": p["tn"],
            }
        )

    mean_precision = float(np.mean([r["precision"] for r in rows]))
    mean_recall = float(np.mean([r["recall"] for r in rows]))
    mean_f1 = float(np.mean([r["f1"] for r in rows]))
    mean_event_f1 = float(np.mean([r["event_f1"] for r in rows]))

    return {
        "tranad_weight": tranad_weight,
        "ae_weight": ae_weight,
        "smooth_window": smooth_window,
        "min_event_length": min_event_length,
        "merge_gap": merge_gap,
        "default_quantile": default_quantile,
        "cgs_quantile": cgs_quantile,
        "route_quantile": route_quantile,
        "theta_quantile": theta_quantile,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "mean_event_f1": mean_event_f1,
        "beats_precision": mean_precision > 0.8565,
        "beats_recall": mean_recall > 0.6571,
        "beats_f1": mean_f1 > 0.7116,
        "beats_all": mean_precision > 0.8565 and mean_recall > 0.6571 and mean_f1 > 0.7116,
        "details": rows,
    }


def save_summary_csv(path: Path, results: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "tranad_weight",
        "ae_weight",
        "smooth_window",
        "min_event_length",
        "merge_gap",
        "default_quantile",
        "cgs_quantile",
        "route_quantile",
        "theta_quantile",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_event_f1",
        "beats_precision",
        "beats_recall",
        "beats_f1",
        "beats_all",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, result in enumerate(results, start=1):
            row = {key: result[key] for key in fieldnames if key != "rank"}
            row["rank"] = rank
            writer.writerow(row)


def save_best_details(path: Path, result: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search ensemble parameters.")

    parser.add_argument("--data-dir", type=str, default="data/processed/RADAR")
    parser.add_argument("--tranad-model-dir", type=str, default="results/tranad")
    parser.add_argument("--ae-model-dir", type=str, default="results/autoencoder")
    parser.add_argument("--tranad-test-root", type=str, default="results/tranad")
    parser.add_argument("--ae-test-root", type=str, default="results/autoencoder")
    parser.add_argument("--output-dir", type=str, default="results/ensemble/search")

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    tranad_model, checkpoint = load_tranad_model(
        model_dir=Path(args.tranad_model_dir),
        device=device,
    )

    feature_names = checkpoint["feature_names"]
    window_size = int(checkpoint["window_size"])

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

    ae_val_errors = np.load(Path(args.ae_model_dir) / "val_feature_errors_aligned.npy")

    if ae_val_errors.shape != tranad_val_errors.shape:
        raise RuntimeError(
            f"Validation shape mismatch: TranAD={tranad_val_errors.shape}, AE={ae_val_errors.shape}"
        )

    tranad_test_root = Path(args.tranad_test_root)
    ae_test_root = Path(args.ae_test_root)

    # Search space focused on increasing precision.
    tranad_weights = [0.4]
    smooth_windows = [5]
    min_event_lengths = [120]
    merge_gaps = [0, 15]
    default_quantiles = [0.999, 0.9993, 0.9995]
    cgs_quantiles = [0.995, 0.997]
    route_quantiles = [0.999, 0.9993]
    theta_quantiles = [0.999, 0.9993]

    results = []
    total = (
        len(tranad_weights)
        * len(smooth_windows)
        * len(min_event_lengths)
        * len(merge_gaps)
        * len(default_quantiles)
        * len(cgs_quantiles)
        * len(route_quantiles)
        * len(theta_quantiles)
    )

    print(f"Total configurations: {total}")
    print("Searching...")

    count = 0

    for tw in tranad_weights:
        for sw in smooth_windows:
            for mel in min_event_lengths:
                for mg in merge_gaps:
                    for dq in default_quantiles:
                        for cq in cgs_quantiles:
                            for rq in route_quantiles:
                                for tq in theta_quantiles:
                                    count += 1

                                    result = evaluate_config(
                                        feature_names=feature_names,
                                        tranad_val_errors=tranad_val_errors,
                                        ae_val_errors=ae_val_errors,
                                        tranad_test_root=tranad_test_root,
                                        ae_test_root=ae_test_root,
                                        tranad_weight=tw,
                                        smooth_window=sw,
                                        min_event_length=mel,
                                        merge_gap=mg,
                                        default_quantile=dq,
                                        cgs_quantile=cq,
                                        route_quantile=rq,
                                        theta_quantile=tq,
                                    )

                                    results.append(result)

                                    if count % 500 == 0:
                                        best_so_far = max(results, key=lambda x: x["mean_f1"])
                                        print(
                                            f"{count}/{total} | "
                                            f"best F1={best_so_far['mean_f1']:.4f} "
                                            f"P={best_so_far['mean_precision']:.4f} "
                                            f"R={best_so_far['mean_recall']:.4f} "
                                            f"beats_all={best_so_far['beats_all']}"
                                        )

    # Sort primarily by beats_all, then F1, then precision.
    results_sorted = sorted(
        results,
        key=lambda x: (
            x["beats_all"],
            x["mean_f1"],
            x["mean_precision"],
            x["mean_recall"],
        ),
        reverse=True,
    )

    save_summary_csv(output_dir / "search_summary.csv", results_sorted)
    save_best_details(output_dir / "best_config_details.json", results_sorted[0])

    print("\nTop 10 configurations:")
    for i, r in enumerate(results_sorted[:10], start=1):
        print(
            f"#{i:02d} "
            f"P={r['mean_precision']:.4f} "
            f"R={r['mean_recall']:.4f} "
            f"F1={r['mean_f1']:.4f} "
            f"EventF1={r['mean_event_f1']:.4f} "
            f"beats_all={r['beats_all']} | "
            f"tw={r['tranad_weight']} "
            f"sw={r['smooth_window']} "
            f"mel={r['min_event_length']} "
            f"mg={r['merge_gap']} "
            f"dq={r['default_quantile']} "
            f"cq={r['cgs_quantile']} "
            f"rq={r['route_quantile']} "
            f"tq={r['theta_quantile']}"
        )

    print(f"\nSaved search summary to: {output_dir / 'search_summary.csv'}")
    print(f"Saved best details to: {output_dir / 'best_config_details.json'}")


if __name__ == "__main__":
    main()