from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_feature_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def feature_indices(feature_names: List[str], selected: List[str]) -> List[int]:
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    return [name_to_idx[name] for name in selected if name in name_to_idx]


def choose_non_overlapping_intervals(
    n: int,
    number: int,
    length: int,
    margin: int,
    seed: int,
) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    intervals: List[Tuple[int, int]] = []

    max_attempts = 10000
    attempts = 0

    while len(intervals) < number and attempts < max_attempts:
        attempts += 1

        start = int(rng.integers(margin, n - length - margin))
        end = start + length

        overlaps = False
        for old_start, old_end in intervals:
            if not (end < old_start - margin or start > old_end + margin):
                overlaps = True
                break

        if not overlaps:
            intervals.append((start, end))

    if len(intervals) < number:
        raise RuntimeError("Could not create enough non-overlapping anomaly intervals")

    return sorted(intervals)


def inject_message_drop(
    x: np.ndarray,
    start: int,
    end: int,
    feature_names: List[str],
) -> List[str]:
    affected = [
        "packet_count",
        "udp_byte_count",
        "asterix_block_count",
        "cat048_count",
        "cat048_record_count",
        "unique_tracks",
        "unique_aircraft",
    ]

    idxs = feature_indices(feature_names, affected)

    for idx in idxs:
        low_value = np.percentile(x[:, idx], 1) - 2.0
        x[start:end, idx] = low_value

    return [feature_names[i] for i in idxs]


def inject_traffic_surge(
    x: np.ndarray,
    start: int,
    end: int,
    feature_names: List[str],
) -> List[str]:
    affected = [
        "packet_count",
        "udp_byte_count",
        "asterix_block_count",
        "cat048_count",
        "cat048_record_count",
        "unique_tracks",
        "unique_aircraft",
    ]

    idxs = feature_indices(feature_names, affected)

    for idx in idxs:
        high_value = np.percentile(x[:, idx], 99) + 3.0
        x[start:end, idx] = high_value

    return [feature_names[i] for i in idxs]


def inject_position_jump(
    x: np.ndarray,
    start: int,
    end: int,
    feature_names: List[str],
    seed: int,
) -> List[str]:
    rng = np.random.default_rng(seed)

    affected = [
        "x_nm_mean",
        "x_nm_max",
        "x_nm_min",
        "y_nm_mean",
        "y_nm_max",
        "y_nm_min",
        "rho_nm_mean",
        "theta_deg_mean",
    ]

    idxs = feature_indices(feature_names, affected)

    for idx in idxs:
        direction = rng.choice([-1.0, 1.0])
        x[start:end, idx] += direction * 5.0

    return [feature_names[i] for i in idxs]


def inject_velocity_heading_spike(
    x: np.ndarray,
    start: int,
    end: int,
    feature_names: List[str],
    seed: int,
) -> List[str]:
    rng = np.random.default_rng(seed)

    affected = [
        "ground_speed_nm_s_mean",
        "ground_speed_nm_s_max",
        "ground_speed_nm_s_std",
        "heading_deg_mean",
        "heading_deg_std",
    ]

    idxs = feature_indices(feature_names, affected)

    for idx in idxs:
        direction = rng.choice([-1.0, 1.0])
        x[start:end, idx] += direction * 6.0

    return [feature_names[i] for i in idxs]


def inject_frozen_radar_values(
    x: np.ndarray,
    start: int,
    end: int,
    feature_names: List[str],
) -> List[str]:
    affected = [
        "rho_nm_mean",
        "theta_deg_mean",
        "flight_level_mean",
        "ground_speed_nm_s_mean",
        "heading_deg_mean",
        "x_nm_mean",
        "y_nm_mean",
        "unique_tracks",
        "unique_aircraft",
    ]

    idxs = feature_indices(feature_names, affected)

    if not idxs:
        return []

    frozen_values = x[start, idxs].copy()
    x[start:end, idxs] = frozen_values

    return [feature_names[i] for i in idxs]


def inject_noise_burst(
    x: np.ndarray,
    start: int,
    end: int,
    feature_names: List[str],
    seed: int,
) -> List[str]:
    rng = np.random.default_rng(seed)

    affected = [
        "rho_nm_mean",
        "rho_nm_std",
        "theta_deg_mean",
        "theta_deg_std",
        "flight_level_mean",
        "flight_level_std",
        "ground_speed_nm_s_mean",
        "ground_speed_nm_s_std",
        "heading_deg_mean",
        "heading_deg_std",
        "x_nm_mean",
        "x_nm_std",
        "y_nm_mean",
        "y_nm_std",
    ]

    idxs = feature_indices(feature_names, affected)

    if not idxs:
        return []

    noise = rng.normal(loc=0.0, scale=3.0, size=(end - start, len(idxs)))
    x[start:end, idxs] += noise

    return [feature_names[i] for i in idxs]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject controlled synthetic anomalies into RADAR test.npy."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed/RADAR",
        help="Directory containing train.npy, val.npy, test.npy, feature_names.json",
    )

    parser.add_argument(
        "--num-anomalies",
        type=int,
        default=12,
        help="Number of anomaly intervals to inject",
    )

    parser.add_argument(
        "--length",
        type=int,
        default=300,
        help="Length of each anomaly interval in timesteps. With 1-second data, 300 = 5 minutes.",
    )

    parser.add_argument(
        "--margin",
        type=int,
        default=900,
        help="Minimum margin between anomaly intervals.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    test_path = data_dir / "test.npy"
    labels_path = data_dir / "test_labels.npy"
    feature_path = data_dir / "feature_names.json"

    if not test_path.exists():
        raise FileNotFoundError(test_path)

    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    x_original = np.load(test_path).astype(np.float32)
    feature_names = load_feature_names(feature_path)

    if x_original.ndim != 2:
        raise ValueError(f"Expected test.npy to have shape (T, F), got {x_original.shape}")

    n, num_features = x_original.shape

    if num_features != len(feature_names):
        raise ValueError(
            f"Feature mismatch: test has {num_features} columns but feature_names has {len(feature_names)}"
        )

    if n < args.length * args.num_anomalies:
        raise ValueError("Test set is too small for requested anomaly intervals")

    clean_backup_path = data_dir / "test_clean.npy"
    clean_labels_backup_path = data_dir / "test_labels_clean.npy"

    if not clean_backup_path.exists():
        np.save(clean_backup_path, x_original)

    if labels_path.exists() and not clean_labels_backup_path.exists():
        np.save(clean_labels_backup_path, np.load(labels_path))

    x = x_original.copy()
    y = np.zeros(n, dtype=np.int64)

    intervals = choose_non_overlapping_intervals(
        n=n,
        number=args.num_anomalies,
        length=args.length,
        margin=args.margin,
        seed=args.seed,
    )

    anomaly_functions = [
        ("message_drop", inject_message_drop),
        ("traffic_surge", inject_traffic_surge),
        ("position_jump", inject_position_jump),
        ("velocity_heading_spike", inject_velocity_heading_spike),
        ("frozen_radar_values", inject_frozen_radar_values),
        ("noise_burst", inject_noise_burst),
    ]

    report: List[Dict[str, object]] = []

    for i, (start, end) in enumerate(intervals):
        anomaly_name, func = anomaly_functions[i % len(anomaly_functions)]

        if anomaly_name in {"position_jump", "velocity_heading_spike", "noise_burst"}:
            affected_features = func(x, start, end, feature_names, args.seed + i)
        else:
            affected_features = func(x, start, end, feature_names)

        y[start:end] = 1

        report.append(
            {
                "id": i + 1,
                "type": anomaly_name,
                "start_index": int(start),
                "end_index": int(end),
                "length": int(end - start),
                "affected_features": affected_features,
            }
        )

    np.save(test_path, x)
    np.save(labels_path, y)

    with (data_dir / "anomaly_injection_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Synthetic anomalies injected successfully.")
    print(f"Data directory: {data_dir}")
    print(f"Original clean test backup: {clean_backup_path}")
    print(f"New test.npy shape: {x.shape}")
    print(f"New test_labels.npy shape: {y.shape}")
    print(f"Anomalous points: {int(y.sum())}")
    print(f"Anomaly intervals: {len(report)}")
    print("\nInjected anomalies:")
    for item in report:
        print(
            f"  #{item['id']:02d} {item['type']}: "
            f"{item['start_index']} -> {item['end_index']} "
            f"({item['length']} points)"
        )


if __name__ == "__main__":
    main()