from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


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


def load_feature_names(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_indices(feature_names: List[str], keywords: List[str]) -> List[int]:
    indices = []

    for i, name in enumerate(feature_names):
        if any(keyword in name for keyword in keywords):
            indices.append(i)

    return indices


def choose_intervals(
    n: int,
    number: int,
    length: int,
    margin: int,
    seed: int,
) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    intervals: List[Tuple[int, int]] = []

    attempts = 0

    while len(intervals) < number and attempts < 10000:
        attempts += 1

        start = int(rng.integers(margin, n - length - margin))
        end = start + length

        ok = True

        for old_start, old_end in intervals:
            if not (end < old_start - margin or start > old_end + margin):
                ok = False
                break

        if ok:
            intervals.append((start, end))

    if len(intervals) < number:
        raise RuntimeError("Could not generate enough non-overlapping intervals")

    return sorted(intervals)


def inject_shift(
    x: np.ndarray,
    start: int,
    end: int,
    indices: List[int],
    value: float,
) -> None:
    if not indices:
        return

    x[start:end, indices] += value


def inject_route_anomaly(
    x: np.ndarray,
    start: int,
    end: int,
    feature_names: List[str],
    strength: float,
) -> List[str]:
    indices = get_indices(feature_names, ["x_nm", "y_nm"])

    if not indices:
        return []

    length = end - start
    ramp = np.linspace(0.0, strength, length).reshape(-1, 1)
    x[start:end, indices] += ramp

    return [feature_names[i] for i in indices]


def inject_random_anomaly(
    x: np.ndarray,
    start: int,
    end: int,
    feature_names: List[str],
    seed: int,
    strength: float,
) -> List[str]:
    rng = np.random.default_rng(seed)

    candidate_indices = [
        i
        for i, name in enumerate(feature_names)
        if not name.startswith("time_")
    ]

    selected = rng.choice(
        candidate_indices,
        size=max(5, len(candidate_indices) // 4),
        replace=False,
    ).tolist()

    noise = rng.normal(
        loc=0.0,
        scale=strength,
        size=(end - start, len(selected)),
    )

    x[start:end, selected] += noise

    return [feature_names[i] for i in selected]


def inject_all_anomaly(
    x: np.ndarray,
    start: int,
    end: int,
    feature_names: List[str],
    strength: float,
) -> List[str]:
    keywords = [
        "theta_deg",
        "rho_nm",
        "x_nm",
        "y_nm",
        "flight_level",
        "ground_speed_nm_s",
        "heading_deg",
    ]

    indices = get_indices(feature_names, keywords)

    if not indices:
        return []

    x[start:end, indices] += strength

    return [feature_names[i] for i in indices]


def inject_one_type(
    clean_test: np.ndarray,
    feature_names: List[str],
    anomaly_type: str,
    intervals: List[Tuple[int, int]],
    seed: int,
    strength: float,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    x = clean_test.copy()
    y = np.zeros(len(x), dtype=np.int64)
    report: List[Dict[str, object]] = []

    for event_id, (start, end) in enumerate(intervals, start=1):
        if anomaly_type == "THETA":
            indices = get_indices(feature_names, ["theta_deg"])
            inject_shift(x, start, end, indices, strength)
            affected = [feature_names[i] for i in indices]

        elif anomaly_type == "RHO":
            indices = get_indices(feature_names, ["rho_nm"])
            inject_shift(x, start, end, indices, strength)
            affected = [feature_names[i] for i in indices]

        elif anomaly_type == "ALL":
            affected = inject_all_anomaly(
                x=x,
                start=start,
                end=end,
                feature_names=feature_names,
                strength=strength,
            )

        elif anomaly_type == "RND":
            affected = inject_random_anomaly(
                x=x,
                start=start,
                end=end,
                feature_names=feature_names,
                seed=seed + event_id,
                strength=strength,
            )

        elif anomaly_type == "ROUTE":
            affected = inject_route_anomaly(
                x=x,
                start=start,
                end=end,
                feature_names=feature_names,
                strength=strength,
            )

        elif anomaly_type == "FL_PLUS":
            indices = get_indices(feature_names, ["flight_level"])
            inject_shift(x, start, end, indices, strength)
            affected = [feature_names[i] for i in indices]

        elif anomaly_type == "FL_MINUS":
            indices = get_indices(feature_names, ["flight_level"])
            inject_shift(x, start, end, indices, -strength)
            affected = [feature_names[i] for i in indices]

        elif anomaly_type == "CGS_PLUS":
            indices = get_indices(feature_names, ["ground_speed_nm_s"])
            inject_shift(x, start, end, indices, strength)
            affected = [feature_names[i] for i in indices]

        elif anomaly_type == "CGS_MINUS":
            indices = get_indices(feature_names, ["ground_speed_nm_s"])
            inject_shift(x, start, end, indices, -strength)
            affected = [feature_names[i] for i in indices]

        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

        y[start:end] = 1

        report.append(
            {
                "event_id": event_id,
                "type": anomaly_type,
                "start_index": int(start),
                "end_index": int(end),
                "length": int(end - start),
                "affected_features": affected,
            }
        )

    return x, y, report


def copy_common_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    common_files = [
        "train.npy",
        "val.npy",
        "train_labels.npy",
        "val_labels.npy",
        "feature_names.json",
        "scaler.pkl",
        "split_info.json",
    ]

    for file_name in common_files:
        src = source_dir / file_name

        if src.exists():
            shutil.copy2(src, target_dir / file_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create ENAC-style anomaly datasets for RADAR."
    )

    parser.add_argument(
        "--source",
        type=str,
        default="data/processed/RADAR",
        help="Source processed RADAR directory",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/RADAR_ENAC",
        help="Output directory for ENAC-style datasets",
    )

    parser.add_argument("--num-anomalies", type=int, default=12)
    parser.add_argument("--length", type=int, default=300)
    parser.add_argument("--margin", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strength", type=float, default=6.0)

    args = parser.parse_args()

    source_dir = Path(args.source)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    clean_test_path = source_dir / "test_clean.npy"

    if not clean_test_path.exists():
        raise FileNotFoundError(
            f"{clean_test_path} not found. "
            "You need test_clean.npy so ENAC anomalies are injected into the clean test set."
        )

    clean_test = np.load(clean_test_path).astype(np.float32)
    feature_names = load_feature_names(source_dir / "feature_names.json")

    intervals = choose_intervals(
        n=len(clean_test),
        number=args.num_anomalies,
        length=args.length,
        margin=args.margin,
        seed=args.seed,
    )

    print("Creating ENAC-style anomaly datasets...")
    print(f"Source clean test: {clean_test_path}")
    print(f"Output root: {output_root}")
    print(f"Test shape: {clean_test.shape}")
    print(f"Intervals: {len(intervals)}")
    print(f"Anomaly length: {args.length}")
    print(f"Strength: {args.strength}")

    summary = {}

    for anomaly_type in ENAC_TYPES:
        target_dir = output_root / anomaly_type
        copy_common_files(source_dir, target_dir)

        x_test, y_test, report = inject_one_type(
            clean_test=clean_test,
            feature_names=feature_names,
            anomaly_type=anomaly_type,
            intervals=intervals,
            seed=args.seed,
            strength=args.strength,
        )

        np.save(target_dir / "test.npy", x_test)
        np.save(target_dir / "test_labels.npy", y_test)

        with (target_dir / "anomaly_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        summary[anomaly_type] = {
            "test_shape": list(x_test.shape),
            "anomaly_points": int(y_test.sum()),
            "events": len(report),
        }

        print(
            f"  {anomaly_type:9s} -> {target_dir} | "
            f"anomaly points: {int(y_test.sum())}"
        )

    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Created datasets in: {output_root}")


if __name__ == "__main__":
    main()