from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


NON_FEATURE_COLUMNS = {
    "file",
    "utc_time",
    "epoch_second",
    "label",
    "anomaly_type",
}


def parse_date_from_filename(file_name: str) -> str:
    """
    Example:
        2019-06-23-0207_features.csv -> 2019-06-23
    """
    return file_name[:10]


def split_name_from_date(date_str: str) -> str:
    """
    Your current split:
        Train: 2019-06-23 to 2019-06-26
        Val:   2019-06-27
        Test:  2019-06-28 to 2019-06-29
    """
    if date_str <= "2019-06-26":
        return "train"
    if date_str == "2019-06-27":
        return "val"
    return "test"


def load_decoded_csvs(decoded_dir: Path) -> Dict[str, pd.DataFrame]:
    csv_files = sorted(decoded_dir.glob("*_features.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No *_features.csv files found in {decoded_dir}")

    split_frames: Dict[str, List[pd.DataFrame]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if "epoch_second" not in df.columns:
            raise ValueError(f"{csv_file} does not contain epoch_second column")

        if "file" not in df.columns:
            df["file"] = csv_file.name.replace("_features.csv", ".pcap")

        df = df.sort_values("epoch_second").reset_index(drop=True)

        # Make every PCAP continuous second-by-second.
        # Missing seconds become rows with zero radar activity.
        start_sec = int(df["epoch_second"].min())
        end_sec = int(df["epoch_second"].max())
        full_seconds = pd.DataFrame({"epoch_second": np.arange(start_sec, end_sec + 1)})

        df = full_seconds.merge(df, on="epoch_second", how="left")
        df["file"] = df["file"].fillna(csv_file.name.replace("_features.csv", ".pcap"))

        if "utc_time" in df.columns:
            df["utc_time"] = df["utc_time"].fillna(
                pd.to_datetime(df["epoch_second"], unit="s", utc=True).astype(str)
            )

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if col != "epoch_second":
                df[col] = df[col].fillna(0.0)

        date_str = parse_date_from_filename(csv_file.name)
        split = split_name_from_date(date_str)

        split_frames[split].append(df)

    result: Dict[str, pd.DataFrame] = {}

    for split, frames in split_frames.items():
        if not frames:
            raise RuntimeError(f"No files assigned to split: {split}")

        result[split] = pd.concat(frames, ignore_index=True)
        result[split] = result[split].sort_values(["file", "epoch_second"]).reset_index(drop=True)

    return result


def load_anomaly_intervals(labels_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if labels_path is None or not labels_path.exists():
        return None

    labels = pd.read_csv(labels_path)

    required = {"file", "start_time", "end_time"}
    missing = required - set(labels.columns)

    if missing:
        raise ValueError(f"Labels file is missing columns: {missing}")

    labels["start_epoch"] = pd.to_datetime(labels["start_time"], utc=True).astype("int64") // 10**9
    labels["end_epoch"] = pd.to_datetime(labels["end_time"], utc=True).astype("int64") // 10**9

    if "anomaly_type" not in labels.columns:
        labels["anomaly_type"] = "anomaly"

    return labels


def make_labels(df: pd.DataFrame, intervals: Optional[pd.DataFrame]) -> np.ndarray:
    y = np.zeros(len(df), dtype=np.int64)

    if intervals is None:
        return y

    for _, row in intervals.iterrows():
        file_name = str(row["file"])
        start_epoch = int(row["start_epoch"])
        end_epoch = int(row["end_epoch"])

        if file_name == "*" or file_name.lower() == "all":
            mask = (df["epoch_second"] >= start_epoch) & (df["epoch_second"] <= end_epoch)
        else:
            mask = (
                (df["file"] == file_name)
                & (df["epoch_second"] >= start_epoch)
                & (df["epoch_second"] <= end_epoch)
            )

        y[mask.to_numpy()] = 1

    return y


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cyclical time features.
    This helps the model understand normal daily rhythm without using raw absolute timestamp.
    """
    seconds = df["epoch_second"].to_numpy()

    seconds_in_day = 24 * 60 * 60
    sec_of_day = seconds % seconds_in_day

    df["time_sin"] = np.sin(2 * np.pi * sec_of_day / seconds_in_day)
    df["time_cos"] = np.cos(2 * np.pi * sec_of_day / seconds_in_day)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols = [
        col
        for col in numeric_cols
        if col not in NON_FEATURE_COLUMNS and col != "epoch_second"
    ]

    feature_cols = sorted(feature_cols)

    if not feature_cols:
        raise RuntimeError("No numeric feature columns found")

    return feature_cols


def clean_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def save_split_info(output_dir: Path, split_frames: Dict[str, pd.DataFrame]) -> None:
    info = {}

    for split, df in split_frames.items():
        info[split] = {
            "rows": int(len(df)),
            "files": sorted(df["file"].dropna().unique().tolist()),
            "start_utc": pd.to_datetime(df["epoch_second"].min(), unit="s", utc=True).isoformat(),
            "end_utc": pd.to_datetime(df["epoch_second"].max(), unit="s", utc=True).isoformat(),
        }

    with (output_dir / "split_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build train/val/test NumPy arrays from decoded radar CSV files."
    )

    parser.add_argument(
        "--decoded",
        type=str,
        default="data/decoded",
        help="Directory containing *_features.csv files",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/RADAR",
        help="Output directory for processed dataset",
    )

    parser.add_argument(
        "--labels",
        type=str,
        default="data/labels/anomaly_intervals.csv",
        help="Optional anomaly interval CSV file",
    )

    args = parser.parse_args()

    decoded_dir = Path(args.decoded)
    output_dir = Path(args.output)
    labels_path = Path(args.labels)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading decoded CSV files...")
    split_frames = load_decoded_csvs(decoded_dir)

    for split in split_frames:
        split_frames[split] = add_time_features(split_frames[split])

    intervals = load_anomaly_intervals(labels_path)

    if intervals is None:
        print("WARNING: No anomaly_intervals.csv found.")
        print("test_labels.npy will contain only zeros.")
        print("You can still train the model, but evaluation metrics need real or injected anomalies.")

    feature_cols = get_feature_columns(split_frames["train"])

    print(f"Number of features: {len(feature_cols)}")
    print("Features:")
    for col in feature_cols:
        print(f"  - {col}")

    x_train = clean_matrix(split_frames["train"][feature_cols].to_numpy())
    x_val = clean_matrix(split_frames["val"][feature_cols].to_numpy())
    x_test = clean_matrix(split_frames["test"][feature_cols].to_numpy())

    y_train = make_labels(split_frames["train"], intervals)
    y_val = make_labels(split_frames["val"], intervals)
    y_test = make_labels(split_frames["test"], intervals)

    if y_train.sum() > 0:
        print("WARNING: Train split contains anomalies.")
        print("For unsupervised TranAD, training should usually contain only normal data.")

    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = scaler.transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    np.save(output_dir / "train.npy", x_train)
    np.save(output_dir / "val.npy", x_val)
    np.save(output_dir / "test.npy", x_test)
    np.save(output_dir / "train_labels.npy", y_train)
    np.save(output_dir / "val_labels.npy", y_val)
    np.save(output_dir / "test_labels.npy", y_test)

    with (output_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    joblib.dump(scaler, output_dir / "scaler.pkl")

    save_split_info(output_dir, split_frames)

    print("\nDataset created successfully.")
    print(f"Output: {output_dir}")
    print(f"train.npy: {x_train.shape}, anomalies: {int(y_train.sum())}")
    print(f"val.npy:   {x_val.shape}, anomalies: {int(y_val.sum())}")
    print(f"test.npy:  {x_test.shape}, anomalies: {int(y_test.sum())}")


if __name__ == "__main__":
    main()