from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.tranad import TranAD
from utils.dataset import SlidingWindowDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(requested)


def load_feature_names(data_dir: Path) -> List[str]:
    path = data_dir / "feature_names.json"

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def train_one_epoch(
    model: TranAD,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()

    mse = nn.MSELoss()
    total_loss = 0.0
    total_count = 0

    for windows, targets in loader:
        windows = windows.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        rec1, rec2 = model(windows)

        # TranAD-style two-stage reconstruction loss.
        # Stage 2 is weighted more because it is the anomaly-focused reconstruction.
        loss1 = mse(rec1, targets)
        loss2 = mse(rec2, targets)
        loss = 0.5 * loss1 + loss2

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_size = windows.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loss(
    model: TranAD,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    mse = nn.MSELoss()
    total_loss = 0.0
    total_count = 0

    for windows, targets in loader:
        windows = windows.to(device)
        targets = targets.to(device)

        rec1, rec2 = model(windows)

        loss1 = mse(rec1, targets)
        loss2 = mse(rec2, targets)
        loss = 0.5 * loss1 + loss2

        batch_size = windows.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def compute_scores(
    model: TranAD,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Returns one anomaly score per sliding window.

    Score:
        mean squared reconstruction error across features
    """
    model.eval()

    all_scores = []

    for windows, targets in loader:
        windows = windows.to(device)
        targets = targets.to(device)

        _rec1, rec2 = model(windows)

        scores = torch.mean(torch.square(rec2 - targets), dim=1)
        all_scores.append(scores.detach().cpu().numpy())

    return np.concatenate(all_scores, axis=0)


def save_history(path: Path, history: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TranAD on RADAR data.")

    parser.add_argument("--data-dir", type=str, default="data/processed/RADAR")
    parser.add_argument("--output-dir", type=str, default="results/tranad")

    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)

    print(f"Using device: {device}")

    train = np.load(data_dir / "train.npy").astype(np.float32)
    val = np.load(data_dir / "val.npy").astype(np.float32)

    feature_names = load_feature_names(data_dir)
    input_dim = train.shape[1]

    if input_dim != len(feature_names):
        raise ValueError(
            f"Input dim {input_dim} does not match feature_names length {len(feature_names)}"
        )

    train_dataset = SlidingWindowDataset(train, window_size=args.window_size)
    val_dataset = SlidingWindowDataset(val, window_size=args.window_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    model = TranAD(
        input_dim=input_dim,
        window_size=args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []

    print(f"Train data: {train.shape}")
    print(f"Val data:   {val.shape}")
    print(f"Features:   {input_dim}")
    print(f"Window:     {args.window_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        val_loss = evaluate_loss(
            model=model,
            loader=val_loader,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "feature_names": feature_names,
                "window_size": args.window_size,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "threshold_quantile": args.threshold_quantile,
                "best_val_loss": best_val_loss,
            }

            torch.save(checkpoint, output_dir / "checkpoint.pt")

    save_history(output_dir / "training_history.csv", history)

    print("\nLoading best checkpoint for threshold calculation...")
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_scores = compute_scores(
        model=model,
        loader=val_loader,
        device=device,
    )

    threshold = float(np.quantile(val_scores, args.threshold_quantile))

    np.save(output_dir / "val_scores.npy", val_scores)

    with (output_dir / "threshold.txt").open("w", encoding="utf-8") as f:
        f.write(str(threshold))

    checkpoint["threshold"] = threshold
    torch.save(checkpoint, output_dir / "checkpoint.pt")

    print("\nTraining finished.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Validation scores: {val_scores.shape}")
    print(f"Threshold quantile: {args.threshold_quantile}")
    print(f"Threshold: {threshold:.8f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()