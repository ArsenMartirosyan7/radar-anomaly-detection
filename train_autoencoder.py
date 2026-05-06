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
from torch.utils.data import DataLoader, TensorDataset

from models.autoencoder import FeedForwardAutoencoder


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


def parse_hidden_dims(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def save_history(path: Path, history: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def train_one_epoch(
    model: FeedForwardAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    mse = nn.MSELoss()
    total_loss = 0.0
    total_count = 0

    for (batch,) in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        reconstruction = model(batch)

        loss = mse(reconstruction, batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.size(0)
        total_count += batch.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate_loss(
    model: FeedForwardAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    mse = nn.MSELoss()
    total_loss = 0.0
    total_count = 0

    for (batch,) in loader:
        batch = batch.to(device)
        reconstruction = model(batch)

        loss = mse(reconstruction, batch)

        total_loss += loss.item() * batch.size(0)
        total_count += batch.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def compute_feature_errors(
    model: FeedForwardAutoencoder,
    data: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    dataset = TensorDataset(torch.from_numpy(data.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_errors = []

    for (batch,) in loader:
        batch = batch.to(device)
        reconstruction = model(batch)

        errors = torch.square(reconstruction - batch)
        all_errors.append(errors.detach().cpu().numpy())

    return np.concatenate(all_errors, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Feed-Forward Autoencoder on RADAR data.")

    parser.add_argument("--data-dir", type=str, default="data/processed/RADAR")
    parser.add_argument("--output-dir", type=str, default="results/autoencoder")

    parser.add_argument("--hidden-dims", type=str, default="128,64,32")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--threshold-quantile", type=float, default=0.999)
    parser.add_argument("--align-window-size", type=int, default=30)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)

    train = np.load(data_dir / "train.npy").astype(np.float32)
    val = np.load(data_dir / "val.npy").astype(np.float32)

    with (data_dir / "feature_names.json").open("r", encoding="utf-8") as f:
        feature_names = json.load(f)

    input_dim = train.shape[1]
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    train_dataset = TensorDataset(torch.from_numpy(train))
    val_dataset = TensorDataset(torch.from_numpy(val))

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

    model = FeedForwardAutoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []

    print(f"Using device: {device}")
    print(f"Train data: {train.shape}")
    print(f"Val data:   {val.shape}")
    print(f"Features:   {input_dim}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
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
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "feature_names": feature_names,
                "hidden_dims": hidden_dims,
                "dropout": args.dropout,
                "best_val_loss": best_val_loss,
                "threshold_quantile": args.threshold_quantile,
                "align_window_size": args.align_window_size,
            }

            torch.save(checkpoint, output_dir / "checkpoint.pt")

    save_history(output_dir / "training_history.csv", history)

    print("\nLoading best checkpoint for validation scoring...")
    checkpoint = torch.load(output_dir / "checkpoint.pt", map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    val_feature_errors_full = compute_feature_errors(
        model=model,
        data=val,
        batch_size=args.batch_size,
        device=device,
    )

    align_start = args.align_window_size - 1
    val_feature_errors_aligned = val_feature_errors_full[align_start:]
    val_scores_aligned = val_feature_errors_aligned.mean(axis=1)

    threshold = float(np.quantile(val_scores_aligned, args.threshold_quantile))

    np.save(output_dir / "val_feature_errors_full.npy", val_feature_errors_full)
    np.save(output_dir / "val_feature_errors_aligned.npy", val_feature_errors_aligned)
    np.save(output_dir / "val_scores_aligned.npy", val_scores_aligned)

    checkpoint["threshold"] = threshold
    torch.save(checkpoint, output_dir / "checkpoint.pt")

    print("\nTraining finished.")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Validation feature errors full:    {val_feature_errors_full.shape}")
    print(f"Validation feature errors aligned: {val_feature_errors_aligned.shape}")
    print(f"Threshold quantile: {args.threshold_quantile}")
    print(f"Threshold: {threshold:.8f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()