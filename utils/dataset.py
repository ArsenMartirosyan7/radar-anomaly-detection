from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """
    Converts a time-series matrix X with shape (T, F) into sliding windows.

    Each sample:
        window = X[i : i + window_size]
        target = X[i + window_size - 1]

    The model learns to reconstruct the last point of the window.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        window_size: int = 30,
    ) -> None:
        if data.ndim != 2:
            raise ValueError(f"Expected data shape (T, F), got {data.shape}")

        if len(data) < window_size:
            raise ValueError(
                f"Data length {len(data)} is smaller than window_size={window_size}"
            )

        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64) if labels is not None else None
        self.window_size = window_size

        if self.labels is not None and len(self.labels) != len(self.data):
            raise ValueError(
                f"Labels length {len(self.labels)} does not match data length {len(self.data)}"
            )

    def __len__(self) -> int:
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.window_size

        window = self.data[start:end]
        target = self.data[end - 1]

        return torch.from_numpy(window), torch.from_numpy(target)

    def aligned_labels(self) -> Optional[np.ndarray]:
        """
        Labels aligned with model outputs.

        Since each prediction corresponds to the last point of a window,
        labels start from index window_size - 1.
        """
        if self.labels is None:
            return None

        return self.labels[self.window_size - 1 :]