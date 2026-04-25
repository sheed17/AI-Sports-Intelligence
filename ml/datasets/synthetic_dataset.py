"""PyTorch Dataset that loads pre-windowed synthetic (or pre-extracted) data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticWindowDataset(Dataset):
    """Loads (windows.npy, labels.npy) from a directory.

    Works for both synthetic data and any pre-windowed real data.
    """

    def __init__(self, data_dir: str | Path, augment: bool = False):
        data_dir = Path(data_dir)
        self.windows = np.load(str(data_dir / "windows.npy")).astype(np.float32)
        self.labels = np.load(str(data_dir / "labels.npy")).astype(np.int64)
        self.augment = augment
        assert len(self.windows) == len(self.labels)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.windows[idx])
        if self.augment:
            x = x + torch.randn_like(x) * 0.01
        return x, torch.tensor(int(self.labels[idx]), dtype=torch.long)
