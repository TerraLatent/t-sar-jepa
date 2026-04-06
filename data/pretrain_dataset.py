"""Dataset for Stage 1: domain adaptation (shuffled patches)."""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainPatchDataset(Dataset):
    """Loads individual .npy SAR patches from a directory for spatial JEPA pretraining."""

    def __init__(self, patch_dir: str, transform: Optional[Callable] = None):
        self.patch_dir = Path(patch_dir)
        self.transform = transform
        self.files = sorted(self.patch_dir.glob("*.npy"))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        patch = np.load(self.files[idx])  # (H, W) float32
        tensor = torch.from_numpy(patch).unsqueeze(0)  # (1, H, W)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor
