"""Dataset for Stage 2: temporal predictor training."""

from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalPatchDataset(Dataset):
    """Sliding-window dataset over pre-encoded SAR embedding sequences."""

    def __init__(self, sequences: List[Dict], window_size: int = 16):
        self.window_size = window_size
        self.windows: list[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for seq in sequences:
            emb = seq["embeddings"]   # (seq_len, 768)
            days = seq["days"]        # (seq_len,)
            n = len(emb)
            if n <= window_size:
                continue
            for i in range(n - window_size):
                context = emb[i : i + window_size]        # (window_size, 768)
                target = emb[i + window_size]              # (768,)
                window_days = days[i : i + window_size]    # (window_size,)
                self.windows.append((context, target, window_days))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, target, days = self.windows[idx]
        context = torch.from_numpy(context)
        target = torch.from_numpy(target)

        # Normalize days to [0, 1] within the window
        d = days.astype(np.float32)
        d_min, d_max = d[0], d[-1]
        if d_max > d_min:
            time_enc = torch.from_numpy((d - d_min) / (d_max - d_min))
        else:
            time_enc = torch.zeros(len(d))

        return context, target, time_enc
