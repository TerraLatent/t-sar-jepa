"""Load raw patches and metadata for a specific AOI for inference."""

import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_aoi_sequence(
    aoi_name: str,
    patches_dir: str = "data/patches",
) -> Tuple[np.ndarray, np.ndarray, List[datetime]]:
    """Load raw patches and metadata for a specific AOI.

    Finds all .npy patch files matching the AOI name, groups them by grid
    location, picks the group with the most temporal observations, and
    returns the sorted sequence.

    Filename convention: {aoi}_{gridx}_{gridy}_{timestamp}.npy

    Args:
        aoi_name: Name of the AOI to load (e.g. "dubai").
        patches_dir: Directory containing .npy patch files.

    Returns:
        Tuple of:
            patches: np.ndarray of shape (seq_len, 1, 224, 224)
            days: np.ndarray of shape (seq_len,) with days since first acquisition
            dates: List of datetime objects for each timestep
    """
    patches_dir = Path(patches_dir)
    pattern = f"{aoi_name}_*.npy"
    files = sorted(patches_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No .npy files matching '{pattern}' found in {patches_dir}"
        )

    # Group by grid location (aoi_gridx_gridy)
    groups = {}
    for f in files:
        stem = f.stem
        parts = stem.rsplit("_", 3)
        if len(parts) < 4:
            continue

        aoi, gridx, gridy, timestamp = parts[-4], parts[-3], parts[-2], parts[-1]
        grid_key = f"{aoi}_{gridx}_{gridy}"
        groups.setdefault(grid_key, []).append({
            "path": f,
            "timestamp": timestamp,
        })

    if not groups:
        raise ValueError(f"No valid grid groups found for AOI '{aoi_name}'")

    # Pick the group with the most temporal observations
    best_key = max(groups, key=lambda k: len(groups[k]))
    entries = groups[best_key]
    print(f"[data_loader] Selected grid '{best_key}' with {len(entries)} observations")

    # Sort by timestamp
    entries.sort(key=lambda e: e["timestamp"])

    # Parse timestamps and load patches
    patch_list = []
    dates = []
    for entry in entries:
        ts = entry["timestamp"]
        try:
            dt = datetime.strptime(ts, "%Y%m%d")
        except ValueError:
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d")
            except ValueError:
                print(f"[data_loader] Skipping unparseable timestamp: {ts}")
                continue

        patch = np.load(entry["path"]).astype(np.float32)
        # Ensure shape (1, 224, 224)
        if patch.ndim == 2:
            patch = patch[np.newaxis, :, :]

        patch_list.append(patch)
        dates.append(dt)

    if not patch_list:
        raise ValueError(f"No valid patches loaded for AOI '{aoi_name}'")

    # Stack patches
    patches = np.stack(patch_list, axis=0)  # (seq_len, 1, 224, 224)

    # Compute days since first acquisition
    first_date = dates[0]
    days = np.array([(d - first_date).days for d in dates], dtype=np.float32)

    print(f"[data_loader] Loaded {len(dates)} patches, spanning {days[-1]:.0f} days")

    return patches, days, dates
