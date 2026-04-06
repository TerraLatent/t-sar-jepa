"""Encode all SAR patches into latent embeddings using trained SAR-JEPA encoder.

Pre-encoding step: run every .npy patch through the frozen encoder to produce
768-dim feature vectors, then group them into temporal sequences by grid location.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.encoder import SARJEPAEncoder


def encode_all_patches(
    encoder: SARJEPAEncoder,
    patch_dir: str,
    output_dir: str,
    batch_size: int = 64,
    device: str = "cuda",
) -> Dict[str, str]:
    """Run all .npy patches through the frozen encoder and save 768-dim vectors.

    Args:
        encoder: Frozen SARJEPAEncoder instance.
        patch_dir: Directory containing .npy patch files.
        output_dir: Directory to save encoded .npy embedding files.
        batch_size: Number of patches to process at once.
        device: Torch device string.

    Returns:
        Dict mapping filename stems to output embedding file paths.
    """
    patch_dir = Path(patch_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patch_files = sorted(patch_dir.glob("*.npy"))
    if not patch_files:
        print(f"[encode] No .npy files found in {patch_dir}")
        return {}

    encoder = encoder.to(device)
    encoder.eval()

    embedding_map: Dict[str, str] = {}
    total = len(patch_files)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_files = patch_files[batch_start:batch_end]

        # Load and stack patches
        patches = []
        for f in batch_files:
            patch = np.load(f).astype(np.float32)
            # Ensure shape (1, H, W) - add channel dim if needed
            if patch.ndim == 2:
                patch = patch[np.newaxis, :, :]
            patches.append(patch)

        batch_tensor = torch.from_numpy(np.stack(patches, axis=0)).to(device)

        # Forward pass (no grad, frozen encoder)
        with torch.no_grad():
            embeddings = encoder(batch_tensor)  # (B, 768)

        embeddings_np = embeddings.cpu().numpy()

        # Save each embedding with the same filename as input
        for i, f in enumerate(batch_files):
            out_path = output_dir / f.name
            np.save(out_path, embeddings_np[i])
            embedding_map[f.stem] = str(out_path)

        if (batch_start // batch_size) % 10 == 0:
            print(f"[encode] {batch_end}/{total} patches encoded")

    print(f"[encode] Done. {total} patches encoded to {output_dir}")
    return embedding_map


def build_temporal_sequences(
    embeddings_dir: str,
    min_length: int = 10,
) -> List[dict]:
    """Group encoded vectors by grid location and build temporal sequences.

    Filename convention: {aoi}_{gridx}_{gridy}_{timestamp}.npy
    where timestamp is a date string like 20250115.

    Args:
        embeddings_dir: Directory containing .npy embedding files.
        min_length: Minimum number of observations to keep a sequence.

    Returns:
        List of dicts, each with:
            - embeddings: np.ndarray of shape (seq_len, 768)
            - days: np.ndarray of shape (seq_len,) with days since first acquisition
            - dates: list of date strings
            - grid_key: str like "{aoi}_{gridx}_{gridy}"
    """
    from datetime import datetime

    embeddings_dir = Path(embeddings_dir)
    files = sorted(embeddings_dir.glob("*.npy"))

    if not files:
        print(f"[sequences] No .npy files found in {embeddings_dir}")
        return []

    # Group by grid location
    groups: Dict[str, List[dict]] = {}
    for f in files:
        stem = f.stem
        parts = stem.rsplit("_", 3)
        if len(parts) < 4:
            print(f"[sequences] Skipping malformed filename: {f.name}")
            continue

        aoi, gridx, gridy, timestamp = parts[-4], parts[-3], parts[-2], parts[-1]
        grid_key = f"{aoi}_{gridx}_{gridy}"

        groups.setdefault(grid_key, []).append({
            "path": f,
            "timestamp": timestamp,
            "grid_key": grid_key,
        })

    # Build sequences
    sequences = []
    for grid_key, entries in groups.items():
        if len(entries) < min_length:
            continue

        # Sort by timestamp
        entries.sort(key=lambda e: e["timestamp"])

        # Parse timestamps into datetime objects
        dates = []
        for e in entries:
            ts = e["timestamp"]
            try:
                dt = datetime.strptime(ts, "%Y%m%d")
            except ValueError:
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%d")
                except ValueError:
                    print(f"[sequences] Cannot parse timestamp '{ts}' for {grid_key}, skipping entry")
                    continue
            dates.append((dt, e))

        if len(dates) < min_length:
            continue

        # Compute days since first acquisition
        first_date = dates[0][0]
        days = np.array([(d[0] - first_date).days for d in dates], dtype=np.float32)
        date_strings = [d[0].strftime("%Y%m%d") for d in dates]

        # Load embeddings
        emb_list = []
        for _, entry in dates:
            emb = np.load(entry["path"])
            emb_list.append(emb)

        embeddings = np.stack(emb_list, axis=0)  # (seq_len, 768)

        sequences.append({
            "embeddings": embeddings,
            "days": days,
            "dates": date_strings,
            "grid_key": grid_key,
        })

    print(f"[sequences] Built {len(sequences)} sequences (min_length={min_length}) "
          f"from {len(groups)} grid locations")
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Pre-encode SAR patches with frozen SAR-JEPA encoder")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SAR-JEPA encoder checkpoint")
    parser.add_argument("--patch-dir", type=str, required=True,
                        help="Directory containing .npy SAR patches")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save encoded embeddings")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Encoding batch size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Build encoder
    encoder = SARJEPAEncoder(
        pretrained=True,
        checkpoint_path=args.checkpoint,
        freeze=True,
    )

    # Encode patches
    encode_all_patches(
        encoder=encoder,
        patch_dir=args.patch_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Build temporal sequences
    sequences = build_temporal_sequences(
        embeddings_dir=args.output_dir,
        min_length=10,
    )

    # Save sequences
    seq_path = Path(args.output_dir) / "temporal_sequences.npy"
    np.save(seq_path, sequences, allow_pickle=True)
    print(f"[encode] Saved {len(sequences)} temporal sequences to {seq_path}")


if __name__ == "__main__":
    main()
