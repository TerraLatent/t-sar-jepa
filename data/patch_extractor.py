"""Extract amplitude patches from Capella GEO products."""
import os
from pathlib import Path

import numpy as np
import rasterio


def load_geo_amplitude(filepath):
    """Load GEO product, return amplitude array and metadata dict.

    Args:
        filepath: Path to GeoTIFF file.

    Returns:
        Tuple of (amplitude_array, metadata_dict).
        metadata_dict contains 'crs', 'transform', 'bounds', 'width', 'height'.
    """
    with rasterio.open(filepath) as src:
        amplitude = src.read(1).astype(np.float32)
        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "width": src.width,
            "height": src.height,
        }
    return amplitude, metadata


def amplitude_to_db(amplitude, floor=1e-6):
    """Convert amplitude to log dB scale.

    Applies 20*log10(amplitude) and clips to [-40, 60] range.

    Args:
        amplitude: 2D numpy array of amplitude values.
        floor: Minimum amplitude value to avoid log(0).

    Returns:
        2D float32 array in dB scale, clipped to [-40, 60].
    """
    amp = np.clip(amplitude, floor, None)
    db = 20.0 * np.log10(amp)
    db = np.clip(db, -40.0, 60.0)
    return db.astype(np.float32)


def extract_patches(image, patch_size=224, stride=224, normalize=False):
    """Tile image into non-overlapping (or overlapping) patches.

    Args:
        image: 2D numpy array.
        patch_size: Size of square patches.
        stride: Step between patches.
        normalize: If True, scale each patch to [0, 1].

    Returns:
        List of 2D numpy arrays (patches).
    """
    h, w = image.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size].copy()
            if normalize:
                pmin = patch.min()
                pmax = patch.max()
                if pmax - pmin > 1e-8:
                    patch = (patch - pmin) / (pmax - pmin)
                else:
                    patch = np.zeros_like(patch)
                patch = patch.astype(np.float32)
            patches.append(patch)
    return patches


def process_geo_to_patches(filepath, aoi_name, timestamp, patch_size=224, stride=224):
    """Full pipeline: GEO -> amplitude dB -> normalized patches with metadata.

    Args:
        filepath: Path to GeoTIFF file.
        aoi_name: Name/ID of the AOI.
        timestamp: Timestamp string for this acquisition.
        patch_size: Patch dimension.
        stride: Step between patches.

    Returns:
        List of dicts with keys: patch, grid_x, grid_y, lon, lat, aoi_name, timestamp.
    """
    amplitude, metadata = load_geo_amplitude(filepath)
    db = amplitude_to_db(amplitude)
    patches = extract_patches(db, patch_size=patch_size, stride=stride, normalize=True)

    transform = metadata["transform"]
    h, w = amplitude.shape
    records = []
    idx = 0
    for gy, y in enumerate(range(0, h - patch_size + 1, stride)):
        for gx, x in enumerate(range(0, w - patch_size + 1, stride)):
            # Center pixel of the patch in geo coordinates
            cx = x + patch_size // 2
            cy = y + patch_size // 2
            lon, lat = transform * (cx, cy)

            records.append({
                "patch": patches[idx],
                "grid_x": gx,
                "grid_y": gy,
                "lon": lon,
                "lat": lat,
                "aoi_name": aoi_name,
                "timestamp": timestamp,
            })
            idx += 1

    return records


def save_patches(patch_records, output_dir):
    """Save patches as .npy files with naming convention.

    Naming: {aoi}_{gridx}_{gridy}_{timestamp}.npy

    Args:
        patch_records: List of dicts from process_geo_to_patches.
        output_dir: Directory to save patches.

    Returns:
        List of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for rec in patch_records:
        filename = f"{rec['aoi_name']}_{rec['grid_x']}_{rec['grid_y']}_{rec['timestamp']}.npy"
        filepath = output_dir / filename
        np.save(str(filepath), rec["patch"])
        saved.append(str(filepath))

    return saved
