"""Fast patch extraction using rasterio windowed reads.
Only reads the center 2240x2240 pixels instead of full 30Kx30K image."""
import argparse
import sys
import os
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

AOIS = ["hawaii", "la", "pilbara"]
PATCH_SIZE = 224
GRID_SIZE = 10
GRID_PX = GRID_SIZE * PATCH_SIZE  # 2240


def amplitude_to_db(amp, floor=1e-6):
    a = np.clip(amp.astype(np.float32), floor, None)
    db = 20.0 * np.log10(a)
    return np.clip(db, -40.0, 60.0)


def main():
    parser = argparse.ArgumentParser(description="Extract 224x224 patches from GEO images")
    parser.add_argument("--geo-dir", default="data/geo",
                        help="Base directory containing per-AOI GEO files")
    parser.add_argument("--output-dir", default="data/patches",
                        help="Output directory for extracted patches")
    parser.add_argument("--clear-old", action="store_true",
                        help="Delete existing patches before extraction")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear old patches
    if args.clear_old:
        old = list(output_dir.glob("*.npy"))
        if old:
            print(f"Clearing {len(old)} old patches...")
            for f in old:
                f.unlink()

    total = 0
    skipped = 0
    zero_skipped = 0
    t0 = time.time()

    for aoi_name in AOIS:
        geo_dir = Path(args.geo_dir) / aoi_name
        geo_files = sorted(geo_dir.glob("*.tif"))
        print(f"\n=== {aoi_name}: {len(geo_files)} GEO files ===")

        for fi, geo_path in enumerate(geo_files):
            parts = geo_path.stem.split("_")
            date_str = parts[-2][:8]

            with rasterio.open(str(geo_path)) as src:
                h, w = src.height, src.width

                if h < GRID_PX or w < GRID_PX:
                    skipped += 1
                    continue

                # Windowed read of center crop only
                y0 = (h - GRID_PX) // 2
                x0 = (w - GRID_PX) // 2
                window = Window(x0, y0, GRID_PX, GRID_PX)
                crop = src.read(1, window=window).astype(np.float32)

            # Check if crop has data
            if crop.max() == 0:
                zero_skipped += 1
                continue

            db_crop = amplitude_to_db(crop)

            count = 0
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    py = gy * PATCH_SIZE
                    px = gx * PATCH_SIZE
                    patch = db_crop[py:py+PATCH_SIZE, px:px+PATCH_SIZE].copy()
                    pmin, pmax = patch.min(), patch.max()
                    if pmax - pmin > 1e-8:
                        patch = (patch - pmin) / (pmax - pmin)
                    else:
                        patch = np.zeros_like(patch)
                    np.save(str(output_dir / f"{aoi_name}_{gx}_{gy}_{date_str}.npy"), patch.astype(np.float32))
                    count += 1

            total += count
            if (fi + 1) % 10 == 0 or fi == len(geo_files) - 1:
                elapsed = time.time() - t0
                rate = (fi + 1) / elapsed
                print(f"  [{fi+1}/{len(geo_files)}] {date_str}: +{count} patches (total={total}, {rate:.1f} img/s, zero_skip={zero_skipped})")

    elapsed = time.time() - t0
    print(f"\n=== DONE ===")
    print(f"Total patches: {total}")
    print(f"Skipped (too small): {skipped}")
    print(f"Skipped (all zero): {zero_skipped}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
