"""Download SLC products per AOI, compute coherence, save maps, delete raw SLC."""
import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import boto3
from botocore import UNSIGNED
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.coherence import load_slc_complex, compute_coherence

AOIS = {
    "hawaii": {"bbox": [-155.4, 19.3, -155.1, 19.5]},
    "la": {"bbox": [-118.2, 34.7, -117.9, 34.9]},
    "pilbara": {"bbox": [118.6, -23.3, 118.9, -23.1]},
}


def get_sat_id(item_id):
    return item_id.split("_")[1]


def main():
    parser = argparse.ArgumentParser(description="Download SLC products and compute InSAR coherence")
    parser.add_argument("--stac-cache", default="data/stac_items_cache.json",
                        help="Path to STAC items cache JSON")
    parser.add_argument("--output-dir", default="data",
                        help="Base data directory (SLC and coherence subdirs created here)")
    parser.add_argument("--max-gap-days", type=int, default=12,
                        help="Maximum temporal gap between SLC pairs (days)")
    args = parser.parse_args()

    with open(args.stac_cache) as f:
        all_items = json.load(f)

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    for aoi_name, aoi_cfg in AOIS.items():
        bbox = aoi_cfg["bbox"]
        slc_items = sorted([
            i for i in all_items
            if i["product_type"] == "SLC"
            and bbox[0] <= (i["bbox"][0] + i["bbox"][2]) / 2 <= bbox[2]
            and bbox[1] <= (i["bbox"][1] + i["bbox"][3]) / 2 <= bbox[3]
        ], key=lambda x: x["datetime"])

        print(f"\n=== {aoi_name}: {len(slc_items)} SLC items ===")

        base_dir = Path(args.output_dir)
        slc_dir = base_dir / "slc" / aoi_name
        coh_dir = base_dir / "coherence" / aoi_name
        slc_dir.mkdir(parents=True, exist_ok=True)
        coh_dir.mkdir(parents=True, exist_ok=True)

        # Group by satellite for valid InSAR pairs
        by_sat = defaultdict(list)
        for item in slc_items:
            by_sat[get_sat_id(item["id"])].append(item)

        for sat, items in sorted(by_sat.items()):
            if len(items) < 2:
                continue
            print(f"  Satellite {sat}: {len(items)} SLCs")

            for i in range(len(items) - 1):
                ref_item = items[i]
                sec_item = items[i + 1]

                dt1 = datetime.fromisoformat(ref_item["datetime"].replace("Z", "+00:00"))
                dt2 = datetime.fromisoformat(sec_item["datetime"].replace("Z", "+00:00"))
                gap_days = abs((dt2 - dt1).days)
                if gap_days > args.max_gap_days:
                    continue

                date1 = ref_item["datetime"][:10]
                date2 = sec_item["datetime"][:10]
                pair_id = f"{aoi_name}_{sat}_{date1}_{date2}"
                coh_path = coh_dir / f"{pair_id}_coherence.npy"
                if coh_path.exists():
                    continue

                # Download both SLCs
                ref_local = slc_dir / f"{ref_item['id']}.tif"
                sec_local = slc_dir / f"{sec_item['id']}.tif"

                for item, local in [(ref_item, ref_local), (sec_item, sec_local)]:
                    if not local.exists():
                        try:
                            parsed = urlparse(item["hh_href"])
                            bucket = parsed.hostname.split(".s3")[0]
                            key = parsed.path.lstrip("/")
                            print(f"    Downloading {item['id'][:50]}...")
                            s3.download_file(bucket, key, str(local))
                        except Exception as e:
                            print(f"    ERROR downloading {item['id'][:40]}: {e}")
                            continue

                # Compute coherence
                try:
                    ref_slc = load_slc_complex(str(ref_local))
                    sec_slc = load_slc_complex(str(sec_local))
                    min_h = min(ref_slc.shape[0], sec_slc.shape[0])
                    min_w = min(ref_slc.shape[1], sec_slc.shape[1])
                    coh = compute_coherence(ref_slc[:min_h, :min_w], sec_slc[:min_h, :min_w])
                    np.save(str(coh_path), coh)
                    print(f"    Coherence {pair_id}: shape={coh.shape}, mean={coh.mean():.3f}")
                except Exception as e:
                    print(f"    ERROR coherence {pair_id}: {e}")

                # Delete downloaded SLCs to save space
                for f in [ref_local, sec_local]:
                    if f.exists():
                        f.unlink()

        print(f"  {aoi_name}: {len(list(coh_dir.glob('*.npy')))} coherence maps saved")


if __name__ == "__main__":
    main()
