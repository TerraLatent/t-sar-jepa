"""Download GEO products for all 3 AOIs from Capella S3."""
import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import boto3
from botocore import UNSIGNED
from botocore.config import Config

AOIS = {
    "hawaii": {"bbox": [-155.4, 19.3, -155.1, 19.5]},
    "la": {"bbox": [-118.2, 34.7, -117.9, 34.9]},
    "pilbara": {"bbox": [118.6, -23.3, 118.9, -23.1]},
}


def main():
    parser = argparse.ArgumentParser(description="Download GEO products from Capella S3")
    parser.add_argument("--stac-cache", default="data/stac_items_cache.json",
                        help="Path to STAC items cache JSON")
    parser.add_argument("--output-dir", default="data/geo",
                        help="Base output directory for downloaded GEO files")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download threads")
    args = parser.parse_args()

    with open(args.stac_cache) as f:
        all_items = json.load(f)
    print(f"Loaded {len(all_items)} items from cache")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    for aoi_name, aoi_cfg in AOIS.items():
        bbox = aoi_cfg["bbox"]
        geo_items = sorted([
            i for i in all_items
            if i["product_type"] == "GEO"
            and bbox[0] <= (i["bbox"][0] + i["bbox"][2]) / 2 <= bbox[2]
            and bbox[1] <= (i["bbox"][1] + i["bbox"][3]) / 2 <= bbox[3]
        ], key=lambda x: x["datetime"])

        output_dir = Path(args.output_dir) / aoi_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {aoi_name}: {len(geo_items)} GEO items ===")

        def download_one(item):
            if item.get("hh_href") is None:
                return "skip", True
            href = item["hh_href"]
            filename = href.split("/")[-1]
            local_path = output_dir / filename
            if local_path.exists():
                return str(local_path), True
            try:
                parsed = urlparse(href)
                bucket = parsed.hostname.split(".s3")[0]
                key = parsed.path.lstrip("/")
                s3.download_file(bucket, key, str(local_path))
                return str(local_path), False
            except Exception as e:
                print(f"  ERROR downloading {filename}: {e}")
                return str(local_path), True

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(download_one, geo_items))

        downloaded = sum(1 for _, skipped in results if not skipped)
        skipped = sum(1 for _, s in results if s)
        print(f"  Downloaded: {downloaded}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
