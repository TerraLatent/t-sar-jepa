"""Download Capella GEO products from AWS S3."""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import boto3
from botocore import UNSIGNED
from botocore.config import Config


def _parse_s3_path(href: str):
    """Parse S3 bucket and key from an asset href.

    Supports both s3:// and https://*.s3.amazonaws.com/ formats.

    Returns:
        Tuple of (bucket, key).
    """
    if href.startswith("s3://"):
        parsed = urlparse(href)
        return parsed.netloc, parsed.path.lstrip("/")
    elif "s3.amazonaws.com" in href:
        parsed = urlparse(href)
        # https://bucket.s3.amazonaws.com/key or
        # https://s3.amazonaws.com/bucket/key
        host = parsed.hostname
        if host.endswith(".s3.amazonaws.com"):
            bucket = host.replace(".s3.amazonaws.com", "")
            key = parsed.path.lstrip("/")
        else:
            # s3.amazonaws.com/bucket/key
            parts = parsed.path.lstrip("/").split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        return bucket, key
    elif "s3." in href and ".amazonaws.com" in href:
        # https://bucket.s3.region.amazonaws.com/key
        parsed = urlparse(href)
        host = parsed.hostname
        bucket = host.split(".s3.")[0]
        key = parsed.path.lstrip("/")
        return bucket, key
    else:
        raise ValueError(f"Cannot parse S3 path from href: {href}")


def _download_single(s3_client, bucket, key, local_path):
    """Download a single file from S3 if it doesn't already exist."""
    if os.path.exists(local_path):
        return local_path, True  # skipped

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, key, local_path)
    return local_path, False  # downloaded


def download_geo_products(items, output_dir, max_workers=4):
    """Download GEO assets from STAC items to local directory.

    S3 is public (unsigned access). Parses S3 path from asset href
    (s3:// or https:// format). Skips files that already exist locally.

    Args:
        items: List of pystac Items with GEO assets.
        output_dir: Local directory to save downloaded files.
        max_workers: Number of parallel download threads.

    Returns:
        List of dicts with 'item_id', 'local_path', 'skipped' keys.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    download_tasks = []
    for item in items:
        # Look for GEO asset (try common keys)
        asset = None
        for key in ["GEO", "geo", "data", "image", "visual"]:
            if key in item.assets:
                asset = item.assets[key]
                break
        if asset is None:
            # Fallback: use first asset with a tif href
            for a in item.assets.values():
                if a.href and a.href.endswith(".tif"):
                    asset = a
                    break
        if asset is None:
            continue

        try:
            bucket, s3_key = _parse_s3_path(asset.href)
        except ValueError:
            continue

        filename = os.path.basename(s3_key)
        local_path = str(output_dir / filename)
        download_tasks.append((item.id, bucket, s3_key, local_path))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for item_id, bucket, key, local_path in download_tasks:
            future = executor.submit(_download_single, s3_client, bucket, key, local_path)
            futures[future] = item_id

        for future in as_completed(futures):
            item_id = futures[future]
            try:
                local_path, skipped = future.result()
                results.append({
                    "item_id": item_id,
                    "local_path": local_path,
                    "skipped": skipped,
                })
            except Exception as e:
                results.append({
                    "item_id": item_id,
                    "local_path": None,
                    "skipped": False,
                    "error": str(e),
                })

    return results
