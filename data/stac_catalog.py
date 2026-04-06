"""Query DFC 2026 STAC catalog and analyze AOI metadata."""
import pystac_client
from collections import defaultdict


def connect_catalog(stac_url: str):
    """Connect to STAC catalog."""
    return pystac_client.Client.open(stac_url)


def get_all_collections(client):
    """List all collections."""
    return list(client.get_collections())


def search_items(client, collection, bbox=None, datetime_range=None):
    """Search for items matching criteria.

    Args:
        client: pystac_client.Client instance.
        collection: Collection ID string.
        bbox: Optional bounding box [west, south, east, north].
        datetime_range: Optional datetime string (e.g. "2024-01-01/2025-01-01").

    Returns:
        List of pystac Items.
    """
    search_kwargs = {"collections": [collection]}
    if bbox is not None:
        search_kwargs["bbox"] = bbox
    if datetime_range is not None:
        search_kwargs["datetime"] = datetime_range

    search = client.search(**search_kwargs)
    return list(search.items())


def _round_bbox(bbox, precision=1):
    """Round bbox coordinates to group nearby geometries.

    Args:
        bbox: Tuple/list of (west, south, east, north).
        precision: Number of decimal places.

    Returns:
        Tuple of rounded coordinates.
    """
    return tuple(round(c, precision) for c in bbox)


def analyze_temporal_density(items):
    """Group items by AOI/geometry, count temporal observations.

    Groups items by their bounding box rounded to 0.1 degree,
    then counts how many temporal observations exist per AOI.

    Args:
        items: List of pystac Items.

    Returns:
        Dict mapping rounded-bbox tuple to list of items in that AOI.
    """
    aoi_groups = defaultdict(list)
    for item in items:
        bbox = item.bbox
        if bbox is None:
            continue
        key = _round_bbox(bbox, precision=1)
        aoi_groups[key].append(item)
    return dict(aoi_groups)


def rank_aois(aoi_groups, min_observations=30):
    """Rank AOIs by temporal density (number of observations).

    Args:
        aoi_groups: Dict from analyze_temporal_density.
        min_observations: Minimum number of observations to include.

    Returns:
        List of (bbox_key, item_count, items) tuples sorted descending by count.
    """
    ranked = []
    for bbox_key, items in aoi_groups.items():
        count = len(items)
        if count >= min_observations:
            ranked.append((bbox_key, count, items))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
