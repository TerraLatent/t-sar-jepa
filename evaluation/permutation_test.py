"""Permutation test for spatial coherence significance."""
import numpy as np

def compute_spatial_coherence(anomaly_flags: np.ndarray, grid_size: int = 10) -> float:
    """Fraction of flagged timesteps with at least one flagged 4-connected neighbor.
    anomaly_flags: (n_locations, n_timesteps) binary."""
    n_locs, n_times = anomaly_flags.shape
    assert n_locs == grid_size * grid_size
    neighbor_count = 0
    total_anomalies = 0
    for t in range(n_times):
        for loc in range(n_locs):
            if not anomaly_flags[loc, t]:
                continue
            total_anomalies += 1
            gy, gx = divmod(loc, grid_size)
            neighbors = []
            if gy > 0: neighbors.append((gy-1) * grid_size + gx)
            if gy < grid_size - 1: neighbors.append((gy+1) * grid_size + gx)
            if gx > 0: neighbors.append(gy * grid_size + gx - 1)
            if gx < grid_size - 1: neighbors.append(gy * grid_size + gx + 1)
            if any(anomaly_flags[n, t] for n in neighbors):
                neighbor_count += 1
    return neighbor_count / max(total_anomalies, 1)

def permutation_test(anomaly_scores: np.ndarray, threshold_percentile: float = 80, grid_size: int = 10, n_permutations: int = 1000, seed: int = 42) -> dict:
    """Shuffle time indices to test significance. Returns observed_coherence, null_mean, null_std, p_value, null_distribution."""
    rng = np.random.RandomState(seed)
    threshold = np.percentile(anomaly_scores, threshold_percentile)
    flags = (anomaly_scores >= threshold).astype(int)
    observed = compute_spatial_coherence(flags, grid_size)
    null_dist = []
    for _ in range(n_permutations):
        shuffled = flags.copy()
        for loc in range(shuffled.shape[0]):
            rng.shuffle(shuffled[loc])
        null_dist.append(compute_spatial_coherence(shuffled, grid_size))
    null_dist = np.array(null_dist)
    p_value = (null_dist >= observed).mean()
    return {"observed_coherence": observed, "null_mean": null_dist.mean(), "null_std": null_dist.std(), "p_value": p_value, "null_distribution": null_dist}
