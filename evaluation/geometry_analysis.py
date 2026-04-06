"""Satellite and acquisition geometry invariance analysis."""
import numpy as np
from scipy.stats import pearsonr, spearmanr

def satellite_correlation(anomaly_scores: np.ndarray, satellite_ids: list) -> dict:
    """Test correlation between anomaly scores and satellite transitions/IDs."""
    transitions = np.array([0 if i == 0 else int(satellite_ids[i] != satellite_ids[i-1]) for i in range(len(satellite_ids))])
    unique_sats = sorted(set(satellite_ids))
    sat_to_idx = {s: i for i, s in enumerate(unique_sats)}
    sat_indices = np.array([sat_to_idx[s] for s in satellite_ids])
    r_transition, p_transition = pearsonr(anomaly_scores, transitions)
    r_sat, p_sat = spearmanr(anomaly_scores, sat_indices)
    return {"transition_pearson_r": r_transition, "transition_p_value": p_transition, "satellite_spearman_r": r_sat, "satellite_p_value": p_sat, "unique_satellites": unique_sats, "n_transitions": int(transitions.sum())}
