"""RX anomaly detector on embedding features."""
import numpy as np
from scipy.spatial.distance import mahalanobis

def rx_anomaly_scores(embeddings: np.ndarray) -> np.ndarray:
    """Compute RX scores for temporal sequence. embeddings: (seq_len, embed_dim). Returns (seq_len,)."""
    mean = embeddings.mean(axis=0)
    cov = np.cov(embeddings.T)
    cov += np.eye(cov.shape[0]) * 1e-6
    cov_inv = np.linalg.inv(cov)
    return np.array([mahalanobis(embeddings[i], mean, cov_inv) for i in range(len(embeddings))])
