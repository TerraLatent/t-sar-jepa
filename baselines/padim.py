"""PaDiM: Per-location Gaussian feature distribution modeling."""
import numpy as np
from scipy.spatial.distance import mahalanobis

def padim_fit(embeddings: np.ndarray) -> dict:
    """Fit Gaussian to embeddings (seq_len, embed_dim). Returns dict with mean, cov_inv."""
    mean = embeddings.mean(axis=0)
    cov = np.cov(embeddings.T) + np.eye(embeddings.shape[1]) * 1e-6
    return {"mean": mean, "cov_inv": np.linalg.inv(cov)}

def padim_score(embeddings: np.ndarray, params: dict) -> np.ndarray:
    """Score against fitted model. Returns (seq_len,) Mahalanobis distances."""
    return np.array([mahalanobis(embeddings[i], params["mean"], params["cov_inv"]) for i in range(len(embeddings))])
