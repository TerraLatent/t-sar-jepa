"""Compute InSAR coherence from SLC pairs.

InSAR coherence measures the similarity of the complex signal between two
SAR acquisitions. High coherence = stable surface, low coherence = change.

Uses a sliding window estimator:
    gamma = |sum(s1 * conj(s2))| / sqrt(sum(|s1|^2) * sum(|s2|^2))
"""
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter


def load_slc_complex(filepath: str) -> np.ndarray:
    """Load SLC product as complex array.

    Capella SLC GeoTIFFs store complex data as two-band (real, imaginary)
    or as a single complex64 band.

    Returns:
        Complex64 2D array.
    """
    with rasterio.open(filepath) as src:
        if src.count == 2:
            real = src.read(1).astype(np.float32)
            imag = src.read(2).astype(np.float32)
            return real + 1j * imag
        else:
            data = src.read(1)
            if np.issubdtype(data.dtype, np.complexfloating):
                return data.astype(np.complex64)
            # Fallback: treat as amplitude only
            return data.astype(np.float32) + 0j


def compute_coherence(
    slc1: np.ndarray,
    slc2: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Compute interferometric coherence between two SLC images.

    Args:
        slc1: Complex64 array (reference).
        slc2: Complex64 array (secondary). Must be same shape.
        window_size: Averaging window size.

    Returns:
        Float32 coherence array in [0, 1].
    """
    assert slc1.shape == slc2.shape, f"Shape mismatch: {slc1.shape} vs {slc2.shape}"

    cross = slc1 * np.conj(slc2)
    power1 = np.abs(slc1) ** 2
    power2 = np.abs(slc2) ** 2

    cross_avg = uniform_filter(cross.real, window_size) + 1j * uniform_filter(cross.imag, window_size)
    power1_avg = uniform_filter(power1, window_size)
    power2_avg = uniform_filter(power2, window_size)

    denom = np.sqrt(power1_avg * power2_avg)
    denom = np.maximum(denom, 1e-10)
    coherence = np.abs(cross_avg) / denom

    return np.clip(coherence, 0.0, 1.0).astype(np.float32)
