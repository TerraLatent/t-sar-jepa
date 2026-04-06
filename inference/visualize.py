"""Publication-quality visualization functions for T-SAR-JEPA results.

All figures use clean, minimal matplotlib styling suitable for paper
submission. Saved at 300 DPI with tight bounding boxes.
"""

from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _clean_axes(ax, hide_top=True, hide_right=True):
    """Remove top/right spines for a cleaner look."""
    if hide_top:
        ax.spines["top"].set_visible(False)
    if hide_right:
        ax.spines["right"].set_visible(False)


def _save_or_show(fig, save_path=None):
    """Save figure at 300 DPI or show it."""
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_temporal_anomaly_curve(
    scores: np.ndarray,
    dates: Optional[List] = None,
    title: str = "Temporal Anomaly Scores",
    anomaly_indices: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Line plot of anomaly scores over time (Figure 3).

    Args:
        scores: 1D array of anomaly scores.
        dates: Optional list of datetime objects for x-axis.
        title: Figure title.
        anomaly_indices: Indices of detected anomalies (red dots).
        save_path: Path to save figure. None to display.
        figsize: Figure dimensions.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = dates if dates is not None else np.arange(len(scores))
    ax.plot(x, scores, color="#2077B4", linewidth=1.2, zorder=2)
    ax.fill_between(x, scores, alpha=0.15, color="#2077B4", zorder=1)

    if anomaly_indices is not None and len(anomaly_indices) > 0:
        ax_vals = [x[i] for i in anomaly_indices if i < len(x)]
        score_vals = [scores[i] for i in anomaly_indices if i < len(scores)]
        ax.scatter(ax_vals, score_vals, color="#D62728", s=40, zorder=3, label="Anomaly")
        ax.legend(frameon=False)

    if dates is not None:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.set_ylabel("Anomaly Score")
    ax.set_title(title)
    _clean_axes(ax)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_spatial_anomaly_heatmap(
    scores_grid: np.ndarray,
    amplitude_image: Optional[np.ndarray] = None,
    title: str = "Spatial Anomaly Heatmap",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Heatmap of anomaly scores on a spatial grid (Figure 4).

    Args:
        scores_grid: 2D array of anomaly scores (H, W).
        amplitude_image: Optional SAR amplitude image as underlay.
        title: Figure title.
        save_path: Path to save figure. None to display.
        figsize: Figure dimensions.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if amplitude_image is not None:
        ax.imshow(amplitude_image, cmap="gray", aspect="auto")
        im = ax.imshow(scores_grid, cmap="hot", alpha=0.6, aspect="auto")
    else:
        im = ax.imshow(scores_grid, cmap="hot", aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Anomaly Score")

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    dates: Optional[List] = None,
    title: str = "Attention Weights",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Heatmap of averaged attention weights across heads.

    Args:
        attention_weights: Array of shape (num_heads, seq_len, seq_len)
            or (seq_len, seq_len) if already averaged.
        dates: Optional date labels for axes.
        title: Figure title.
        save_path: Path to save figure. None to display.
        figsize: Figure dimensions.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if attention_weights.ndim == 3:
        attn = attention_weights.mean(axis=0)
    else:
        attn = attention_weights

    im = ax.imshow(attn, cmap="Blues", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight")

    if dates is not None:
        labels = [d.strftime("%Y-%m") if hasattr(d, "strftime") else str(d) for d in dates]
        step = max(1, len(labels) // 10)
        ax.set_xticks(range(0, len(labels), step))
        ax.set_xticklabels(labels[::step], rotation=45, ha="right")
        ax.set_yticks(range(0, len(labels), step))
        ax.set_yticklabels(labels[::step])

    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title(title)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_before_after(
    before_image: np.ndarray,
    after_image: np.ndarray,
    anomaly_overlay: Optional[np.ndarray] = None,
    title: str = "Before / After Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Side-by-side before/after with optional anomaly overlay (Figure 5).

    Args:
        before_image: 2D SAR amplitude image (before event).
        after_image: 2D SAR amplitude image (after event).
        anomaly_overlay: Optional 2D anomaly mask/scores for overlay.
        title: Figure title.
        save_path: Path to save figure. None to display.
        figsize: Figure dimensions.

    Returns:
        The matplotlib Figure.
    """
    n_panels = 3 if anomaly_overlay is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    axes[0].imshow(before_image, cmap="gray")
    axes[0].set_title("Before")
    axes[0].axis("off")

    axes[1].imshow(after_image, cmap="gray")
    axes[1].set_title("After")
    axes[1].axis("off")

    if anomaly_overlay is not None:
        axes[2].imshow(after_image, cmap="gray")
        im = axes[2].imshow(anomaly_overlay, cmap="hot", alpha=0.6)
        axes[2].set_title("Detected Changes")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Anomaly Score")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_latent_tsne(
    embeddings: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    title: str = "Latent Space (t-SNE)",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """t-SNE visualization of 768-dim embeddings colored by timestep.

    Args:
        embeddings: Array of shape (N, 768).
        timestamps: Optional 1D array for color mapping (temporal progression).
        title: Figure title.
        save_path: Path to save figure. None to display.
        figsize: Figure dimensions.

    Returns:
        The matplotlib Figure.
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)

    if timestamps is not None:
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=timestamps, cmap="viridis", s=30, alpha=0.8, edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Timestep")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.8, edgecolors="none", color="#2077B4")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    _clean_axes(ax)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig


def plot_score_distribution(
    scores: np.ndarray,
    title: str = "Anomaly Score Distribution",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 4),
) -> plt.Figure:
    """Histogram of anomaly scores with percentile markers.

    Args:
        scores: 1D array of anomaly scores.
        title: Figure title.
        save_path: Path to save figure. None to display.
        figsize: Figure dimensions.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(scores, bins=50, color="#2077B4", alpha=0.7, edgecolor="white", linewidth=0.5)

    percentiles = [50, 90, 95, 99]
    colors = ["#2CA02C", "#FF7F0E", "#D62728", "#9467BD"]
    for pct, color in zip(percentiles, colors):
        val = np.percentile(scores, pct)
        ax.axvline(val, color=color, linestyle="--", linewidth=1.2, label=f"P{pct} = {val:.3f}")

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    _clean_axes(ax)
    fig.tight_layout()
    _save_or_show(fig, save_path)
    return fig
