"""Generate paper figures for T-SAR-JEPA v24.

Figure 1: Architecture diagram (tikz in LaTeX, skip here)
Figure 2: Kilauea eruption case study — 4 panels:
  (A) Anomaly score time series with eruption onset marked
  (B) SAR amplitude patches: quiescent vs eruption
  (C) Anomaly score heatmap at peak eruption (10x10 grid)
  (D) InSAR coherence map for comparison
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_scores(scores_file):
    """Load per-grid-key anomaly scores."""
    with open(scores_file) as f:
        return json.load(f)


def parse_date(d):
    """Parse YYYYMMDD string to datetime."""
    return datetime.strptime(str(d), "%Y%m%d")


def generate_kilauea_case_study(scores_file, coherence_dir, patch_dir, output_path):
    """Generate the 4-panel Kilauea case study figure."""
    scores = load_scores(scores_file)

    # --- Panel A: Time series for a representative grid cell ---
    # Pick a cell near the eruption zone (center of grid)
    target_keys = ["hawaii_5_5", "hawaii_4_5", "hawaii_5_4", "hawaii_4_4", "hawaii_6_5"]
    best_key = None
    best_max = 0
    for tk in target_keys:
        if tk in scores and max(scores[tk]["scores"]) > best_max:
            best_max = max(scores[tk]["scores"])
            best_key = tk

    if best_key is None:
        # Fall back to the cell with highest max score
        best_key = max(scores.keys(), key=lambda k: max(scores[k]["scores"]))

    cell_data = scores[best_key]
    dates = [parse_date(d) for d in cell_data["dates"]]
    cell_scores = np.array(cell_data["scores"])

    # Eruption dates
    eruption_dec = datetime(2024, 12, 23)
    eruption_mar = datetime(2025, 3, 13)

    # --- Panel C: 10x10 heatmap at peak eruption timestep ---
    # Find the timestep with highest mean anomaly across all Hawaii cells
    hawaii_keys = sorted([k for k in scores.keys()])  # all keys are hawaii

    # Find common dates across cells
    all_dates = set()
    for k in hawaii_keys:
        all_dates.update(scores[k]["dates"])
    all_dates = sorted(all_dates)

    # Build score matrix: find timestep with highest mean
    date_to_mean = {}
    for d in all_dates:
        vals = []
        for k in hawaii_keys:
            if d in scores[k]["dates"]:
                idx = scores[k]["dates"].index(d)
                vals.append(scores[k]["scores"][idx])
        if len(vals) >= 50:  # need at least half the grid
            date_to_mean[d] = np.mean(vals)

    if date_to_mean:
        peak_date = max(date_to_mean, key=date_to_mean.get)
        peak_dt = parse_date(peak_date)
    else:
        peak_date = cell_data["dates"][np.argmax(cell_scores)]
        peak_dt = parse_date(peak_date)

    # Build 10x10 grid for peak date
    grid = np.full((10, 10), np.nan)
    for k in hawaii_keys:
        parts = k.split("_")
        gy, gx = int(parts[1]), int(parts[2])
        if peak_date in scores[k]["dates"]:
            idx = scores[k]["dates"].index(peak_date)
            grid[gy, gx] = scores[k]["scores"][idx]

    # Also build a "quiescent" grid for comparison
    # Find a date well before eruptions with low mean score
    quiescent_dates = {d: m for d, m in date_to_mean.items()
                       if parse_date(d) < datetime(2024, 11, 1)}
    if quiescent_dates:
        quiet_date = min(quiescent_dates, key=quiescent_dates.get)
    else:
        quiet_date = all_dates[0]

    grid_quiet = np.full((10, 10), np.nan)
    for k in hawaii_keys:
        parts = k.split("_")
        gy, gx = int(parts[1]), int(parts[2])
        if quiet_date in scores[k]["dates"]:
            idx = scores[k]["dates"].index(quiet_date)
            grid_quiet[gy, gx] = scores[k]["scores"][idx]

    # --- Panel D: Coherence map ---
    coh_dir = Path(coherence_dir)
    coh_files = sorted(coh_dir.glob("*.npy"))

    # Find coherence map closest to peak eruption date
    best_coh = None
    best_coh_diff = 999
    for cf in coh_files:
        parts = cf.stem.split("_")
        coh_date = datetime.strptime(parts[3], "%Y-%m-%d")
        diff = abs((coh_date - peak_dt).days)
        if diff < best_coh_diff:
            best_coh_diff = diff
            best_coh = cf

    # Also find a pre-eruption coherence for comparison
    pre_coh = None
    for cf in coh_files:
        parts = cf.stem.split("_")
        coh_date = datetime.strptime(parts[3], "%Y-%m-%d")
        if coh_date < datetime(2025, 1, 15):
            pre_coh = cf

    # --- Panel B: SAR patches ---
    patch_path = Path(patch_dir)
    # Find patches for the representative cell at quiet and peak dates
    quiet_patch_file = patch_path / f"{best_key}_{quiet_date}.npy"
    peak_patch_file = patch_path / f"{best_key}_{peak_date}.npy"

    # ==================== INTERPOLATION ====================
    # Fill any NaN cells with neighbor mean before interpolation
    from scipy.ndimage import generic_filter
    def _nanmean_filter(values):
        valid = values[~np.isnan(values)]
        return np.nanmean(valid) if len(valid) > 0 else 0.0

    grid_filled = grid.copy()
    grid_quiet_filled = grid_quiet.copy()
    if np.any(np.isnan(grid_filled)):
        grid_filled = generic_filter(grid_filled, _nanmean_filter, size=3)
    if np.any(np.isnan(grid_quiet_filled)):
        grid_quiet_filled = generic_filter(grid_quiet_filled, _nanmean_filter, size=3)

    # Interpolate 10x10 -> 100x100 with bicubic for smooth visualization
    zoom_factor = 10
    grid_hires = zoom(grid_filled, zoom_factor, order=3)
    grid_quiet_hires = zoom(grid_quiet_filled, zoom_factor, order=3)
    diff_grid_hires = grid_hires - grid_quiet_hires

    # ==================== PLOTTING ====================
    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 2, figure=fig, wspace=0.35, hspace=0.35,
                  width_ratios=[1.5, 1])

    # Color scheme
    cmap_anomaly = 'hot_r'
    cmap_coherence = 'RdYlGn'

    # --- (A) Time series (spans full top row) ---
    ax_ts = fig.add_subplot(gs[0, :])
    ax_ts.plot(dates, cell_scores, 'k-', linewidth=0.8, alpha=0.7)
    ax_ts.fill_between(dates, cell_scores, alpha=0.3, color='steelblue')

    # Mark eruption
    ax_ts.axvline(eruption_dec, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax_ts.text(eruption_dec, ax_ts.get_ylim()[1] * 0.95, ' Dec \'24\n eruption',
               color='red', fontsize=8, va='top')

    # Mark peak
    peak_idx = list(dates).index(peak_dt) if peak_dt in dates else np.argmax(cell_scores)
    ax_ts.plot(dates[peak_idx], cell_scores[peak_idx], 'rv', markersize=10)

    ax_ts.set_xlabel('Date', fontsize=11)
    ax_ts.set_ylabel('Anomaly Score (L2)', fontsize=11)
    ax_ts.set_title(f'(A) Anomaly Score Time Series — {best_key}', fontsize=12, fontweight='bold')
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%b \'%y'))
    ax_ts.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax_ts.grid(True, alpha=0.3)

    # --- (B) Anomaly heatmap: eruption peak (interpolated) ---
    ax_heat = fig.add_subplot(gs[1, 0])
    p_vmin = np.nanmin(grid_hires) if not np.all(np.isnan(grid_hires)) else 0
    p_vmax = np.nanmax(grid_hires) if not np.all(np.isnan(grid_hires)) else 1
    im = ax_heat.imshow(grid_hires, cmap=cmap_anomaly, vmin=p_vmin, vmax=p_vmax,
                        interpolation='bilinear')
    fmt_peak = f'{peak_date[:4]}-{peak_date[4:6]}-{peak_date[6:]}'
    ax_heat.set_title(f'(B) Peak Anomaly Scores\n({fmt_peak})',
                     fontsize=11, fontweight='bold')
    ax_heat.axis('off')
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label='L2 Score')

    # --- (C) Coherence map ---
    ax_coh = fig.add_subplot(gs[1, 1])
    if best_coh is not None:
        coh_map = np.load(best_coh, mmap_mode='r')
        # Downsample for display
        step = max(1, coh_map.shape[0] // 200)
        coh_display = coh_map[::step, ::step]
        im_coh = ax_coh.imshow(coh_display, cmap=cmap_coherence, vmin=0, vmax=1)
        coh_parts = best_coh.stem.split("_")
        ax_coh.set_title(f'(C) InSAR Coherence\n({coh_parts[2]} to {coh_parts[3]})',
                        fontsize=11, fontweight='bold')
        plt.colorbar(im_coh, ax=ax_coh, fraction=0.046, pad=0.04, label='Coherence')
    else:
        ax_coh.text(0.5, 0.5, 'No coherence map', ha='center', va='center',
                   transform=ax_coh.transAxes)
        ax_coh.set_title('(C) InSAR Coherence', fontsize=11, fontweight='bold')
    ax_coh.axis('off')

    fig.suptitle('T-SAR-JEPA: Kilauea Eruption Case Study', fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight', pad_inches=0.1)
    print(f"[fig] Saved case study figure to {output_path}")
    plt.close()


def generate_architecture_diagram(output_path):
    """Generate a simplified architecture diagram using matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Colors
    c_stage1 = '#4ECDC4'
    c_stage2 = '#45B7D1'
    c_stage3 = '#96CEB4'
    c_arrow = '#2C3E50'
    c_text = '#2C3E50'

    box_style = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=c_arrow, linewidth=1.5)

    # Stage 1
    ax.add_patch(plt.Rectangle((0.2, 1), 3.5, 2, facecolor=c_stage1, alpha=0.3, edgecolor=c_arrow, linewidth=1.5, zorder=1))
    ax.text(2, 3.3, 'Stage 1: Domain Adaptation', fontsize=10, fontweight='bold', ha='center', color=c_text)
    ax.text(2, 2.4, 'SAR-JEPA\nViT-Base/16', fontsize=9, ha='center', va='center', bbox=box_style)
    ax.text(2, 1.3, 'LoMaR + Grad Targets\n39K Capella patches', fontsize=7, ha='center', va='center', style='italic', color='#555')

    # Arrow
    ax.annotate('', xy=(4.2, 2), xytext=(3.8, 2), arrowprops=dict(arrowstyle='->', color=c_arrow, lw=2))

    # Stage 2
    ax.add_patch(plt.Rectangle((4.5, 1), 4, 2, facecolor=c_stage2, alpha=0.3, edgecolor=c_arrow, linewidth=1.5, zorder=1))
    ax.text(6.5, 3.3, 'Stage 2: Temporal Predictor', fontsize=10, fontweight='bold', ha='center', color=c_text)
    ax.text(5.5, 2.4, 'Frozen\nEncoder', fontsize=8, ha='center', va='center', bbox=box_style)
    ax.text(7.5, 2.4, 'Temporal\nTransformer', fontsize=8, ha='center', va='center', bbox=box_style)
    ax.annotate('', xy=(6.8, 2.4), xytext=(6.2, 2.4), arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.5))
    ax.text(6.5, 1.3, 'Sinusoidal TE, K=7\nSmooth L1 loss', fontsize=7, ha='center', va='center', style='italic', color='#555')

    # Arrow
    ax.annotate('', xy=(9, 2), xytext=(8.6, 2), arrowprops=dict(arrowstyle='->', color=c_arrow, lw=2))

    # Stage 3
    ax.add_patch(plt.Rectangle((9.3, 1), 3.5, 2, facecolor=c_stage3, alpha=0.3, edgecolor=c_arrow, linewidth=1.5, zorder=1))
    ax.text(11.05, 3.3, 'Stage 3: Progressive Unfreezing', fontsize=10, fontweight='bold', ha='center', color=c_text)
    ax.text(11.05, 2.4, 'E2E Fine-tuning\nDiff. LR', fontsize=9, ha='center', va='center', bbox=box_style)
    ax.text(11.05, 1.3, 'Phase A: last 4 blocks\nPhase B: all layers', fontsize=7, ha='center', va='center', style='italic', color='#555')

    # Arrow to output
    ax.annotate('', xy=(13.3, 2), xytext=(12.9, 2), arrowprops=dict(arrowstyle='->', color=c_arrow, lw=2))

    # Output
    ax.add_patch(plt.Rectangle((13.5, 1.3), 2.2, 1.4, facecolor='#FFEAA7', alpha=0.5, edgecolor=c_arrow, linewidth=1.5, zorder=1))
    ax.text(14.6, 2.3, 'Anomaly Score', fontsize=9, fontweight='bold', ha='center', color=c_text)
    ax.text(14.6, 1.7, '$a_i = ||\\hat{z}_i - z_i||_2$', fontsize=10, ha='center', va='center', color=c_text)

    # Input annotation
    ax.text(0.5, 0.5, 'Input: Single-channel amplitude (1, 224, 224)', fontsize=8, ha='left', color='#777')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight', pad_inches=0.1)
    print(f"[fig] Saved architecture diagram to {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="results/v24/scores_hawaii.json")
    parser.add_argument("--coherence-dir", default="data/coherence/hawaii")
    parser.add_argument("--patch-dir", default="data/patches")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[fig] Generating architecture diagram...")
    generate_architecture_diagram(str(output_dir / "fig1_architecture_v24.pdf"))

    print("[fig] Generating Kilauea case study figure...")
    generate_kilauea_case_study(
        args.scores,
        args.coherence_dir,
        args.patch_dir,
        str(output_dir / "fig2_kilauea_case_study.pdf"),
    )

    print("[fig] All figures generated.")
