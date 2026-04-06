"""T-SAR-JEPA — Complete evaluation with baselines ROC/PR, geometry, cross-AOI.

Fixes from v1:
- Saves per-grid baseline scores (not just aggregates)
- Computes ROC/PR for ALL methods (not just T-SAR-JEPA)
- Adds geometry invariance using satellite IDs from STAC cache
- Adds cross-AOI generalization summary
- Outputs everything needed for paper tables
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.encoder import SARJEPAEncoder
from models.temporal_predictor import TemporalPredictor
from training.finetune_e2e import (
    RawPatchTemporalDataset, build_raw_sequences, _load_patch
)
from baselines.rx_detector import rx_anomaly_scores
from baselines.padim import padim_fit, padim_score
from baselines.linear_ar import train_linear_ar
from baselines.lstm_temporal import train_lstm
from evaluation.roc_pr import coherence_to_labels, compute_roc_pr
from evaluation.permutation_test import permutation_test
from evaluation.geometry_analysis import satellite_correlation


def load_model(config, checkpoint_path, device):
    model_cfg = config["model"]
    encoder = SARJEPAEncoder(
        pretrained=False, embed_dim=768, freeze=False,
        in_chans=model_cfg.get("in_chans", 1)
    ).to(device)
    predictor = TemporalPredictor(
        embed_dim=768, num_layers=model_cfg.get("num_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        ffn_dim=model_cfg.get("ffn_dim", 2048),
        dropout=model_cfg.get("dropout", 0.1),
        time_encoding_type=model_cfg.get("time_encoding_type", "sinusoidal"),
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    predictor.load_state_dict(ckpt["predictor_state_dict"])
    encoder.eval()
    predictor.eval()
    print(f"[eval] Loaded: epoch={ckpt['epoch']}, phase={ckpt['phase']}, val_loss={ckpt['val_loss']:.6f}")
    return encoder, predictor


def compute_all_scores(encoder, predictor, sequences, window_size, device, batch_size=64):
    """Compute T-SAR-JEPA anomaly scores AND save embeddings for baselines."""
    model_results = {}
    encoder.eval()
    predictor.eval()

    with torch.no_grad():
        for si, seq in enumerate(sequences):
            grid_key = seq["grid_key"]
            aoi = grid_key.rsplit("_", 2)[0]
            patch_paths = seq["paths"]
            day_offsets = seq["days"]

            dates = [Path(p).stem.rsplit("_", 1)[-1] for p in patch_paths]

            if len(patch_paths) < window_size + 1:
                continue

            if si % 50 == 0:
                print(f"  [{si+1}/{len(sequences)}] {grid_key}...")

            # Encode all patches
            patches = []
            for p in patch_paths:
                patch = _load_patch(Path(p))
                patches.append(torch.from_numpy(patch).float())
            patches_tensor = torch.stack(patches).to(device)

            embeddings = []
            for i in range(0, len(patches_tensor), batch_size):
                batch = patches_tensor[i:i+batch_size]
                emb = encoder(batch)
                embeddings.append(emb.cpu())
            embeddings = torch.cat(embeddings, dim=0)  # (T, 768)

            # Normalize days
            day_arr = np.array(day_offsets, dtype=np.float32)
            if day_arr.max() > day_arr.min():
                day_norm = (day_arr - day_arr.min()) / (day_arr.max() - day_arr.min())
            else:
                day_norm = np.zeros_like(day_arr)

            # T-SAR-JEPA sliding window scores
            scores = []
            score_dates = []
            for i in range(len(embeddings) - window_size):
                context = embeddings[i:i+window_size].unsqueeze(0).to(device)
                target = embeddings[i+window_size]
                time_enc = torch.tensor(day_norm[i:i+window_size]).unsqueeze(0).to(device)
                pred = predictor(context, time_enc)
                score = torch.norm(pred.cpu() - target, dim=-1).item()
                scores.append(score)
                score_dates.append(dates[i + window_size])

            model_results[grid_key] = {
                "scores": np.array(scores),
                "dates": score_dates,
                "aoi": aoi,
                "embeddings": embeddings.numpy(),  # KEEP for baselines
                "day_offsets": day_offsets,
                "all_dates": dates,
            }

    return model_results


def compute_baseline_scores_per_grid(model_results, window_size=7):
    """Run baselines and save PER-GRID scores (not just aggregates)."""
    baseline_scores = {"rx": {}, "padim": {}, "linear_ar": {}, "lstm": {}}

    # RX and PaDiM: per-sequence
    print("[eval] Running RX + PaDiM per grid...")
    for gk, data in model_results.items():
        emb = data["embeddings"]
        if len(emb) <= window_size:
            continue
        # RX: on post-window embeddings
        rx_s = rx_anomaly_scores(emb[window_size:])
        baseline_scores["rx"][gk] = rx_s.tolist()
        # PaDiM: fit on first K, score rest
        params = padim_fit(emb[:window_size])
        padim_s = padim_score(emb[window_size:], params)
        baseline_scores["padim"][gk] = padim_s.tolist()

    # Linear AR + LSTM: trained globally then scored per-grid
    seq_list = []
    seq_keys = []
    for gk, data in model_results.items():
        if len(data["embeddings"]) > window_size:
            seq_list.append({"embeddings": data["embeddings"], "grid_key": gk})
            seq_keys.append(gk)

    print("[eval] Training Linear AR...")
    try:
        _, linear_all = train_linear_ar(seq_list, context_k=window_size, epochs=50)
        idx = 0
        for seq in seq_list:
            n = len(seq["embeddings"]) - window_size
            if n > 0:
                baseline_scores["linear_ar"][seq["grid_key"]] = linear_all[idx:idx+n].tolist()
                idx += n
    except Exception as e:
        print(f"  Linear AR failed: {e}")

    print("[eval] Training LSTM...")
    try:
        _, lstm_all = train_lstm(seq_list, context_k=window_size, epochs=100)
        idx = 0
        for seq in seq_list:
            n = len(seq["embeddings"]) - window_size
            if n > 0:
                baseline_scores["lstm"][seq["grid_key"]] = lstm_all[idx:idx+n].tolist()
                idx += n
    except Exception as e:
        print(f"  LSTM failed: {e}")

    return baseline_scores


def load_coherence_maps(coherence_dir, grid_size=10):
    """Load coherence maps with memory-efficient block averaging.

    Coherence maps can be huge (12K x 18K pixels, ~850MB each).
    We use memory-mapped loading + strided block means to avoid OOM.
    """
    coherence_data = {}
    coh_dir = Path(coherence_dir)
    for fi, f in enumerate(sorted(coh_dir.glob("*.npy"))):
        parts = f.stem.split("_")
        date1, date2 = parts[2], parts[3]

        # Memory-map to avoid loading full array
        coh_map = np.load(f, mmap_mode='r')
        h, w = coh_map.shape
        cell_h, cell_w = h // grid_size, w // grid_size

        grid_values = np.zeros(grid_size * grid_size)
        for gy in range(grid_size):
            for gx in range(grid_size):
                # Read only this cell's data
                cell = coh_map[gy*cell_h:(gy+1)*cell_h, gx*cell_w:(gx+1)*cell_w]
                # Subsample for speed: take every 10th pixel
                sub = cell[::10, ::10]
                grid_values[gy * grid_size + gx] = np.nanmean(sub)

        coherence_data[f"{date1}_{date2}"] = {
            "values": grid_values, "date1": date1, "date2": date2,
        }
        print(f"  [{fi+1}] {f.name}: {h}x{w} -> grid mean coherence {grid_values.mean():.3f}")

    return coherence_data


def _date_diff(d1, d2):
    try:
        fmt1 = "%Y%m%d" if "-" not in str(d1) else "%Y-%m-%d"
        fmt2 = "%Y%m%d" if "-" not in str(d2) else "%Y-%m-%d"
        return abs((datetime.strptime(str(d1), fmt1) - datetime.strptime(str(d2), fmt2)).days)
    except:
        return 999


def compute_roc_pr_for_method(method_scores, model_results, coherence_data, aoi="hawaii"):
    """Compute ROC/PR for any method's per-grid scores against coherence."""
    all_scores = []
    all_coherence = []

    aoi_keys = [gk for gk in method_scores if gk.startswith(aoi)]

    for date_key, coh_data in coherence_data.items():
        date2_compact = coh_data["date2"].replace("-", "")

        for gk in aoi_keys:
            if gk not in model_results:
                continue
            dates = model_results[gk]["dates"]
            if not dates:
                continue

            parts = gk.split("_")
            gy, gx = int(parts[-2]), int(parts[-1])
            grid_idx = gy * 10 + gx

            scores_arr = method_scores[gk]
            if isinstance(scores_arr, list):
                scores_arr = np.array(scores_arr)

            best_idx, best_diff = None, 999
            for i, d in enumerate(dates):
                if i < len(scores_arr):
                    diff = _date_diff(d, date2_compact)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = i

            if best_idx is not None and best_diff <= 3:
                all_scores.append(float(scores_arr[best_idx]))
                all_coherence.append(coh_data["values"][grid_idx])

    if len(all_scores) == 0:
        return None

    all_scores = np.array(all_scores)
    all_coherence = np.array(all_coherence)

    results = {}
    for coh_thresh in [0.2, 0.3, 0.4, 0.5]:
        labels = coherence_to_labels(all_coherence, drop_threshold=coh_thresh)
        n_pos = labels.sum()
        if n_pos == 0 or n_pos == len(labels):
            continue
        metrics = compute_roc_pr(all_scores, labels)
        results[f"coh_{coh_thresh}"] = {
            "roc_auc": float(metrics["roc_auc"]),
            "pr_auc": float(metrics["pr_auc"]),
            "n_positive": int(n_pos),
            "n_total": len(labels),
        }
    return results


def load_stac_metadata(stac_cache_path):
    """Load STAC items and build date->satellite mapping per AOI."""
    with open(stac_cache_path) as f:
        items = json.load(f)

    metadata = {}  # {aoi: {date_str: {"satellite": str}}}
    for item in items:
        item_id = item["id"]
        sat = item_id.split("_")[1]  # e.g., C13
        dt_str = item.get("datetime", "")
        if dt_str:
            date = dt_str[:10].replace("-", "")  # YYYYMMDD
        else:
            # Extract from ID
            parts = item_id.split("_")
            date = parts[4][:8] if len(parts) > 4 else ""

        lon = (item["bbox"][0] + item["bbox"][2]) / 2
        if lon < -150:
            aoi = "hawaii"
        elif lon < -100:
            aoi = "la"
        elif lon > 100:
            aoi = "pilbara"
        else:
            continue

        prod_type = item.get("product_type", "")
        if prod_type != "GEO":
            continue

        if aoi not in metadata:
            metadata[aoi] = {}
        metadata[aoi][date] = {"satellite": sat}

    return metadata


def geometry_invariance(model_results, stac_metadata):
    """Correlate anomaly scores with satellite IDs per AOI."""
    from scipy.stats import pearsonr, spearmanr
    results = {}

    for aoi in ["hawaii", "la", "pilbara"]:
        aoi_meta = stac_metadata.get(aoi, {})
        if not aoi_meta:
            continue

        all_scores = []
        all_sats = []
        all_transitions = []

        aoi_keys = [gk for gk, d in model_results.items() if d["aoi"] == aoi]

        for gk in aoi_keys:
            data = model_results[gk]
            for i, (score, date) in enumerate(zip(data["scores"], data["dates"])):
                meta = aoi_meta.get(date, None)
                if meta:
                    all_scores.append(float(score))
                    all_sats.append(meta["satellite"])
                    # Transition: did satellite change from previous?
                    if i > 0:
                        prev_date = data["dates"][i-1]
                        prev_meta = aoi_meta.get(prev_date, None)
                        if prev_meta:
                            all_transitions.append(1 if meta["satellite"] != prev_meta["satellite"] else 0)
                        else:
                            all_transitions.append(0)
                    else:
                        all_transitions.append(0)

        if len(all_scores) < 10:
            continue

        # Satellite ID correlation (encode as numeric)
        unique_sats = sorted(set(all_sats))
        sat_to_idx = {s: i for i, s in enumerate(unique_sats)}
        sat_indices = np.array([sat_to_idx[s] for s in all_sats])
        scores_arr = np.array(all_scores)
        trans_arr = np.array(all_transitions[:len(all_scores)])

        r_sat, p_sat = spearmanr(scores_arr, sat_indices)
        r_trans, p_trans = pearsonr(scores_arr[:len(trans_arr)], trans_arr) if len(trans_arr) > 1 else (0, 1)

        results[aoi] = {
            "satellite_spearman_r": float(r_sat),
            "satellite_p_value": float(p_sat),
            "transition_pearson_r": float(r_trans),
            "transition_p_value": float(p_trans),
            "unique_satellites": unique_sats,
            "n_matched": len(all_scores),
            "n_transitions": int(sum(all_transitions)),
        }
        print(f"  [{aoi}] sat_rho={r_sat:.4f} (p={p_sat:.4f}), trans_r={r_trans:.4f} (p={p_trans:.4f}), "
              f"sats={unique_sats}, n={len(all_scores)}")

    return results


def cross_aoi_summary(model_results):
    """Per-AOI score statistics for cross-AOI table."""
    aoi_stats = {}
    for gk, data in model_results.items():
        aoi = data["aoi"]
        if aoi not in aoi_stats:
            aoi_stats[aoi] = []
        aoi_stats[aoi].extend(data["scores"].tolist())

    summary = {}
    for aoi, scores in aoi_stats.items():
        s = np.array(scores)
        summary[aoi] = {
            "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
            "median": float(np.median(s)), "n": len(s),
            "p80": float(np.percentile(s, 80)),
            "p90": float(np.percentile(s, 90)),
        }
    return summary


def run_permutation_per_aoi(model_results, grid_size=10):
    aois = set(data["aoi"] for data in model_results.values())
    results = {}
    for aoi in sorted(aois):
        aoi_keys = sorted(gk for gk, d in model_results.items() if d["aoi"] == aoi)
        if len(aoi_keys) == 0:
            continue
        min_len = min(len(model_results[gk]["scores"]) for gk in aoi_keys)
        if min_len == 0:
            continue
        score_matrix = np.zeros((len(aoi_keys), min_len))
        for i, gk in enumerate(aoi_keys):
            score_matrix[i] = model_results[gk]["scores"][:min_len]
        for pct in [75, 80, 85]:
            pr = permutation_test(score_matrix, threshold_percentile=pct, grid_size=grid_size)
            results[f"{aoi}_P{pct}"] = {
                "observed": float(pr["observed_coherence"]),
                "null_mean": float(pr["null_mean"]),
                "null_std": float(pr["null_std"]),
                "p_value": float(pr["p_value"]),
                "ratio": float(pr["observed_coherence"] / max(pr["null_mean"], 1e-6)),
            }
            print(f"  [{aoi} P{pct}] obs={pr['observed_coherence']:.4f} null={pr['null_mean']:.4f}±{pr['null_std']:.4f} p={pr['p_value']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coherence-dir", default="data/coherence/hawaii")
    parser.add_argument("--stac-cache", default="data/stac_items_cache.json")
    parser.add_argument("--output-dir", default="results/v24_complete")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_size = config["data"]["window_size"]

    ALL = {}

    # 1. Load model
    print("\n=== Loading Model ===")
    encoder, predictor = load_model(config, args.checkpoint, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ALL["model_info"] = {"epoch": ckpt["epoch"], "phase": ckpt["phase"], "val_loss": float(ckpt["val_loss"])}

    # 2. Build sequences
    print("\n=== Building Sequences ===")
    sequences = build_raw_sequences(config["data"]["patch_dir"], min_length=config["data"].get("min_sequence_length", 6))
    print(f"  {len(sequences)} sequences")

    # 3. Compute T-SAR-JEPA scores + embeddings
    print("\n=== Computing T-SAR-JEPA Scores ===")
    t0 = time.time()
    model_results = compute_all_scores(encoder, predictor, sequences, window_size, device, args.batch_size)
    print(f"  Done: {len(model_results)} sequences, {time.time()-t0:.1f}s")

    # 4. Cross-AOI summary
    print("\n=== Cross-AOI Summary ===")
    ALL["cross_aoi"] = cross_aoi_summary(model_results)
    for aoi, stats in ALL["cross_aoi"].items():
        print(f"  {aoi}: mean={stats['mean']:.4f} std={stats['std']:.4f} n={stats['n']}")

    # 5. Baselines (per-grid)
    print("\n=== Baselines ===")
    baseline_scores = compute_baseline_scores_per_grid(model_results, window_size)

    # 6. Load coherence
    print("\n=== Loading Coherence ===")
    coherence_data = load_coherence_maps(args.coherence_dir)
    print(f"  {len(coherence_data)} coherence maps")

    # 7. ROC/PR for ALL methods
    print("\n=== ROC/PR for All Methods ===")
    tsar_scores = {gk: data["scores"] for gk, data in model_results.items()}
    all_roc_pr = {}

    for method_name, scores_dict in [("t_sar_jepa", tsar_scores)] + list(baseline_scores.items()):
        roc = compute_roc_pr_for_method(scores_dict, model_results, coherence_data, aoi="hawaii")
        if roc:
            all_roc_pr[method_name] = roc
            best = roc.get("coh_0.2", {})
            print(f"  {method_name}: ROC-AUC={best.get('roc_auc', 0):.4f}, PR-AUC={best.get('pr_auc', 0):.4f}")
        else:
            print(f"  {method_name}: no matched pairs")
            all_roc_pr[method_name] = {}
    ALL["roc_pr_all_methods"] = all_roc_pr

    # 8. Permutation test
    print("\n=== Permutation Test ===")
    ALL["permutation_test"] = run_permutation_per_aoi(model_results)

    # 9. Threshold sweep
    print("\n=== Threshold Sweep ===")
    all_scores_flat = np.concatenate([d["scores"] for d in model_results.values()])
    ALL["threshold_sweep"] = {}
    for pct in [70, 75, 80, 85, 90]:
        thresh = np.percentile(all_scores_flat, pct)
        ALL["threshold_sweep"][f"P{pct}"] = {
            "value": float(thresh),
            "n_flagged": int((all_scores_flat >= thresh).sum()),
            "pct_flagged": float((all_scores_flat >= thresh).mean() * 100),
        }

    # 10. Geometry invariance
    print("\n=== Geometry Invariance ===")
    if os.path.exists(args.stac_cache):
        stac_meta = load_stac_metadata(args.stac_cache)
        ALL["geometry"] = geometry_invariance(model_results, stac_meta)
    else:
        print("  STAC cache not found, skipping geometry analysis")
        ALL["geometry"] = {}

    # 11. Baseline aggregate stats
    ALL["baseline_stats"] = {}
    for method, scores_dict in baseline_scores.items():
        if scores_dict:
            flat = np.concatenate([np.array(v) for v in scores_dict.values()])
            ALL["baseline_stats"][method] = {
                "mean": float(flat.mean()), "std": float(flat.std()),
                "min": float(flat.min()), "max": float(flat.max()), "n": len(flat),
            }

    # === SAVE EVERYTHING ===
    print("\n=== Saving ===")

    with open(output_dir / "results_complete.json", "w") as f:
        json.dump(ALL, f, indent=2, default=str)

    # Save per-grid scores for T-SAR-JEPA
    for aoi in ["hawaii", "la", "pilbara"]:
        aoi_data = {}
        for gk, data in model_results.items():
            if data["aoi"] == aoi:
                aoi_data[gk] = {"scores": data["scores"].tolist(), "dates": data["dates"]}
        with open(output_dir / f"scores_{aoi}.json", "w") as f:
            json.dump(aoi_data, f, indent=2)

    # Save per-grid baseline scores
    for method, scores_dict in baseline_scores.items():
        with open(output_dir / f"baseline_{method}_scores.json", "w") as f:
            json.dump(scores_dict, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("T-SAR-JEPA — Complete Results")
    print("=" * 70)
    print(f"\nModel: epoch={ALL['model_info']['epoch']}, val_loss={ALL['model_info']['val_loss']:.6f}")

    print("\n--- ROC-AUC (Hawaii, coh<0.2) ---")
    for method, roc in all_roc_pr.items():
        r = roc.get("coh_0.2", {})
        print(f"  {method:15s}: ROC-AUC={r.get('roc_auc', 0):.4f}  PR-AUC={r.get('pr_auc', 0):.4f}")

    print("\n--- Cross-AOI ---")
    for aoi, stats in ALL["cross_aoi"].items():
        print(f"  {aoi:10s}: mean={stats['mean']:.4f} +/- {stats['std']:.4f}")

    if ALL["geometry"]:
        print("\n--- Geometry Invariance ---")
        for aoi, geo in ALL["geometry"].items():
            print(f"  {aoi:10s}: sat_rho={geo['satellite_spearman_r']:.4f} (p={geo['satellite_p_value']:.4f}), "
                  f"sats={geo['unique_satellites']}")

    print(f"\nAll results saved to {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
