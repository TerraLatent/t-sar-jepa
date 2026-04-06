"""Cross-AOI generalization evaluation."""
import numpy as np

def cross_aoi_summary(results_per_aoi: dict) -> dict:
    """Summarize cross-AOI results. Input: {aoi_name: {roc_auc, pr_auc, coherence}}."""
    aoi_names = sorted(results_per_aoi.keys())
    summary = {"per_aoi": {}, "aggregate": {}}
    for metric in ["roc_auc", "pr_auc", "coherence"]:
        values = [results_per_aoi[aoi].get(metric, 0) for aoi in aoi_names]
        summary["aggregate"][f"mean_{metric}"] = float(np.mean(values))
        summary["aggregate"][f"std_{metric}"] = float(np.std(values))
        for aoi, val in zip(aoi_names, values):
            summary["per_aoi"].setdefault(aoi, {})[metric] = val
    return summary
