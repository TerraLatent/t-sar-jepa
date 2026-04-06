# Results

Pre-computed results are available in this directory for verification and reproducibility.

## Files

- `results_complete.json` — Full evaluation results including ROC/PR, baselines, permutation test, geometry invariance
- `scores_hawaii.json` — Per-patch anomaly scores for Hawaii AOI
- `scores_la.json` — Per-patch anomaly scores for Los Angeles AOI
- `scores_pilbara.json` — Per-patch anomaly scores for Pilbara AOI

## Key Metrics

| AOI | ROC-AUC | Spatial Coherence |
|-----|---------|-------------------|
| Hawaii | See results_complete.json | p<0.001 |
| LA | See results_complete.json | p<0.001 |
| Pilbara | See results_complete.json | p<0.001 |
| **Overall** | **77.0%** | **99.9%** |
