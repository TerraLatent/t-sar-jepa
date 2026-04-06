# T-SAR-JEPA: Temporal Self-Supervised Anomaly Detection in SAR Amplitude Stacks via Latent Prediction

[![Paper](https://img.shields.io/badge/Paper-IEEE%20GRSS%20DFC%202026-blue)](https://www.grss-ieee.org/community/technical-committees/2026-ieee-grss-data-fusion-contest/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)

## Overview

T-SAR-JEPA is a self-supervised framework for detecting temporal anomalies in SAR amplitude stacks using JEPA-based latent prediction. Operating exclusively on single-channel SAR amplitude data, it learns to predict future latent representations of SAR patches and flags frames whose prediction error deviates significantly from the temporal norm. InSAR coherence is used strictly as an independent evaluation reference and is never seen during training or inference. The framework follows a 3-stage pipeline: (1) domain adaptation of pre-trained SAR-JEPA weights to the target SAR amplitude distribution, (2) temporal prediction training where a temporal module learns to forecast latent representations across time, and (3) progressive unfreezing that jointly fine-tunes the full encoder-predictor stack end-to-end.

## Key Results

| Metric | T-SAR-JEPA | Baselines |
|--------|-----------|-----------|
| ROC-AUC vs coherence pseudo-GT | **77.0%** | ~50% |
| Spatial coherence (permutation test) | **99.9%** (p<0.001, 1000-shuffle) | — |
| Val loss improvement (progressive unfreezing) | **50x** | — |
| Geometry independence | \|ρ\|<0.11 across all AOIs | — |

## Architecture

T-SAR-JEPA employs a three-stage training pipeline built on top of the SAR-JEPA Vision Transformer backbone:

1. **Stage 1 — Domain Adaptation**: The pre-trained SAR-JEPA encoder is adapted to the target Capella SAR amplitude distribution using masked image modeling with an exponential moving average (EMA) target encoder.

2. **Stage 2 — Temporal Prediction**: A 4-layer causal transformer with sinusoidal time encoding is trained on top of the frozen adapted encoder. Given a context window of K=7 SAR amplitude patches over time, it predicts the latent representation of the next timestep. Anomaly scores are L2 prediction errors.

3. **Stage 3 — Progressive Unfreezing**: The encoder is unfrozen and jointly fine-tuned with the temporal predictor end-to-end, yielding representations that are simultaneously good for spatial encoding and temporal prediction.

## Installation

```bash
git clone https://github.com/kerod/t-sar-jepa.git
cd t-sar-jepa
pip install -r requirements.txt
```

## Data Download

The DFC 2026 Capella SAR dataset is publicly available via STAC catalog on AWS S3 (unsigned access).

Download SAR data for a specific AOI:
```bash
python scripts/download_data.py --aoi hawaii --output-dir data/geo/hawaii
```

Extract patches from downloaded GeoTIFFs:
```bash
python scripts/extract_patches.py --geo-dir data/geo --output-dir data/patches
```

Available AOIs: `hawaii`, `la`, `pilbara`

## Training

### Stage 1: Domain Adaptation
```bash
python training/pretrain.py --config configs/stage1_pretrain.yaml
```

### Encode Patches (between Stage 1 and Stage 2)
```bash
python training/encode_dataset.py --config configs/stage1_pretrain.yaml
```

### Stage 2: Temporal Prediction
```bash
python training/train_temporal.py --config configs/stage2_temporal.yaml
```

### Stage 3: Progressive Unfreezing (End-to-End)
```bash
python training/finetune_e2e.py --config configs/stage3_e2e.yaml
```

## Evaluation

```bash
python scripts/run_evaluation.py --config configs/stage3_e2e.yaml --checkpoint checkpoints/stage3/best_e2e.pt
```

## Repository Structure

```
t-sar-jepa/
├── configs/                    # YAML configs for all stages + ablations
│   ├── stage1_pretrain.yaml    # Stage 1: domain adaptation
│   ├── stage2_temporal.yaml    # Stage 2: temporal predictor (K=7, sinusoidal)
│   ├── stage3_e2e.yaml         # Stage 3: progressive unfreezing
│   ├── ablation_k_*.yaml       # K ablations (3, 5, 7, 9)
│   ├── ablation_te_*.yaml      # Time encoding ablations (sinusoidal, ctlpe, linear)
│   └── selected_aois.json      # AOI bounding boxes (Hawaii, LA, Pilbara)
├── models/                     # Model definitions
│   ├── encoder.py              # SAR-JEPA ViT-Base/16 encoder wrapper
│   ├── temporal_predictor.py   # 4-layer causal transformer
│   ├── time_encodings.py       # Sinusoidal, CTLPE, linear encodings
│   └── t_sar_jepa.py           # Combined pipeline wrapper
├── training/                   # Training scripts for all 3 stages
│   ├── pretrain.py             # Stage 1: domain adaptation
│   ├── train_temporal.py       # Stage 2: temporal predictor
│   ├── finetune_e2e.py         # Stage 3: progressive unfreezing
│   └── encode_dataset.py       # Encode patches through frozen encoder
├── baselines/                  # Baseline methods (RX, PaDiM, Linear AR, LSTM)
├── evaluation/                 # Metrics (ROC/PR, permutation test, geometry, cross-AOI)
├── inference/                  # Inference pipeline and visualization
├── data/                       # Data loading modules (datasets, patch extraction, coherence)
├── sarjepa/                    # Vendored SAR-JEPA code (CC BY-NC 4.0)
├── scripts/                    # Entry points
│   ├── download_data.py        # Download GEO products from STAC catalog
│   ├── extract_patches.py      # Extract amplitude patches from GeoTIFFs
│   ├── compute_coherence.py    # Compute InSAR coherence for evaluation
│   ├── run_evaluation.py       # Full evaluation suite
│   ├── generate_figures.py     # Paper figure generation
│   └── run_pipeline.sh         # End-to-end pipeline runner
├── results/                    # Pre-computed results for verification
├── checkpoints/                # Model weights (not in repo, see README)
├── LICENSE                     # Apache 2.0
├── README.md
└── requirements.txt
```

## Hardware Requirements
- Training was performed on a single NVIDIA H200 (140 GB VRAM)
- Stage 1 and Stage 3 require significant VRAM (batch 256, ViT-Base/16 on 224x224)
- For smaller GPUs, reduce batch size in the config files
- Evaluation and inference can run on any GPU with >= 8 GB VRAM

## Citation

```bibtex
@inproceedings{woldesenbet2026tsarjepa,
  title={T-SAR-JEPA: Temporal Self-Supervised Anomaly Detection in SAR Amplitude Stacks via Latent Prediction},
  author={Woldesenbet, Kerod and Woldesenbet, Abem},
  booktitle={IEEE GRSS Data Fusion Contest, IGARSS 2026},
  year={2026}
}
```

## License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

**Note:** The `sarjepa/` directory contains vendored code from [SAR-JEPA](https://github.com/waterdisappear/SAR-JEPA), which is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). See [sarjepa/LICENSE](sarjepa/LICENSE) for details.

## Acknowledgments

- [SAR-JEPA](https://github.com/waterdisappear/SAR-JEPA) — Pre-trained SAR foundation model backbone
- [Capella Space](https://www.capellaspace.com/) — SAR imagery for the DFC 2026 dataset
- [IEEE GRSS Data Fusion Contest 2026](https://www.grss-ieee.org/community/technical-committees/2026-ieee-grss-data-fusion-contest/) — Contest organization and dataset curation
