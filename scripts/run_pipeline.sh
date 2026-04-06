#!/bin/bash
# T-SAR-JEPA: Full 3-Stage Training Pipeline
#
# Usage:
#   cd /path/to/t-sar-jepa
#   bash scripts/run_pipeline.sh
#
# Prerequisites:
#   - Data downloaded and patches extracted (see scripts/download_data.py, extract_patches.py)
#   - SAR-JEPA pretrained checkpoint at checkpoints/sarjepa_pretrained/checkpoint-200.pth

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================"
echo "T-SAR-JEPA Training Pipeline"
echo "Project root: $PROJECT_ROOT"
echo "============================================"

# -----------------------------------------------
# Stage 1: Domain Adaptation (SAR-JEPA pre-training)
#
# Fine-tunes a LoMaR/I-JEPA encoder on Capella SAR patches
# using masked image modeling with gradient-based targets.
# -----------------------------------------------
echo ""
echo "=== Stage 1: Domain Adaptation ==="
echo "Config: configs/stage1_pretrain.yaml"

python training/pretrain.py \
    --config configs/stage1_pretrain.yaml

echo "Stage 1 complete. Checkpoint: checkpoints/stage1_v24/"

# -----------------------------------------------
# Stage 2: Temporal Predictor Training
#
# Freezes the encoder and trains a temporal transformer
# to predict the next embedding from a context window (K=7).
# Uses sinusoidal time encodings and smooth L1 loss.
# -----------------------------------------------
echo ""
echo "=== Stage 2: Temporal Predictor ==="
echo "Config: configs/stage2_temporal.yaml"

# First, encode all patches with the Stage 1 encoder
python training/encode_patches.py \
    --checkpoint checkpoints/stage1_v24/checkpoint_latest.pth \
    --patch-dir data/patches \
    --output-dir data/encoded_v24

# Train the temporal predictor
python training/train_temporal.py \
    --config configs/stage2_temporal.yaml

echo "Stage 2 complete. Checkpoint: checkpoints/stage2_temporal/"

# -----------------------------------------------
# Stage 3: End-to-End Fine-tuning
#
# Progressively unfreezes the encoder while jointly training
# with the temporal predictor. Two phases:
#   Phase A (30 epochs): Unfreeze last 4 encoder blocks
#   Phase B (20 epochs): Unfreeze all layers, lower LR
# -----------------------------------------------
echo ""
echo "=== Stage 3: End-to-End Fine-tuning ==="
echo "Config: configs/stage3_e2e.yaml"

python training/finetune_e2e.py \
    --config configs/stage3_e2e.yaml

echo "Stage 3 complete. Checkpoint: checkpoints/stage3_v24/"

# -----------------------------------------------
# Evaluation
# -----------------------------------------------
echo ""
echo "=== Running Evaluation ==="

python scripts/run_evaluation.py \
    --config configs/stage3_e2e.yaml \
    --checkpoint checkpoints/stage3_v24/best_e2e.pt \
    --output-dir results/

echo ""
echo "============================================"
echo "Pipeline complete. Results saved to results/"
echo "============================================"
