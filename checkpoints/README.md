# Checkpoints

Checkpoints are not included in the repo due to size.

## Pre-trained Weights
- SAR-JEPA checkpoint-200: [Download from SAR-JEPA repo](https://github.com/waterdisappear/SAR-JEPA)

## Trained Checkpoints
To reproduce, run the full pipeline:
1. Stage 1: `python training/pretrain.py --config configs/stage1_pretrain.yaml`
2. Stage 2: `python training/train_temporal.py --config configs/stage2_temporal.yaml`
3. Stage 3: `python training/finetune_e2e.py --config configs/stage3_e2e.yaml`
