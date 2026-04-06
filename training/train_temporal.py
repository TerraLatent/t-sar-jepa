"""Stage 2: Train temporal predictor on pre-encoded sequences.
Lightweight - runs on RTX 3070 since it operates on 768-dim vectors, not images.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.temporal_predictor import TemporalPredictor
from data.temporal_dataset import TemporalPatchDataset
from training.encode_dataset import build_temporal_sequences


def train_stage2(config_path: str) -> None:
    """Run Stage 2 training: temporal predictor on pre-encoded embeddings.

    Args:
        config_path: Path to temporal.yaml config file.
    """
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    output_cfg = cfg["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage2] Device: {device}")

    # Build temporal sequences from pre-encoded embeddings
    print("[stage2] Building temporal sequences...")
    sequences = build_temporal_sequences(
        embeddings_dir=data_cfg["embeddings_dir"],
        min_length=data_cfg.get("min_sequence_length", 10),
    )

    if not sequences:
        print("[stage2] No sequences found. Ensure Stage 1 encoding is complete.")
        return

    # Spatial split by grid location
    grid_keys = list(set(seq["grid_key"] for seq in sequences))
    grid_keys.sort()
    n_train_locs = int(0.8 * len(grid_keys))
    train_keys = set(grid_keys[:n_train_locs])
    val_keys = set(grid_keys[n_train_locs:])

    train_sequences = [s for s in sequences if s["grid_key"] in train_keys]
    val_sequences = [s for s in sequences if s["grid_key"] in val_keys]

    train_dataset = TemporalPatchDataset(train_sequences, window_size=data_cfg["window_size"])
    val_dataset = TemporalPatchDataset(val_sequences, window_size=data_cfg["window_size"])
    print(f"[stage2] Spatial split: {len(train_keys)} train locs ({len(train_dataset)} windows), "
          f"{len(val_keys)} val locs ({len(val_dataset)} windows)")

    if len(train_dataset) == 0:
        print("[stage2] No valid training windows. Sequences may be too short for window_size.")
        return

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Initialize model
    time_encoding_type = model_cfg.get("time_encoding_type", "sinusoidal")
    model = TemporalPredictor(
        embed_dim=model_cfg["embed_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ffn_dim=model_cfg["ffn_dim"],
        dropout=model_cfg["dropout"],
        time_encoding_type=time_encoding_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[stage2] TemporalPredictor: {n_params:,} trainable parameters")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    total_steps = train_cfg["epochs"] * len(train_loader)
    warmup_steps = train_cfg.get("warmup_steps", 500)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss
    loss_type = train_cfg.get("loss_type", "smooth_l1")
    if loss_type == "l1":
        criterion = nn.L1Loss()
    elif loss_type == "smooth_l1":
        criterion = nn.SmoothL1Loss()
    elif loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_type}")
    print(f"[stage2] Loss: {loss_type}")

    # Output dirs
    ckpt_dir = Path(output_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_every = output_cfg.get("save_every", 10)

    # Training loop
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, train_cfg["epochs"] + 1):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_steps = 0

        for context, target, time_enc in train_loader:
            context = context.to(device)
            target = target.to(device)
            time_enc = time_enc.to(device)

            prediction = model(context, time_enc)
            loss = criterion(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0

        with torch.no_grad():
            for context, target, time_enc in val_loader:
                context = context.to(device)
                target = target.to(device)
                time_enc = time_enc.to(device)

                prediction = model(context, time_enc)
                loss = criterion(prediction, target)

                val_loss_sum += loss.item()
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"[stage2] Epoch {epoch:03d}/{train_cfg['epochs']} | "
            f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | "
            f"lr={lr_now:.2e}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_path = ckpt_dir / "best_temporal.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, best_path)
            print(f"[stage2] Saved best model (val_loss={avg_val_loss:.6f}) -> {best_path}")

        # Periodic checkpoints
        if epoch % save_every == 0:
            ckpt_path = ckpt_dir / f"temporal_epoch{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, ckpt_path)
            print(f"[stage2] Checkpoint -> {ckpt_path}")

    print(f"\n[stage2] Training complete. Best val_loss={best_val_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Train temporal predictor")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/temporal.yaml",
        help="Path to temporal config YAML",
    )
    args = parser.parse_args()
    train_stage2(args.config)
