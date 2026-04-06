"""Stage 1: Domain adaptation pretraining of SAR-JEPA encoder on Capella data.

Fine-tunes the full MaskedAutoencoderViT on Capella SAR patches using the
spatial JEPA objective (LoMaR masking + multi-scale gradient feature targets).

The model's forward() handles masking and loss computation internally:
    loss, pred, mask = model(imgs, window_size, num_window, mask_ratio)

Reconstruction targets are multi-scale gradient features (GF at kernel sizes
5, 9, 13, 17) computed from the raw input images. This is NOT a JEPA with a
target encoder. The EMA model is maintained as a smoothed checkpoint for
downstream use (encoding), but does NOT participate in the loss computation.
"""

import copy
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_SARJEPA_DIR = str(Path(_PROJECT_ROOT) / "sarjepa")


def _load_config(config_path: str) -> dict:
    """Load YAML config file."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_full_model(img_size: int = 224, in_chans: int = 1) -> nn.Module:
    """Build the full MaskedAutoencoderViT with sarjepa on sys.path."""
    prev_path = sys.path.copy()
    try:
        if _SARJEPA_DIR not in sys.path:
            sys.path.insert(0, _SARJEPA_DIR)
        from models_lomar import mae_vit_base_patch16
        model = mae_vit_base_patch16(img_size=img_size, in_chans=in_chans)
    finally:
        sys.path = prev_path
    return model


def load_pretrained_weights(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load pretrained weights into the full MaskedAutoencoderViT."""
    if not os.path.exists(checkpoint_path):
        print(f"[pretrain] No checkpoint at {checkpoint_path}, training from scratch")
        return model

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[pretrain] Unexpected keys: {unexpected[:5]}...")
    if missing:
        print(f"[pretrain] Missing keys: {missing[:5]}...")
    print(f"[pretrain] Loaded pretrained weights from {checkpoint_path}")
    return model


def create_ema_model(model: nn.Module) -> nn.Module:
    """Create an EMA copy of the model with no gradients."""
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad = False
    return ema


@torch.no_grad()
def update_ema(online: nn.Module, ema: nn.Module, momentum: float) -> None:
    """Exponential moving average update: p_ema = m * p_ema + (1-m) * p_online."""
    for p_online, p_ema in zip(online.parameters(), ema.parameters()):
        p_ema.data.mul_(momentum).add_(p_online.data, alpha=1.0 - momentum)


def cosine_lr_schedule(optimizer, epoch: int, total_epochs: int,
                       warmup_epochs: int, base_lr: float, min_lr: float = 1e-6):
    """Cosine learning rate schedule with linear warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(model: nn.Module, ema_model: nn.Module, optimizer,
                    epoch: int, loss: float, output_dir: str, tag: str = ""):
    """Save training checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{tag}" if tag else ""
    path = output_dir / f"checkpoint_epoch{epoch:04d}{suffix}.pth"

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"[pretrain] Saved checkpoint: {path}")

    # Also save a "latest" symlink-style copy
    latest_path = output_dir / "checkpoint_latest.pth"
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": loss,
    }, latest_path)


def load_resume_checkpoint(model, ema_model, optimizer, checkpoint_dir: str):
    """Resume training from latest checkpoint if available."""
    latest = Path(checkpoint_dir) / "checkpoint_latest.pth"
    if not latest.exists():
        return 0

    ckpt = torch.load(latest, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    ema_model.load_state_dict(ckpt["ema_model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    print(f"[pretrain] Resumed from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")
    return start_epoch


def train_stage1(config_path: str):
    """Main Stage 1 domain adaptation training loop.

    Loads pretrained SAR-JEPA, fine-tunes on Capella SAR patches using the
    built-in spatial JEPA objective (LoMaR masking + gradient feature targets).
    """
    config = _load_config(config_path)

    # Unpack config
    patch_dir = config["data"]["patch_dir"]
    patch_size = config["data"].get("patch_size", 224)
    batch_size = config["data"]["batch_size"]

    checkpoint_path = config["model"]["checkpoint"]

    epochs = config["training"]["epochs"]
    base_lr = float(config["training"]["lr"])
    weight_decay = float(config["training"]["weight_decay"])
    warmup_epochs = config["training"]["warmup_epochs"]
    beta1 = float(config["training"].get("beta1", 0.9))
    beta2 = float(config["training"].get("beta2", 0.95))
    ema_momentum = float(config["training"].get("ema_momentum", 0.996))

    # Masking parameters (LoMaR)
    window_size = config["training"].get("window_size", 7)
    num_window = config["training"].get("num_window", 4)
    mask_ratio = float(config["training"].get("mask_ratio", 0.8))

    checkpoint_dir = config["output"]["checkpoint_dir"]
    save_every = config["output"]["save_every"]
    log_dir = config["output"].get("log_dir", "logs/stage1")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pretrain] Device: {device}")

    # Build model
    # The model expects img_size matching the patch size used during extraction.
    # SAR-JEPA base uses patch16 with img_size=224 by default. If our patches
    # are a different size (e.g., 128), we must instantiate accordingly.
    # For non-standard sizes, window_size must be <= img_size // patch_size.
    model = _build_full_model(img_size=patch_size, in_chans=1)
    model = load_pretrained_weights(model, checkpoint_path)
    model = model.to(device)

    # EMA model (smoothed weights for downstream encoding, not used in loss)
    ema_model = create_ema_model(model)
    ema_model = ema_model.to(device)

    # Dataset & dataloader
    from data.pretrain_dataset import PretrainPatchDataset

    dataset = PretrainPatchDataset(patch_dir=patch_dir)
    print(f"[pretrain] Dataset: {len(dataset)} patches from {patch_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    # Resume from checkpoint if available
    start_epoch = load_resume_checkpoint(model, ema_model, optimizer, checkpoint_dir)

    # Training loop
    print(f"[pretrain] Starting Stage 1 training: epochs {start_epoch}-{epochs-1}")
    print(f"[pretrain] LoMaR params: window_size={window_size}, num_window={num_window}, mask_ratio={mask_ratio}")

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train_log.csv")
    if start_epoch == 0:
        with open(log_file, "w") as f:
            f.write("epoch,loss,lr,time_s\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        lr = cosine_lr_schedule(optimizer, epoch, epochs, warmup_epochs, base_lr)

        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch_idx, patches in enumerate(dataloader):
            patches = patches.to(device, non_blocking=True)

            # Forward: model handles masking and gradient-feature loss internally
            loss, pred, mask_indices = model(
                patches,
                window_size=window_size,
                num_window=num_window,
                mask_ratio=mask_ratio,
            )
            loss = loss.mean()

            if not math.isfinite(loss.item()):
                print(f"[pretrain] Loss is {loss.item()} at epoch {epoch} batch {batch_idx}, skipping")
                optimizer.zero_grad()
                continue

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # EMA update
            update_ema(model, ema_model, ema_momentum)

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 20 == 0:
                print(f"  [epoch {epoch}] batch {batch_idx}/{len(dataloader)} "
                      f"loss={loss.item():.4f} lr={lr:.6f}")

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"[pretrain] Epoch {epoch}: avg_loss={avg_loss:.4f} lr={lr:.6f} time={elapsed:.1f}s")

        # Log
        with open(log_file, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{lr:.8f},{elapsed:.1f}\n")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            save_checkpoint(model, ema_model, optimizer, epoch, avg_loss, checkpoint_dir)

    # Save final EMA model in a format compatible with SARJEPAEncoder loading
    final_path = Path(checkpoint_dir) / "stage1_final.pth"
    torch.save({"model": ema_model.state_dict()}, final_path)
    print(f"[pretrain] Saved final EMA model: {final_path}")
    print("[pretrain] Stage 1 training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: SAR-JEPA domain adaptation")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml",
                        help="Path to pretrain config YAML")
    args = parser.parse_args()
    train_stage1(args.config)
