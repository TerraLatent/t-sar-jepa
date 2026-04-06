"""Stage 3: End-to-end fine-tuning with progressive encoder unfreezing.

Addresses the reviewer question "why freeze the encoder?" by demonstrating
that joint fine-tuning of the encoder + temporal predictor further improves
temporal prediction quality.

Two-phase progressive unfreezing:
  Phase A: Unfreeze last 4 encoder blocks + norm. Encoder LR = 1e-5, predictor LR = 1e-4.
  Phase B: Unfreeze ALL encoder layers.         Encoder LR = 5e-6, predictor LR = 5e-5.

Differences from Stage 2:
  - Operates on raw single-channel amplitude patches, NOT pre-encoded embeddings.
  - Encodes each patch through the *trainable* encoder during the forward pass.
  - Backprop flows through both encoder and predictor.
  - Gradient clipping at 1.0.
"""

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

# Ensure project root is on sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.encoder import SARJEPAEncoder
from models.temporal_predictor import TemporalPredictor


# ---------------------------------------------------------------------------
# Raw-patch dataset (Stage 3 specific — no pre-encoded embeddings)
# ---------------------------------------------------------------------------

class RawPatchTemporalDataset(Dataset):
    """Sliding-window dataset over raw single-channel amplitude SAR patches.

    Each window yields:
        patches   : (window_size, 1, H, W)  float32 — the context patches
        target    : (1, H, W)               float32 — the next raw patch
        time_enc  : (window_size,)           float32 — normalized day values in [0, 1]

    The encoder is applied inside the training loop (not here) so that
    gradients flow through it.

    Filename convention: {aoi}_{gridx}_{gridy}_{timestamp}.npy
        timestamp: YYYYMMDD or YYYY-MM-DD
    """

    def __init__(self, sequences: List[Dict], window_size: int = 5):
        self.window_size = window_size
        # Each element: (list_of_patch_paths, target_path, days_array)
        self.windows: List[Tuple[List[Path], Path, np.ndarray]] = []

        for seq in sequences:
            paths = seq["paths"]   # list of Path, sorted by date
            days = seq["days"]     # np.ndarray (seq_len,)
            n = len(paths)
            if n <= window_size:
                continue
            for i in range(n - window_size):
                context_paths = paths[i : i + window_size]
                target_path = paths[i + window_size]
                window_days = days[i : i + window_size]
                self.windows.append((context_paths, target_path, window_days))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context_paths, target_path, window_days = self.windows[idx]

        # Load context patches
        context_patches = []
        for p in context_paths:
            patch = _load_patch(p)
            context_patches.append(patch)
        context = np.stack(context_patches, axis=0)  # (window_size, 1, H, W)

        # Load target patch
        target = _load_patch(target_path)  # (1, H, W)

        # Normalize time within window to [0, 1]
        d = window_days.astype(np.float32)
        d_min, d_max = d[0], d[-1]
        if d_max > d_min:
            time_enc = (d - d_min) / (d_max - d_min)
        else:
            time_enc = np.zeros(len(d), dtype=np.float32)

        return (
            torch.from_numpy(context),
            torch.from_numpy(target),
            torch.from_numpy(time_enc),
        )


def _load_patch(path: Path) -> np.ndarray:
    """Load a .npy patch and ensure shape (C, H, W) float32."""
    patch = np.load(path).astype(np.float32)
    if patch.ndim == 2:
        # Single channel -> (1, H, W)
        patch = patch[np.newaxis, :, :]
    return patch


# ---------------------------------------------------------------------------
# Data discovery: build raw-patch sequences from patch directory
# ---------------------------------------------------------------------------

def build_raw_sequences(patch_dir: str, min_length: int = 6) -> List[Dict]:
    """Scan patch_dir for .npy files and group by grid location.

    Returns list of dicts with keys:
        paths    : list[Path] sorted by timestamp
        days     : np.ndarray(seq_len,) days since first acquisition
        grid_key : str "{aoi}_{gridx}_{gridy}"
    """
    patch_dir = Path(patch_dir)
    files = sorted(patch_dir.glob("*.npy"))
    if not files:
        print(f"[stage3] No .npy patches found in {patch_dir}")
        return []

    groups: Dict[str, List[dict]] = {}
    for f in files:
        stem = f.stem
        parts = stem.rsplit("_", 3)
        if len(parts) < 4:
            print(f"[stage3] Skipping malformed filename: {f.name}")
            continue
        aoi, gridx, gridy, timestamp = parts[-4], parts[-3], parts[-2], parts[-1]
        grid_key = f"{aoi}_{gridx}_{gridy}"
        groups.setdefault(grid_key, []).append({"path": f, "timestamp": timestamp})

    sequences = []
    for grid_key, entries in groups.items():
        if len(entries) < min_length:
            continue

        # Sort by timestamp string (YYYYMMDD sorts lexicographically)
        entries.sort(key=lambda e: e["timestamp"])

        # Parse timestamps into datetime objects for day-delta computation
        parsed = []
        for e in entries:
            ts = e["timestamp"]
            for fmt in ("%Y%m%d", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(ts, fmt)
                    parsed.append((dt, e["path"]))
                    break
                except ValueError:
                    continue
            else:
                print(f"[stage3] Cannot parse timestamp '{ts}', skipping entry")

        if len(parsed) < min_length:
            continue

        first_date = parsed[0][0]
        days = np.array([(dt - first_date).days for dt, _ in parsed], dtype=np.float32)
        paths = [p for _, p in parsed]

        sequences.append({"paths": paths, "days": days, "grid_key": grid_key})

    print(f"[stage3] Found {len(sequences)} valid grid sequences "
          f"(min_length={min_length}) from {len(groups)} locations")
    return sequences


# ---------------------------------------------------------------------------
# Learning-rate helpers
# ---------------------------------------------------------------------------

def _cosine_lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Unfreezing helpers
# ---------------------------------------------------------------------------

def freeze_encoder(encoder: SARJEPAEncoder) -> None:
    """Freeze all encoder parameters."""
    for param in encoder.parameters():
        param.requires_grad = False


def unfreeze_phase_a(encoder: SARJEPAEncoder) -> None:
    """Phase A: unfreeze last 4 transformer blocks + final norm layer."""
    freeze_encoder(encoder)  # start fully frozen, then selectively unfreeze
    for blk in encoder.blocks[-4:]:
        for param in blk.parameters():
            param.requires_grad = True
    for param in encoder.norm.parameters():
        param.requires_grad = True


def unfreeze_phase_b(encoder: SARJEPAEncoder) -> None:
    """Phase B: unfreeze ALL encoder parameters."""
    for param in encoder.parameters():
        param.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# One epoch of training / validation
# ---------------------------------------------------------------------------

def run_epoch(
    encoder: SARJEPAEncoder,
    predictor: TemporalPredictor,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    grad_clip: float = 1.0,
) -> float:
    """Run one pass over the data loader.

    Returns:
        Average loss over all batches.
    """
    encoder.train(train)
    predictor.train(train)

    total_loss = 0.0
    n_steps = 0

    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for context_patches, target_patches, time_enc in loader:
            # context_patches : (B, window_size, 1, H, W)
            # target_patches  : (B, 1, H, W)
            # time_enc        : (B, window_size)
            context_patches = context_patches.to(device)
            target_patches = target_patches.to(device)
            time_enc = time_enc.to(device)

            B, W, C, H, Wp = context_patches.shape

            # Encode all context patches: flatten (B, W) -> (B*W, C, H, Wh)
            flat_patches = context_patches.view(B * W, C, H, Wp)
            flat_embs = encoder(flat_patches)           # (B*W, embed_dim)
            context_embs = flat_embs.view(B, W, -1)    # (B, window_size, embed_dim)

            # Encode target patches
            target_embs = encoder(target_patches)       # (B, embed_dim)

            # Temporal prediction
            prediction = predictor(context_embs, time_enc)  # (B, embed_dim)

            loss = criterion(prediction, target_embs)

            if train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping across both models
                all_params = list(encoder.parameters()) + list(predictor.parameters())
                nn.utils.clip_grad_norm_(
                    [p for p in all_params if p.requires_grad],
                    grad_clip,
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            n_steps += 1

    return total_loss / max(n_steps, 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_stage3(config_path: str) -> None:
    """Run Stage 3 end-to-end fine-tuning with progressive encoder unfreezing.

    Args:
        config_path: Path to e2e_finetune.yaml config file.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    output_cfg = cfg["output"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage3] Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("[stage3] Scanning raw patches...")
    sequences = build_raw_sequences(
        patch_dir=data_cfg["patch_dir"],
        min_length=data_cfg.get("min_sequence_length", 6),
    )

    if not sequences:
        print("[stage3] No sequences found. Ensure patch_dir contains .npy files.")
        return

    # Spatial 80/20 split by grid location (same convention as Stage 2)
    grid_keys = sorted(set(seq["grid_key"] for seq in sequences))
    n_train = int(0.8 * len(grid_keys))
    train_keys = set(grid_keys[:n_train])
    val_keys = set(grid_keys[n_train:])

    train_seqs = [s for s in sequences if s["grid_key"] in train_keys]
    val_seqs = [s for s in sequences if s["grid_key"] in val_keys]

    window_size = data_cfg["window_size"]
    train_dataset = RawPatchTemporalDataset(train_seqs, window_size=window_size)
    val_dataset = RawPatchTemporalDataset(val_seqs, window_size=window_size)

    print(
        f"[stage3] Spatial split: {len(train_keys)} train locs "
        f"({len(train_dataset)} windows), "
        f"{len(val_keys)} val locs ({len(val_dataset)} windows)"
    )

    if len(train_dataset) == 0:
        print("[stage3] No valid training windows. Sequences may be too short.")
        return

    num_workers = data_cfg.get("num_workers", 4)
    batch_size = data_cfg.get("batch_size", 8)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    print("[stage3] Loading encoder from Stage 1 checkpoint...")
    encoder = SARJEPAEncoder(
        pretrained=True,
        checkpoint_path=model_cfg["encoder_checkpoint"],
        embed_dim=model_cfg.get("embed_dim", 768),
        freeze=False,   # we manage freezing manually below
        in_chans=model_cfg.get("in_chans", 1),
    ).to(device)

    print("[stage3] Loading temporal predictor from Stage 2 checkpoint...")
    predictor = TemporalPredictor(
        embed_dim=model_cfg.get("embed_dim", 768),
        num_layers=model_cfg.get("num_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        ffn_dim=model_cfg.get("ffn_dim", 2048),
        dropout=model_cfg.get("dropout", 0.1),
        time_encoding_type=model_cfg.get("time_encoding_type", "sinusoidal"),
    ).to(device)

    # Load Stage 2 predictor weights
    stage2_ckpt = torch.load(
        model_cfg["predictor_checkpoint"], map_location="cpu", weights_only=False
    )
    pred_state = (
        stage2_ckpt["model_state_dict"]
        if "model_state_dict" in stage2_ckpt
        else stage2_ckpt
    )
    predictor.load_state_dict(pred_state)
    print("[stage3] Predictor weights loaded.")

    # Loss: Smooth L1 (more robust than MSE, less sensitive to outliers)
    criterion = nn.SmoothL1Loss()

    # Output directory
    ckpt_dir = Path(output_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = -1
    grad_clip = train_cfg.get("grad_clip", 1.0)
    weight_decay = train_cfg.get("weight_decay", 0.01)

    # ------------------------------------------------------------------
    # Phase A: unfreeze last 4 encoder blocks + norm
    # ------------------------------------------------------------------
    epochs_a = train_cfg.get("epochs_phase_a", 30)
    encoder_lr_a = train_cfg.get("encoder_lr", 1e-5)
    predictor_lr_a = train_cfg.get("predictor_lr", 1e-4)

    unfreeze_phase_a(encoder)
    print(
        f"[stage3] Phase A: last 4 blocks + norm unfrozen. "
        f"Encoder trainable params: {count_trainable(encoder):,}"
    )

    optimizer_a = torch.optim.AdamW(
        [
            {"params": [p for p in encoder.parameters() if p.requires_grad],
             "lr": encoder_lr_a},
            {"params": predictor.parameters(),
             "lr": predictor_lr_a},
        ],
        weight_decay=weight_decay,
    )

    total_steps_a = epochs_a * len(train_loader)
    warmup_a = train_cfg.get("warmup_steps", 200)
    scheduler_a = torch.optim.lr_scheduler.LambdaLR(
        optimizer_a,
        lr_lambda=lambda step: _cosine_lr_lambda(step, warmup_a, total_steps_a),
    )

    print(f"\n[stage3] ===== Phase A: {epochs_a} epochs =====")
    for epoch in range(1, epochs_a + 1):
        train_loss = run_epoch(
            encoder, predictor, train_loader,
            optimizer_a, scheduler_a, criterion, device,
            train=True, grad_clip=grad_clip,
        )
        val_loss = run_epoch(
            encoder, predictor, val_loader,
            None, None, criterion, device,
            train=False,
        )

        enc_lr = scheduler_a.get_last_lr()[0]
        pred_lr = scheduler_a.get_last_lr()[1] if len(scheduler_a.get_last_lr()) > 1 else predictor_lr_a
        print(
            f"[stage3][A] Epoch {epoch:03d}/{epochs_a} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"enc_lr={enc_lr:.2e} | pred_lr={pred_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            _save_checkpoint(encoder, predictor, epoch, val_loss,
                             ckpt_dir / "best_e2e.pt", phase="A")
            print(f"[stage3] New best val_loss={val_loss:.6f} -> best_e2e.pt")

    # ------------------------------------------------------------------
    # Phase B: unfreeze ALL encoder layers
    # ------------------------------------------------------------------
    epochs_b = train_cfg.get("epochs_phase_b", 20)
    encoder_lr_b = train_cfg.get("encoder_lr_phase_b", 5e-6)
    predictor_lr_b = train_cfg.get("predictor_lr_phase_b", 5e-5)

    unfreeze_phase_b(encoder)
    print(
        f"\n[stage3] Phase B: all encoder layers unfrozen. "
        f"Encoder trainable params: {count_trainable(encoder):,}"
    )

    optimizer_b = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": encoder_lr_b},
            {"params": predictor.parameters(), "lr": predictor_lr_b},
        ],
        weight_decay=weight_decay,
    )

    total_steps_b = epochs_b * len(train_loader)
    warmup_b = train_cfg.get("warmup_steps_phase_b", 100)
    scheduler_b = torch.optim.lr_scheduler.LambdaLR(
        optimizer_b,
        lr_lambda=lambda step: _cosine_lr_lambda(step, warmup_b, total_steps_b),
    )

    print(f"\n[stage3] ===== Phase B: {epochs_b} epochs =====")
    for epoch in range(1, epochs_b + 1):
        train_loss = run_epoch(
            encoder, predictor, train_loader,
            optimizer_b, scheduler_b, criterion, device,
            train=True, grad_clip=grad_clip,
        )
        val_loss = run_epoch(
            encoder, predictor, val_loader,
            None, None, criterion, device,
            train=False,
        )

        enc_lr = scheduler_b.get_last_lr()[0]
        pred_lr = scheduler_b.get_last_lr()[1] if len(scheduler_b.get_last_lr()) > 1 else predictor_lr_b
        global_epoch = epochs_a + epoch
        print(
            f"[stage3][B] Epoch {epoch:03d}/{epochs_b} (global {global_epoch}) | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"enc_lr={enc_lr:.2e} | pred_lr={pred_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = global_epoch
            _save_checkpoint(encoder, predictor, global_epoch, val_loss,
                             ckpt_dir / "best_e2e.pt", phase="B")
            print(f"[stage3] New best val_loss={val_loss:.6f} -> best_e2e.pt")

    print(
        f"\n[stage3] Training complete. "
        f"Best val_loss={best_val_loss:.6f} at epoch {best_epoch}"
    )

    # Save final checkpoint
    _save_checkpoint(
        encoder, predictor,
        epoch=epochs_a + epochs_b,
        val_loss=best_val_loss,
        path=ckpt_dir / "stage3_final.pt",
        phase="final",
    )
    print(f"[stage3] Final checkpoint saved -> {ckpt_dir / 'stage3_final.pt'}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    encoder: SARJEPAEncoder,
    predictor: TemporalPredictor,
    epoch: int,
    val_loss: float,
    path: Path,
    phase: str = "",
) -> None:
    """Save encoder + predictor state dicts to a single checkpoint file."""
    torch.save(
        {
            "epoch": epoch,
            "phase": phase,
            "encoder_state_dict": encoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 3: End-to-end fine-tuning with progressive encoder unfreezing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/e2e_finetune.yaml",
        help="Path to e2e_finetune.yaml config file",
    )
    args = parser.parse_args()
    train_stage3(args.config)
