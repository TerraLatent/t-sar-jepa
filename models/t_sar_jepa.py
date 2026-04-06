"""T-SAR-JEPA: Full pipeline wrapper.

Combines SARJEPAEncoder and TemporalPredictor into a single inference
pipeline that takes raw SAR patches and produces anomaly scores.
"""

from typing import Dict, Optional

import numpy as np
import torch

from models.encoder import SARJEPAEncoder
from models.temporal_predictor import TemporalPredictor


class TSARJEPAPipeline:
    """End-to-end T-SAR-JEPA inference pipeline.

    Wraps the frozen SAR-JEPA encoder and trained temporal predictor
    to run sliding-window anomaly detection on a sequence of SAR patches.

    Args:
        encoder_checkpoint: Path to encoder weights. None for random init.
        predictor_checkpoint: Path to predictor weights. None for random init.
        pretrained: If True, load encoder from SAR-JEPA checkpoint.
        embed_dim: Embedding dimension. Default 768.
        num_layers: Transformer layers in predictor. Default 4.
        num_heads: Attention heads in predictor. Default 8.
    """

    def __init__(
        self,
        encoder_checkpoint: Optional[str] = None,
        predictor_checkpoint: Optional[str] = None,
        pretrained: bool = True,
        embed_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        self.embed_dim = embed_dim

        # Build encoder
        self.encoder = SARJEPAEncoder(
            pretrained=pretrained and encoder_checkpoint is not None,
            checkpoint_path=encoder_checkpoint,
            embed_dim=embed_dim,
            freeze=True,
        )

        # Build predictor
        self.predictor = TemporalPredictor(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        # Load predictor checkpoint if provided
        if predictor_checkpoint is not None:
            checkpoint = torch.load(predictor_checkpoint, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
            self.predictor.load_state_dict(state_dict)

    def run(
        self,
        patches: np.ndarray,
        days: np.ndarray,
        window_size: int = 16,
        device: str = "cuda",
    ) -> Dict[str, object]:
        """Run full inference: encode -> predict -> score.

        Args:
            patches: Raw SAR patches of shape (seq_len, 1, 224, 224).
            days: Day values of shape (seq_len,).
            window_size: Context window for temporal prediction.
            device: Torch device string.

        Returns:
            Dict with keys:
                anomaly_scores: (N,) tensor of L2 prediction errors
                predictions: (N, 768) predicted embeddings
                actuals: (N, 768) actual embeddings
                attention_weights: (N, heads, W, W) last-layer attention
                timestep_indices: (N,) array of scored timestep indices
        """
        from inference.anomaly_scorer import run_inference_on_sequence

        patches_tensor = torch.from_numpy(patches.astype(np.float32))

        results = run_inference_on_sequence(
            encoder=self.encoder,
            predictor=self.predictor,
            patches=patches_tensor,
            days=days,
            window_size=window_size,
            device=device,
        )

        return results
