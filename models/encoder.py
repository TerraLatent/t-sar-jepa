"""SAR-JEPA encoder wrapper for feature extraction from SAR patches.

Wraps the MaskedAutoencoderViT encoder (patch_embed -> blocks -> norm)
and mean-pools spatial tokens to produce a single (B, embed_dim) vector.
"""

from typing import Optional

import torch
import torch.nn as nn

def _build_mae_model(in_chans: int = 1) -> nn.Module:
    """Construct a MaskedAutoencoderViT (base, patch16, 224)."""
    from sarjepa.models_lomar import mae_vit_base_patch16
    model = mae_vit_base_patch16(in_chans=in_chans)
    return model


class SARJEPAEncoder(nn.Module):
    """Feature extractor built on the SAR-JEPA ViT encoder.

    Forward pass: patch_embed -> cls_token prepend -> transformer blocks -> layer norm
                  -> mean pool spatial tokens -> (B, 768)

    NOTE: The original SAR-JEPA forward_encoder does NOT add pos_embed because
    it uses local window masking (which reorders patches). The Attention layers
    use iRPE (relative position encoding on keys) instead.  We match that
    behaviour here: pos_embed is NOT added during the forward pass. If you want
    to experiment with adding it (e.g., after fine-tuning), set
    ``use_pos_embed=True``.

    Args:
        pretrained: If True, load weights from ``checkpoint_path``.
        checkpoint_path: Path to a SAR-JEPA checkpoint file.
        embed_dim: Embedding dimension (must match the model variant). Default 768.
        use_pos_embed: Whether to add the absolute sin-cos pos_embed. Default False.
        freeze: If True, set all parameters to requires_grad=False.
        in_chans: Number of input channels. Default 1. Single-channel amplitude input.
    """

    def __init__(
        self,
        pretrained: bool = False,
        checkpoint_path: Optional[str] = None,
        embed_dim: int = 768,
        use_pos_embed: bool = False,
        freeze: bool = True,
        in_chans: int = 1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_pos_embed = use_pos_embed

        # Build the full MAE model, then keep only encoder parts.
        mae = _build_mae_model(in_chans=in_chans)

        # Extract encoder components.
        self.patch_embed = mae.patch_embed
        self.cls_token = mae.cls_token
        self.pos_embed = mae.pos_embed      # kept even if unused (for weight loading)
        self.blocks = mae.blocks
        self.norm = mae.norm

        if pretrained:
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required when pretrained=True")
            self._load_pretrained(checkpoint_path)

        if freeze:
            self._freeze()

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_pretrained(self, checkpoint_path: str) -> None:
        """Load encoder weights from a SAR-JEPA checkpoint.

        Expects checkpoint format: ``checkpoint['model']`` containing the full
        MaskedAutoencoderViT state_dict.  Only encoder keys are loaded
        (patch_embed, cls_token, pos_embed, blocks, norm).
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        # Strip 'module.' prefix from DDP-trained checkpoints.
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        # Filter to encoder-only keys.
        encoder_prefixes = ("patch_embed.", "cls_token", "pos_embed", "blocks.", "norm.")
        encoder_state = {
            k: v for k, v in state_dict.items()
            if any(k.startswith(p) for p in encoder_prefixes)
        }

        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        if unexpected:
            print(f"[SARJEPAEncoder] Unexpected keys ignored: {unexpected}")
        if missing:
            print(f"[SARJEPAEncoder] Missing keys (may be expected): {missing}")

    def _freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from SAR patches.

        Args:
            x: Input tensor of shape ``(B, in_chans, H, W)``.

        Returns:
            Feature tensor of shape ``(B, embed_dim)``, mean-pooled over
            spatial tokens.
        """
        # Patch embed: (B, in_chans, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)

        # Prepend CLS token: (B, 1+num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Optionally add absolute position embedding.
        if self.use_pos_embed:
            x = x + self.pos_embed

        # Transformer blocks.
        for blk in self.blocks:
            x = blk(x)

        # Layer norm.
        x = self.norm(x)

        # Mean pool over spatial tokens (exclude CLS at index 0).
        x = x[:, 1:, :].mean(dim=1)  # (B, embed_dim)

        return x
