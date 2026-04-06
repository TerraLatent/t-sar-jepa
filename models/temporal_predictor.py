"""Temporal transformer predictor for latent SAR embedding sequences.

Takes a sequence of encoder-produced 768-dim vectors (from consecutive
timesteps of the same location) and predicts the next vector in the sequence.
Anomalies are detected via high prediction error.
"""

import copy

import torch
import torch.nn as nn

from models.time_encodings import build_time_encoding


class TemporalPredictor(nn.Module):
    """Transformer-based temporal predictor for SAR latent sequences.

    Takes a sequence of encoder embeddings with associated time encodings
    and predicts the next latent state via mean-pooled transformer output.

    Args:
        embed_dim: Embedding dimension. Default 768.
        num_layers: Number of transformer encoder layers. Default 4.
        num_heads: Number of attention heads. Default 8.
        ffn_dim: Feed-forward network hidden dimension. Default 2048.
        dropout: Dropout rate. Default 0.1.
        time_encoding_type: Time encoding variant. One of 'sinusoidal', 'ctlpe', 'linear'. Default 'sinusoidal'.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        time_encoding_type: str = "sinusoidal",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.time_encoding_type = time_encoding_type

        # Time encoding
        self.time_enc = build_time_encoding(time_encoding_type, embed_dim=embed_dim)

        # Transformer encoder layers (pre-norm, batch_first)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.layers = nn.ModuleList(
            [self._clone_layer(encoder_layer) for _ in range(num_layers)]
        )

        # Output head: LayerNorm + Linear projection
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    @staticmethod
    def _clone_layer(layer: nn.TransformerEncoderLayer) -> nn.TransformerEncoderLayer:
        """Create a fresh copy of a TransformerEncoderLayer with the same config."""
        return copy.deepcopy(layer)

    def forward(
        self,
        context: torch.Tensor,
        time_enc: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Predict the next latent state from a temporal sequence.

        Args:
            context: Encoder embeddings of shape (B, seq_len, 768).
            time_enc: Normalized day values of shape (B, seq_len) in [0, 1].
            return_attention: If True, also return last-layer attention weights.

        Returns:
            Predicted next latent state of shape (B, 768).
            If return_attention: tuple of (prediction, attention_weights)
                where attention_weights has shape (B, num_heads, seq_len, seq_len).
        """
        # Add time encoding to input embeddings
        x = context + self.time_enc(time_enc)

        if not return_attention:
            # Standard forward through all layers
            for layer in self.layers:
                x = layer(x)

            # Mean pool over sequence dimension
            pooled = x.mean(dim=1)  # (B, embed_dim)

            # Project through output head
            return self.out_proj(self.out_norm(pooled))

        # Forward with attention extraction from last layer
        for layer in self.layers[:-1]:
            x = layer(x)

        # Manual forward through last layer to capture attention weights
        last_layer = self.layers[-1]

        # Pre-norm self-attention
        x_normed = last_layer.norm1(x)

        # Compute multi-head attention with weights
        attn_output, attn_weights = last_layer.self_attn(
            x_normed, x_normed, x_normed,
            need_weights=True,
            average_attn_weights=False,  # keep per-head weights
        )
        x = x + last_layer.dropout1(attn_output)

        # Pre-norm feed-forward
        x_normed = last_layer.norm2(x)
        ff_output = last_layer.linear2(
            last_layer.dropout(last_layer.activation(last_layer.linear1(x_normed)))
        )
        x = x + last_layer.dropout2(ff_output)

        # Mean pool and project
        pooled = x.mean(dim=1)
        prediction = self.out_proj(self.out_norm(pooled))

        return prediction, attn_weights
