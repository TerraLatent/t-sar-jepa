"""Time encoding modules for irregular temporal sequences."""
import math
import torch
import torch.nn as nn


class SinusoidalTimeEncoding(nn.Module):
    """Sinusoidal time encoding with learned projection."""
    def __init__(self, embed_dim: int = 768, max_period: int = 10000):
        super().__init__()
        assert embed_dim % 2 == 0
        half = embed_dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B, seq_len) normalized [0,1]. Returns: (B, seq_len, embed_dim)."""
        t_scaled = t.unsqueeze(-1) * 1000.0
        args = t_scaled * self.freqs
        enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.proj(enc)


class CTLPETimeEncoding(nn.Module):
    """Continuous-Time Learnable Positional Encoding (arXiv 2409.20092).
    Maps scalar time through learnable MLP."""
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B, seq_len) normalized [0,1]. Returns: (B, seq_len, embed_dim)."""
        return self.net(t.unsqueeze(-1))


class LinearLearnableTimeEncoding(nn.Module):
    """Simple linear mapping from time to embedding space."""
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(1, embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.proj(t.unsqueeze(-1))


def build_time_encoding(encoding_type: str, embed_dim: int = 768) -> nn.Module:
    """Factory: 'sinusoidal', 'ctlpe', or 'linear'."""
    if encoding_type == "sinusoidal":
        return SinusoidalTimeEncoding(embed_dim=embed_dim)
    elif encoding_type == "ctlpe":
        return CTLPETimeEncoding(embed_dim=embed_dim)
    elif encoding_type == "linear":
        return LinearLearnableTimeEncoding(embed_dim=embed_dim)
    else:
        raise ValueError(f"Unknown time encoding: {encoding_type}")
