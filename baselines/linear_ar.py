"""Linear autoregressive baseline on latent embeddings."""
import numpy as np
import torch
import torch.nn as nn

class LinearAR(nn.Module):
    def __init__(self, embed_dim: int = 768, context_k: int = 5):
        super().__init__()
        self.proj = nn.Linear(embed_dim * context_k, embed_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: (B, K, 768). Returns: (B, 768)."""
        return self.proj(context.reshape(context.shape[0], -1))

def train_linear_ar(sequences: list, context_k: int = 5, epochs: int = 50, lr: float = 1e-3):
    """Train and return (model, scores)."""
    model = LinearAR(embed_dim=768, context_k=context_k)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    contexts, targets = [], []
    for seq in sequences:
        emb = torch.from_numpy(seq["embeddings"])
        for i in range(len(emb) - context_k):
            contexts.append(emb[i:i+context_k])
            targets.append(emb[i+context_k])
    contexts = torch.stack(contexts)
    targets = torch.stack(targets)
    for epoch in range(epochs):
        pred = model(contexts)
        loss = criterion(pred, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = torch.norm(model(contexts) - targets, dim=1).numpy()
    return model, scores
