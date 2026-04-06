"""LSTM temporal baseline on latent embeddings."""
import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: (B, seq_len, 768). Returns: (B, 768)."""
        out, _ = self.lstm(context)
        return self.proj(out[:, -1, :])

def train_lstm(sequences: list, context_k: int = 5, epochs: int = 100, lr: float = 1e-3):
    """Train LSTM and return (model, scores)."""
    model = LSTMPredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
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
        if (epoch + 1) % 25 == 0:
            print(f"  LSTM epoch {epoch+1}: loss={loss.item():.6f}")
    with torch.no_grad():
        scores = torch.norm(model(contexts) - targets, dim=1).numpy()
    return model, scores
