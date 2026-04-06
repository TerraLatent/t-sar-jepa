"""Stage 3: Anomaly scoring from temporal predictions."""

from typing import Dict, Optional

import numpy as np
import torch


def compute_anomaly_scores(predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Compute L2 distance between predicted and actual latent vectors.

    Args:
        predicted: Predicted embeddings of shape (N, 768).
        actual: Actual embeddings of shape (N, 768).

    Returns:
        Anomaly scores of shape (N,), where higher = more anomalous.
    """
    return torch.norm(predicted - actual, dim=1, p=2)


def find_anomalies(scores: np.ndarray, top_k_percent: float = 10) -> np.ndarray:
    """Return indices of top-k% highest anomaly scores.

    Args:
        scores: 1D array of anomaly scores.
        top_k_percent: Percentage of top scores to flag as anomalies.

    Returns:
        Array of indices corresponding to the most anomalous timesteps.
    """
    k = max(1, int(len(scores) * top_k_percent / 100))
    # argsort descending, take top k
    top_indices = np.argsort(scores)[::-1][:k]
    return top_indices


def run_inference_on_sequence(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    patches: torch.Tensor,
    days: np.ndarray,
    window_size: int = 16,
    device: str = "cuda",
) -> Dict[str, object]:
    """Run full inference pipeline on a raw patch sequence.

    Takes raw patches, encodes them with the frozen encoder, then runs
    sliding-window temporal prediction with teacher forcing to produce
    anomaly scores.

    Args:
        encoder: Frozen SARJEPAEncoder.
        predictor: Trained TemporalPredictor.
        patches: Raw patches of shape (seq_len, 1, 224, 224).
        days: Array of day values (seq_len,).
        window_size: Context window size for the predictor.
        device: Torch device.

    Returns:
        Dict with:
            anomaly_scores: (N,) tensor of L2 prediction errors
            predictions: (N, 768) tensor of predicted embeddings
            actuals: (N, 768) tensor of actual embeddings
            attention_weights: (N, num_heads, window_size, window_size) attention from last layer
            timestep_indices: (N,) array of which timesteps were scored
    """
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    seq_len = patches.shape[0]

    # Encode all patches with frozen encoder
    with torch.no_grad():
        # Process in batches to avoid OOM
        batch_size = 32
        embeddings_list = []
        for i in range(0, seq_len, batch_size):
            batch = patches[i : i + batch_size].to(device)
            emb = encoder(batch)  # (B, 768)
            embeddings_list.append(emb.cpu())
        embeddings = torch.cat(embeddings_list, dim=0)  # (seq_len, 768)

    # Sliding window prediction with teacher forcing
    predictions = []
    actuals = []
    attention_weights_list = []
    timestep_indices = []

    days_tensor = torch.from_numpy(days.astype(np.float32))

    with torch.no_grad():
        for i in range(seq_len - window_size):
            # Context window
            ctx = embeddings[i : i + window_size].unsqueeze(0).to(device)  # (1, W, 768)
            target = embeddings[i + window_size]  # (768,)

            # Normalize days within window to [0, 1]
            window_days = days_tensor[i : i + window_size]
            d_min, d_max = window_days[0], window_days[-1]
            if d_max > d_min:
                time_enc = ((window_days - d_min) / (d_max - d_min)).unsqueeze(0).to(device)
            else:
                time_enc = torch.zeros(1, window_size).to(device)

            # Predict with attention
            pred, attn = predictor(ctx, time_enc, return_attention=True)

            predictions.append(pred.squeeze(0).cpu())
            actuals.append(target)
            attention_weights_list.append(attn.squeeze(0).cpu())
            timestep_indices.append(i + window_size)

    if not predictions:
        return {
            "anomaly_scores": torch.tensor([]),
            "predictions": torch.tensor([]),
            "actuals": torch.tensor([]),
            "attention_weights": torch.tensor([]),
            "timestep_indices": np.array([]),
        }

    predictions_t = torch.stack(predictions)  # (N, 768)
    actuals_t = torch.stack(actuals)  # (N, 768)
    attention_weights_t = torch.stack(attention_weights_list)  # (N, heads, W, W)

    anomaly_scores = compute_anomaly_scores(predictions_t, actuals_t)

    return {
        "anomaly_scores": anomaly_scores,
        "predictions": predictions_t,
        "actuals": actuals_t,
        "attention_weights": attention_weights_t,
        "timestep_indices": np.array(timestep_indices),
    }
