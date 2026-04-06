"""ROC/PR curves using InSAR coherence drops as pseudo ground truth."""
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def coherence_to_labels(coherence_values: np.ndarray, drop_threshold: float = 0.3) -> np.ndarray:
    """Low coherence = change = positive. Returns binary array."""
    return (coherence_values < drop_threshold).astype(int)

def compute_roc_pr(anomaly_scores: np.ndarray, labels: np.ndarray) -> dict:
    """Compute ROC and PR curves + AUC. Returns dict with fpr, tpr, roc_auc, precision, recall, pr_auc."""
    fpr, tpr, _ = roc_curve(labels, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(labels, anomaly_scores)
    pr_auc = auc(recall, precision)
    return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc, "precision": precision, "recall": recall, "pr_auc": pr_auc}
