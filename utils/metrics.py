from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import numpy as np


def compute_metrics(labels, preds, threshold=0.5):
    """
    Computes standard classification metrics.
    Args:
        labels: Ground truth binary labels (list or array)
        preds: Predicted probabilities (list or array)
        threshold: Decision threshold for binary classification
    """
    labels = np.array(labels).reshape(-1)
    preds = np.array(preds).reshape(-1)

    # Binarize predictions based on threshold
    binary_preds = (preds > threshold).astype(int)

    acc = accuracy_score(labels, binary_preds)
    prec = precision_score(labels, binary_preds, zero_division=0)
    rec = recall_score(labels, binary_preds, zero_division=0)
    f1 = f1_score(labels, binary_preds, zero_division=0)

    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.5  # Handle case with only one class in batch

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}


def save_metrics(metrics, path):
    import json

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
