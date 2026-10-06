"""
evaluation.py
Evaluation metrics for extreme-imbalance edge classification (Chapter 4,
Section E), implementing the supervisor's points 7 and 8.

Primary metrics (as promised in Chapters 1-3):
-AUPRC -- threshold-independent, appropriate under extreme skew. Reported
      WITH the positive prevalence, since a random classifier scores
      AUPRC == prevalence.
-Recall.

Secondary / operational metrics (supervisor point 8):
    - Precision, F1 (F1 for comparability with the Multi-GNN / OES literature).
    - Recall at a FIXED precision level.
    - Precision@k and Recall@k for a top-k alert budget.
    - Confusion matrix and false-positive rate.

Threshold policy (leakage-safe): the operating threshold is chosen on the
VALIDATION scores to hit a target precision, then applied unchanged to test.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    precision_score, recall_score, f1_score,
)


def select_threshold_for_precision(y_val, scores_val, target_precision=0.5):
    """Lowest threshold on validation achieving target precision (max recall s.t. precision)."""
    precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds
    ok = precision >= target_precision
    if ok.any():
        best = int(np.argmax(np.where(ok, recall, -1)))
        return float(thresholds[best])
    return float(thresholds[int(np.argmax(precision))])


def topk_metrics(y_true, scores, k):
    """Precision@k and Recall@k for the k highest-scoring edges (alert budget)."""
    k = min(k, len(scores))
    order = np.argsort(scores)[::-1][:k]
    hits = y_true[order].sum()
    total_pos = y_true.sum()
    return {f"precision@{k}": float(hits / max(k, 1)),
            f"recall@{k}": float(hits / max(total_pos, 1))}


def evaluate_scores(y_true, scores, threshold, topk=(100, 500)):
    """Full metric suite given probability scores and a validation-chosen threshold."""
    y_pred = (scores >= threshold).astype(int)
    prevalence = float(y_true.mean())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auprc = float(average_precision_score(y_true, scores))
    metrics = {
        "auprc": auprc,
        "prevalence": prevalence,
        "auprc_lift": float(auprc / max(prevalence, 1e-12)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        # Ch3 3.10.1 asks for "recall at fixed precision" explicitly. Because the
        # threshold was chosen on VALIDATION to hit a target precision, the recall
        # below IS recall-at-fixed-precision; the precision actually achieved is also
        # achieved on test, so any shortfall against the target is visible rather
        # than hidden (the validation-chosen threshold need not transfer exactly).
        "recall_at_target_precision": float(recall_score(y_true, y_pred, zero_division=0)),
        "achieved_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": float(threshold),
    }
    for k in topk:
        metrics.update(topk_metrics(y_true, scores, k))
    return metrics


def aggregate_seeds(per_seed):
    """Mean +/- std across seeds for every numeric metric (supervisor point 7)."""
    if not per_seed:
        return {}
    keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float))]
    agg = {}
    for k in keys:
        vals = np.array([d[k] for d in per_seed], dtype=float)
        agg[f"{k}_mean"] = float(vals.mean())
        agg[f"{k}_std"] = float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
    agg["n_seeds"] = len(per_seed)
    return agg
