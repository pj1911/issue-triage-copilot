"""
Shared multi-label evaluation utilities.

Both train_encoder.py and evaluate_encoder.py import from here so
the metric definitions never drift between training-time checks and
final reporting.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, hamming_loss


def eval_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> dict:
    """
    Compute standard multi-label metrics.

    Args:
        y_true:  binary matrix (n_samples, n_classes)
        y_pred:  binary matrix (n_samples, n_classes)
        classes: ordered list of label names matching matrix columns

    Returns:
        dict with micro_f1, macro_f1, exact_match_ratio, hamming_loss,
        per_label_f1 (sorted descending by f1)
    """
    micro_f1    = f1_score(y_true, y_pred, average="micro",    zero_division=0)
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    exact_match = accuracy_score(y_true, y_pred)
    hl          = hamming_loss(y_true, y_pred)

    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_label = dict(
        sorted(
            {cls: round(float(f1), 4) for cls, f1 in zip(classes, per_label_f1)}.items(),
            key=lambda x: -x[1],
        )
    )

    return {
        "micro_f1":          round(float(micro_f1),    4),
        "macro_f1":          round(float(macro_f1),    4),
        "exact_match_ratio": round(float(exact_match), 4),
        "hamming_loss":      round(float(hl),          4),
        "per_label_f1":      per_label,
    }


THRESHOLD_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def sweep_thresholds(
    probs: np.ndarray,
    y_true: np.ndarray,
    grid: list[float] | None = None,
) -> list[dict]:
    """
    Sweep thresholds and record micro-F1, macro-F1, precision, recall,
    predicted-positive rate, and mean labels per issue.

    IMPORTANT: call this on val only; never on test.

    Returns a list of dicts, one per threshold, sorted by threshold.
    """
    if grid is None:
        grid = THRESHOLD_GRID

    rows = []
    n = len(y_true)
    for t in grid:
        y_pred     = (probs >= t).astype(int)
        micro_f1   = f1_score(y_true, y_pred, average="micro",      zero_division=0)
        macro_f1   = f1_score(y_true, y_pred, average="macro",      zero_division=0)
        precision  = f1_score(y_true, y_pred, average="micro",      zero_division=0)  # recomputed below
        from sklearn.metrics import precision_score, recall_score
        precision  = precision_score(y_true, y_pred, average="micro", zero_division=0)
        recall     = recall_score(y_true, y_pred,    average="micro", zero_division=0)
        pred_pos_rate  = float((y_pred.sum(axis=1) > 0).mean())   # fraction with ≥1 label
        mean_labels    = float(y_pred.sum(axis=1).mean())

        rows.append({
            "threshold":       round(t, 2),
            "micro_f1":        round(float(micro_f1),      4),
            "macro_f1":        round(float(macro_f1),      4),
            "precision":       round(float(precision),     4),
            "recall":          round(float(recall),        4),
            "pred_pos_rate":   round(pred_pos_rate,        4),
            "mean_labels":     round(mean_labels,          4),
        })
    return rows


def select_threshold(
    sweep_rows: list[dict],
    max_pred_pos_rate: float = 0.80,
) -> tuple[float, float]:
    """
    Choose the threshold with best micro-F1 subject to:
      - predicted positive rate <= max_pred_pos_rate
        (avoids thresholds that predict labels for almost every issue)

    Falls back to best raw micro-F1 if all thresholds exceed the cap.

    Returns (best_threshold, best_micro_f1).
    """
    candidates = [r for r in sweep_rows if r["pred_pos_rate"] <= max_pred_pos_rate]
    if not candidates:
        candidates = sweep_rows   # fallback: no constraint

    best = max(candidates, key=lambda r: r["micro_f1"])
    return best["threshold"], best["micro_f1"]


# Keep a simple alias used by train_encoder for per-epoch checkpoint selection
def tune_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    grid: list[float] | None = None,
) -> tuple[float, float]:
    """Thin wrapper around sweep+select for use during training."""
    rows = sweep_thresholds(probs, y_true, grid)
    return select_threshold(rows)
