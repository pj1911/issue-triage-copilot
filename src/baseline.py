"""
Dumb majority-class baseline for priority (single-label)
and top-k label baseline for issue labels (multi-label).

Run from project root:
    python -m src.baseline

Saves results to reports/baseline_metrics.json.
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)

from src.utils.paths import PROCESSED_DIR, REPORTS_DIR

# ── Load splits ────────────────────────────────────────────────────────────────

train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
val   = pd.read_parquet(PROCESSED_DIR / "val.parquet")
test  = pd.read_parquet(PROCESSED_DIR / "test.parquet")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PRIORITY  – always predict the most common training class
# ══════════════════════════════════════════════════════════════════════════════

majority_priority = train["priority"].value_counts().idxmax()
print(f"Majority priority class (train): '{majority_priority}'")
print(f"Train class dist:\n{train['priority'].value_counts(normalize=True).round(3)}\n")


def eval_priority(df: pd.DataFrame, split: str) -> dict:
    y_true = df["priority"].values
    y_pred = np.full(len(y_true), majority_priority)

    acc = accuracy_score(y_true, y_pred)
    labels = sorted(train["priority"].unique())
    macro_f1    = f1_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    per_class = classification_report(
        y_true, y_pred, labels=labels,
        output_dict=True, zero_division=0,
    )
    # drop "accuracy" key sklearn adds; keep per-class + avg rows
    per_class.pop("accuracy", None)

    result = {
        "accuracy":    round(acc, 4),
        "macro_f1":    round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class":   {
            k: {m: round(v, 4) for m, v in metrics.items()}
            for k, metrics in per_class.items()
            if isinstance(metrics, dict)
        },
    }
    print(f"── priority ({split}) ──")
    print(f"  accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}")
    return result


priority_results = {
    "majority_class": majority_priority,
    "strategy": "always predict most common training class",
    "val":  eval_priority(val,  "val"),
    "test": eval_priority(test, "test"),
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. LABELS  – multi-label baselines
# ══════════════════════════════════════════════════════════════════════════════

# Build the label vocabulary from training set
all_train_labels = [lbl for row in train["labels"] for lbl in row]
label_counts = Counter(all_train_labels)
vocab = [lbl for lbl, _ in label_counts.most_common()]  # sorted by frequency


def binarize(label_lists, vocab):
    """Convert list-of-label-lists to binary matrix (rows x vocab)."""
    idx = {lbl: i for i, lbl in enumerate(vocab)}
    mat = np.zeros((len(label_lists), len(vocab)), dtype=np.int8)
    for i, lbls in enumerate(label_lists):
        for lbl in lbls:
            if lbl in idx:
                mat[i, idx[lbl]] = 1
    return mat


def eval_labels(df: pd.DataFrame, y_pred_mat: np.ndarray, split: str, strategy: str) -> dict:
    y_true = binarize(df["labels"].tolist(), vocab)

    # Hamming loss counts each label independently
    hl  = hamming_loss(y_true, y_pred_mat)
    # Micro/macro F1 across all labels
    micro_f1 = f1_score(y_true, y_pred_mat, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred_mat, average="macro", zero_division=0)
    # Subset accuracy = exact match
    subset_acc = accuracy_score(y_true, y_pred_mat)

    result = {
        "subset_accuracy": round(float(subset_acc), 4),
        "hamming_loss":    round(float(hl), 4),
        "micro_f1":        round(float(micro_f1), 4),
        "macro_f1":        round(float(macro_f1), 4),
    }
    print(f"── labels/{strategy} ({split}) ──")
    print(f"  subset_acc={subset_acc:.4f}  hamming_loss={hl:.4f}  "
          f"micro_f1={micro_f1:.4f}  macro_f1={macro_f1:.4f}")
    return result


# ── 2a. Predict nothing (empty set) ──────────────────────────────────────────

def empty_pred(df):
    return np.zeros((len(df), len(vocab)), dtype=np.int8)

print()
label_results_empty = {
    "strategy": "predict empty set for all issues",
    "val":  eval_labels(val,  empty_pred(val),  "val",  "empty"),
    "test": eval_labels(test, empty_pred(test), "test", "empty"),
}

# ── 2b. Predict top-k most common labels for every issue ─────────────────────

TOP_K_VALUES = [1, 3, 5]
label_results_topk = {}

for k in TOP_K_VALUES:
    top_k_labels = set(vocab[:k])
    top_k_idx    = [vocab.index(lbl) for lbl in vocab[:k]]

    def topk_pred(df, idx=top_k_idx):
        mat = np.zeros((len(df), len(vocab)), dtype=np.int8)
        mat[:, idx] = 1
        return mat

    print()
    label_results_topk[f"top_{k}"] = {
        "strategy": f"always predict top-{k} most common labels: {vocab[:k]}",
        "val":  eval_labels(val,  topk_pred(val),  "val",  f"top_{k}"),
        "test": eval_labels(test, topk_pred(test), "test", f"top_{k}"),
    }

# ── 2c. Label frequency info ─────────────────────────────────────────────────

top20_labels = [{"label": lbl, "train_count": cnt}
                for lbl, cnt in label_counts.most_common(20)]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Save
# ══════════════════════════════════════════════════════════════════════════════

report = {
    "description": (
        "Dumb majority-class baseline. "
        "Priority: always predict most common class. "
        "Labels: predict empty set OR top-k most frequent labels."
    ),
    "priority": priority_results,
    "labels": {
        "vocab_size": len(vocab),
        "top_20_labels": top20_labels,
        "empty_set": label_results_empty,
        **label_results_topk,
    },
}

out_path = REPORTS_DIR / "baseline_metrics.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nSaved → {out_path}")
