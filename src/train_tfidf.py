"""
TF-IDF + Logistic Regression baseline.

  Priority  : LogisticRegression (single-label, balanced weights)
  Labels    : OneVsRestClassifier(LogisticRegression) (multi-label, 50 classes)

Run from project root:
    python -m src.train_tfidf

Artifacts saved to:
    models/tfidf_vectorizer.joblib
    models/priority_lr.joblib
    models/labels_ovr.joblib
    reports/tfidf_metrics.json
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.utils.paths import PROCESSED_DIR, REPORTS_DIR, ROOT

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading data …")
ds = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")

train = ds[ds["split"] == "train"].reset_index(drop=True)
val   = ds[ds["split"] == "val"].reset_index(drop=True)
test  = ds[ds["split"] == "test"].reset_index(drop=True)

print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")

# ── Vectorize ──────────────────────────────────────────────────────────────────

print("\nFitting TF-IDF vectorizer …")
t0 = time.time()

vec = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    max_features=100_000,
    sublinear_tf=True,
)

X_train = vec.fit_transform(train["text"])
X_val   = vec.transform(val["text"])
X_test  = vec.transform(test["text"])

print(f"  vocab size : {len(vec.vocabulary_):,}")
print(f"  X_train    : {X_train.shape}  [{time.time()-t0:.1f}s]")

joblib.dump(vec, MODELS_DIR / "tfidf_vectorizer.joblib")
print(f"  Saved → models/tfidf_vectorizer.joblib")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PRIORITY
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Training priority model ──")
t0 = time.time()

y_train_p = train["priority_clean"].values
y_val_p   = val["priority_clean"].values
y_test_p  = test["priority_clean"].values

priority_lr = OneVsRestClassifier(
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=4.0,
        solver="liblinear",
    )
)
priority_lr.fit(X_train, y_train_p)
print(f"  Trained in {time.time()-t0:.1f}s")

joblib.dump(priority_lr, MODELS_DIR / "priority_lr.joblib")
print("  Saved → models/priority_lr.joblib")


def eval_priority(X, y_true, split: str) -> dict:
    y_pred  = priority_lr.predict(X)
    labels  = sorted(set(y_train_p))
    acc     = accuracy_score(y_true, y_pred)
    macro   = f1_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0)
    weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class.pop("accuracy", None)

    print(f"  {split}: accuracy={acc:.4f}  macro_f1={macro:.4f}  weighted_f1={weighted:.4f}")
    return {
        "accuracy":    round(acc, 4),
        "macro_f1":    round(macro, 4),
        "weighted_f1": round(weighted, 4),
        "per_class":   {
            k: {m: round(v, 4) for m, v in metrics.items()}
            for k, metrics in per_class.items()
            if isinstance(metrics, dict)
        },
    }


priority_results = {
    "model": "OvR(LogisticRegression(C=4.0, class_weight=balanced, solver=liblinear))",
    "val":  eval_priority(X_val,  y_val_p,  "val"),
    "test": eval_priority(X_test, y_test_p, "test"),
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. LABELS  (multi-label, 50 classes)
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Training labels model (OvR) ──")

mlb = MultiLabelBinarizer()
y_train_l = mlb.fit_transform(train["labels_clean"])
y_val_l   = mlb.transform(val["labels_clean"])
y_test_l  = mlb.transform(test["labels_clean"])

print(f"  Label classes : {len(mlb.classes_)}")

t0 = time.time()
labels_ovr = OneVsRestClassifier(
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=4.0,
        solver="liblinear",
    ),
    n_jobs=1,
)
labels_ovr.fit(X_train, y_train_l)
print(f"  Trained in {time.time()-t0:.1f}s")

joblib.dump(labels_ovr, MODELS_DIR / "labels_ovr.joblib")
joblib.dump(mlb,        MODELS_DIR / "labels_mlb.joblib")
print("  Saved → models/labels_ovr.joblib, labels_mlb.joblib")

# ── Tune prediction threshold on VAL only ─────────────────────────────────────
# sklearn's default is 0.5; many rare labels never reach that.
# We search over a grid and pick the threshold with best val micro-F1.

print("\n  Tuning label threshold on val …")
prob_val = labels_ovr.predict_proba(X_val)

best_thresh, best_val_micro = 0.5, 0.0
for t in np.arange(0.10, 0.55, 0.05):  # search on val only
    y_at_t = (prob_val >= t).astype(int)
    micro = f1_score(y_val_l, y_at_t, average="micro", zero_division=0)
    print(f"    threshold={t:.2f}  val micro_f1={micro:.4f}")
    if micro > best_val_micro:
        best_val_micro = micro
        best_thresh = t

print(f"  → Best threshold (val): {best_thresh:.2f}  micro_f1={best_val_micro:.4f}")
joblib.dump(best_thresh, MODELS_DIR / "labels_threshold.joblib")


def eval_labels(X, y_true, split: str, threshold: float) -> dict:
    """Evaluate using a fixed threshold (never re-tuned per split)."""
    prob   = labels_ovr.predict_proba(X)
    y_pred = (prob >= threshold).astype(int)

    hl         = hamming_loss(y_true, y_pred)
    micro_f1   = f1_score(y_true, y_pred, average="micro",    zero_division=0)
    macro_f1   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    subset_acc = accuracy_score(y_true, y_pred)

    label_f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_label = dict(sorted(
        {cls: round(float(f1), 4) for cls, f1 in zip(mlb.classes_, label_f1s)}.items(),
        key=lambda x: -x[1],
    ))

    print(f"  {split} (thr={threshold:.2f}): subset_acc={subset_acc:.4f}  "
          f"hamming={hl:.4f}  micro_f1={micro_f1:.4f}  macro_f1={macro_f1:.4f}")
    return {
        "threshold":       round(float(threshold), 2),
        "subset_accuracy": round(float(subset_acc), 4),
        "hamming_loss":    round(float(hl), 4),
        "micro_f1":        round(float(micro_f1), 4),
        "macro_f1":        round(float(macro_f1), 4),
        "per_label_f1":    per_label,
    }


# Val with tuned threshold (used for all model-selection decisions today)
print("\n  Evaluating with tuned threshold …")
val_results  = eval_labels(X_val,  y_val_l,  "val",  best_thresh)

# Test evaluated exactly once, after threshold is fixed from val
test_results = eval_labels(X_test, y_test_l, "test", best_thresh)

labels_results = {
    "model": "OneVsRestClassifier(LogisticRegression(C=4.0, class_weight=balanced))",
    "threshold_source": "tuned on val set only",
    "n_classes": int(len(mlb.classes_)),
    "classes": mlb.classes_.tolist(),
    "val":  val_results,
    "test": test_results,
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Save report
# ══════════════════════════════════════════════════════════════════════════════

report = {
    "description": "TF-IDF (1-2gram, 100k features, sublinear_tf) + Logistic Regression",
    "vectorizer": {
        "ngram_range": [1, 2],
        "max_features": 100_000,
        "min_df": 3,
        "max_df": 0.95,
        "sublinear_tf": True,
        "vocab_size": len(vec.vocabulary_),
    },
    "priority": priority_results,
    "labels":   labels_results,
}

out_path = REPORTS_DIR / "tfidf_metrics.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\nSaved report → {out_path}")
