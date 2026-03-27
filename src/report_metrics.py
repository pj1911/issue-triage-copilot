"""
Print and save the full evaluation report for the TF-IDF baseline.

Covers:
  Priority  – weighted F1, macro F1, accuracy, confusion matrix
  Labels    – micro F1, macro F1, per-label F1 (top-N), exact match ratio

Threshold for labels is read from models/labels_threshold.joblib
(tuned on val only — test is evaluated once with that fixed threshold).

Run from project root:
    python -m src.report_metrics
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
)

from src.utils.paths import PROCESSED_DIR, REPORTS_DIR

MODELS_DIR = Path("models")

# ── Load ───────────────────────────────────────────────────────────────────────

ds   = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
val  = ds[ds["split"] == "val"].reset_index(drop=True)
test = ds[ds["split"] == "test"].reset_index(drop=True)

vec        = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
priority_lr = joblib.load(MODELS_DIR / "priority_lr.joblib")
labels_ovr  = joblib.load(MODELS_DIR / "labels_ovr.joblib")
mlb         = joblib.load(MODELS_DIR / "labels_mlb.joblib")
threshold   = joblib.load(MODELS_DIR / "labels_threshold.joblib")

X_val  = vec.transform(val["text"])
X_test = vec.transform(test["text"])

y_val_p  = val["priority_clean"].values
y_test_p = test["priority_clean"].values
y_val_l  = mlb.transform(val["labels_clean"])
y_test_l = mlb.transform(test["labels_clean"])

# ── Predictions ───────────────────────────────────────────────────────────────

pred_val_p  = priority_lr.predict(X_val)
pred_test_p = priority_lr.predict(X_test)

prob_val_l  = labels_ovr.predict_proba(X_val)
prob_test_l = labels_ovr.predict_proba(X_test)
pred_val_l  = (prob_val_l  >= threshold).astype(int)
pred_test_l = (prob_test_l >= threshold).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# PRIORITY
# ══════════════════════════════════════════════════════════════════════════════

PRIORITY_CLASSES = ["high", "medium", "low"]   # natural severity order


def priority_metrics(y_true, y_pred, split: str) -> dict:
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=PRIORITY_CLASSES,
                        average="macro",    zero_division=0)
    wt_f1    = f1_score(y_true, y_pred, labels=PRIORITY_CLASSES,
                        average="weighted", zero_division=0)
    cm       = confusion_matrix(y_true, y_pred, labels=PRIORITY_CLASSES)
    report   = classification_report(y_true, y_pred, labels=PRIORITY_CLASSES,
                                     output_dict=True, zero_division=0)
    report.pop("accuracy", None)

    return {
        "accuracy":    round(float(acc), 4),
        "macro_f1":    round(float(macro_f1), 4),
        "weighted_f1": round(float(wt_f1), 4),
        "confusion_matrix": {
            "labels":  PRIORITY_CLASSES,
            "matrix":  cm.tolist(),
        },
        "per_class": {
            k: {m: round(float(v), 4) for m, v in metrics.items()}
            for k, metrics in report.items()
            if isinstance(metrics, dict)
        },
    }


p_val  = priority_metrics(y_val_p,  pred_val_p,  "val")
p_test = priority_metrics(y_test_p, pred_test_p, "test")


# ══════════════════════════════════════════════════════════════════════════════
# LABELS
# ══════════════════════════════════════════════════════════════════════════════

TOP_N_LABELS = 20   # per-label F1 for the N most common labels in train


def label_metrics(y_true, y_pred, split: str) -> dict:
    micro_f1   = f1_score(y_true, y_pred, average="micro",    zero_division=0)
    macro_f1   = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    exact_match = accuracy_score(y_true, y_pred)   # exact match ratio
    hl          = hamming_loss(y_true, y_pred)

    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_label = {
        cls: round(float(f), 4)
        for cls, f in zip(mlb.classes_, per_label_f1)
    }
    # Sort by F1 descending; keep top N
    per_label_sorted = dict(
        sorted(per_label.items(), key=lambda x: -x[1])[:TOP_N_LABELS]
    )

    return {
        "threshold":        round(float(threshold), 2),
        "micro_f1":         round(float(micro_f1), 4),
        "macro_f1":         round(float(macro_f1), 4),
        "exact_match_ratio": round(float(exact_match), 4),
        "hamming_loss":     round(float(hl), 4),
        "per_label_f1_top20": per_label_sorted,
    }


l_val  = label_metrics(y_val_l,  pred_val_l,  "val")
l_test = label_metrics(y_test_l, pred_test_l, "test")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT
# ══════════════════════════════════════════════════════════════════════════════

SEP  = "─" * 60
SEP2 = "═" * 60

def print_priority(m: dict, split: str) -> None:
    print(f"\n{SEP}")
    print(f"  PRIORITY — {split.upper()}")
    print(SEP)
    print(f"  accuracy     : {m['accuracy']:.4f}")
    print(f"  macro F1     : {m['macro_f1']:.4f}   ← rare classes matter equally")
    print(f"  weighted F1  : {m['weighted_f1']:.4f}   ← practical overall perf")
    print()
    print(f"  {'class':<10} {'precision':>10} {'recall':>8} {'f1':>8} {'support':>9}")
    print(f"  {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*9}")
    for cls in PRIORITY_CLASSES:
        pc = m["per_class"].get(cls, {})
        print(f"  {cls:<10} {pc.get('precision',0):>10.4f} "
              f"{pc.get('recall',0):>8.4f} {pc.get('f1-score',0):>8.4f} "
              f"{int(pc.get('support',0)):>9,}")
    for avg in ["macro avg", "weighted avg"]:
        pc = m["per_class"].get(avg, {})
        print(f"  {avg:<10} {pc.get('precision',0):>10.4f} "
              f"{pc.get('recall',0):>8.4f} {pc.get('f1-score',0):>8.4f} "
              f"{int(pc.get('support',0)):>9,}")
    print()
    print("  Confusion matrix (rows=true, cols=pred):")
    cm     = m["confusion_matrix"]["matrix"]
    labels = m["confusion_matrix"]["labels"]
    header = "  " + " " * 10 + "".join(f"{l:>8}" for l in labels)
    print(header)
    for lbl, row in zip(labels, cm):
        print("  " + f"{lbl:<10}" + "".join(f"{v:>8,}" for v in row))


def print_labels(m: dict, split: str) -> None:
    print(f"\n{SEP}")
    print(f"  LABELS — {split.upper()}  (threshold={m['threshold']})")
    print(SEP)
    print(f"  micro F1         : {m['micro_f1']:.4f}   ← main metric")
    print(f"  macro F1         : {m['macro_f1']:.4f}   ← rare labels matter equally")
    print(f"  exact match ratio: {m['exact_match_ratio']:.4f}")
    print(f"  hamming loss     : {m['hamming_loss']:.4f}")
    print()
    print(f"  Per-label F1 (top {TOP_N_LABELS} by F1 score):")
    print(f"  {'label':<30} {'F1':>6}")
    print(f"  {'─'*30} {'─'*6}")
    for lbl, f1 in m["per_label_f1_top20"].items():
        bar = "█" * int(f1 * 20)
        print(f"  {lbl:<30} {f1:>6.4f}  {bar}")


print(f"\n{SEP2}")
print("  TF-IDF + Logistic Regression — Evaluation Report")
print(f"  Threshold tuned on val only; test evaluated once.")
print(SEP2)

print_priority(p_val,  "val")
print_priority(p_test, "test")
print_labels(l_val,  "val")
print_labels(l_test, "test")

print(f"\n{SEP2}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════

full_report = {
    "description": (
        "TF-IDF (1-2gram, 100k features) + Logistic Regression. "
        "Label threshold tuned on val, applied once to test."
    ),
    "priority": {"val": p_val, "test": p_test},
    "labels":   {"val": l_val, "test": l_test},
}

out = REPORTS_DIR / "tfidf_metrics.json"
out.write_text(json.dumps(full_report, indent=2))
print(f"\nSaved → {out}")
