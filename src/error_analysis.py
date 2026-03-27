"""
Error analysis for the TF-IDF baseline.

Samples ~50 mistakes from the validation set across four error types:
  priority_wrong  – misclassified priority (especially high/medium → low)
  label_fn        – label present in truth but not predicted (false negative)
  label_fp        – label predicted but not in truth (false positive)
  low_confidence  – prediction made but model is unsure

Each row is auto-bucketed into one of five error categories:
  too_short       – very little text, model had almost nothing to work with
  noisy_body      – long body dominated by stack traces / code
  rare_label      – the missed/predicted label appeared rarely in training
  repo_jargon     – the title/body uses repo-specific shorthand
  ambiguous       – text is genuinely underspecified or multi-interpretation

Run from project root:
    python -m src.error_analysis

Writes reports/error_analysis.csv
"""
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.paths import PROCESSED_DIR, REPORTS_DIR

MODELS_DIR = Path("models")
N_SAMPLES  = 50   # target total rows in the CSV
SEED       = 42

# ── Load ───────────────────────────────────────────────────────────────────────

ds   = pd.read_parquet(PROCESSED_DIR / "dataset.parquet")
val  = ds[ds["split"] == "val"].reset_index(drop=True)
train = ds[ds["split"] == "train"].reset_index(drop=True)

vec         = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
priority_lr = joblib.load(MODELS_DIR / "priority_lr.joblib")
labels_ovr  = joblib.load(MODELS_DIR / "labels_ovr.joblib")
mlb         = joblib.load(MODELS_DIR / "labels_mlb.joblib")
threshold   = joblib.load(MODELS_DIR / "labels_threshold.joblib")

# Label frequencies in training set
train_label_counts = Counter(
    lbl for row in train["labels_clean"] for lbl in row
)
RARE_THRESHOLD = 300   # labels appearing < this in train are "rare"

X_val = vec.transform(val["text"])

# ── Priority predictions ───────────────────────────────────────────────────────

proba_p   = priority_lr.predict_proba(X_val)
classes_p = priority_lr.classes_
pred_p    = classes_p[np.argmax(proba_p, axis=1)]
conf_p    = np.max(proba_p, axis=1)          # confidence = max class prob

# ── Label predictions ──────────────────────────────────────────────────────────

prob_l  = labels_ovr.predict_proba(X_val)
pred_l  = (prob_l >= threshold).astype(int)
y_val_l = mlb.transform(val["labels_clean"])


# ══════════════════════════════════════════════════════════════════════════════
# Auto-bucketing heuristics
# ══════════════════════════════════════════════════════════════════════════════

_CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_STACK_TRACE = re.compile(r"(at\s+\S+\.\S+\(|Traceback \(most recent|Exception in thread)", re.I)


def auto_bucket(text: str, title: str, missed_labels: list[str]) -> str:
    text_len  = len(text)
    title_len = len(title.strip())

    # 1. Too short
    if text_len < 300 or title_len < 15:
        return "too_short"

    # 2. Noisy body — heavy code / stack traces
    code_chars = sum(len(m.group()) for m in _CODE_BLOCK.finditer(text))
    if code_chars / max(text_len, 1) > 0.4 or _STACK_TRACE.search(text):
        return "noisy_body"

    # 3. Rare label missed
    if any(train_label_counts.get(lbl, 0) < RARE_THRESHOLD for lbl in missed_labels):
        return "rare_label"

    # 4. Repo-specific jargon — title has uppercase acronyms or bracketed tags
    if re.search(r"\[[A-Z]{2,}\]|\b[A-Z]{3,}\b", title):
        return "repo_jargon"

    # 5. Default: ambiguous text
    return "ambiguous"


# ══════════════════════════════════════════════════════════════════════════════
# Build error rows
# ══════════════════════════════════════════════════════════════════════════════

rng = np.random.default_rng(SEED)
rows = []


def add_row(idx, error_type, missed_labels=None, extra_pred_labels=None, confidence=None):
    row   = val.iloc[idx]
    true_lbls = list(row["labels_clean"])
    pred_lbls = list(mlb.classes_[pred_l[idx].astype(bool)])

    # Extract title from formatted text ("TITLE: ...\n\nBODY:\n...")
    text_str   = str(row["text"])
    title_line = text_str.split("\n")[0]                        # "TITLE: ..."
    title_snip = title_line.removeprefix("TITLE: ")[:120]

    missed = missed_labels or []
    bucket = auto_bucket(row["text"], title_snip, missed)

    rows.append({
        "error_type":      error_type,
        "bucket":          bucket,
        "id":              row["id"],
        "repo":            row["repo"],
        "title":           title_snip,
        "true_priority":   row["priority_clean"],
        "pred_priority":   pred_p[idx],
        "priority_conf":   round(float(conf_p[idx]), 3),
        "true_labels":     "|".join(true_lbls) if true_lbls else "(none)",
        "pred_labels":     "|".join(pred_lbls) if pred_lbls else "(none)",
        "missed_labels":   "|".join(missed) if missed else "",
        "fp_labels":       "|".join(extra_pred_labels or []),
        "label_max_prob":  round(float(prob_l[idx].max()), 3),
        "note":            "",   # filled manually
    })


# ── A. Priority wrong – focus on high→low and medium→low (most costly errors) ─

wrong_priority_idx = np.where(val["priority_clean"].values != pred_p)[0]

# Oversample high and medium misses
high_miss = [i for i in wrong_priority_idx if val.iloc[i]["priority_clean"] in ("high", "medium")]
low_miss  = [i for i in wrong_priority_idx if val.iloc[i]["priority_clean"] == "low"]

p_sample = list(rng.choice(high_miss, size=min(12, len(high_miss)), replace=False))
p_sample += list(rng.choice(low_miss,  size=min(3,  len(low_miss)),  replace=False))

for i in p_sample:
    add_row(i, "priority_wrong")

# ── B. Label false negatives (true label missed) ───────────────────────────────

fn_mask = (y_val_l == 1) & (pred_l == 0)
fn_rows, fn_cols = np.where(fn_mask)

# Build per-issue list of missed labels
fn_by_issue: dict[int, list[str]] = {}
for row_idx, col_idx in zip(fn_rows, fn_cols):
    fn_by_issue.setdefault(int(row_idx), []).append(mlb.classes_[col_idx])

fn_issue_idx = list(fn_by_issue.keys())
fn_sampled = list(rng.choice(fn_issue_idx, size=min(15, len(fn_issue_idx)), replace=False))

for i in fn_sampled:
    add_row(i, "label_fn", missed_labels=fn_by_issue[i])

# ── C. Label false positives (predicted but wrong) ────────────────────────────

fp_mask = (y_val_l == 0) & (pred_l == 1)
fp_rows, fp_cols = np.where(fp_mask)

fp_by_issue: dict[int, list[str]] = {}
for row_idx, col_idx in zip(fp_rows, fp_cols):
    fp_by_issue.setdefault(int(row_idx), []).append(mlb.classes_[col_idx])

fp_issue_idx = list(fp_by_issue.keys())
fp_sampled = list(rng.choice(fp_issue_idx, size=min(10, len(fp_issue_idx)), replace=False))

for i in fp_sampled:
    add_row(i, "label_fp", extra_pred_labels=fp_by_issue[i])

# ── D. Low-confidence priority (model unsure, margin < 0.2) ───────────────────

# Margin = gap between top-2 class probabilities
sorted_probs = np.sort(proba_p, axis=1)[:, ::-1]
margin = sorted_probs[:, 0] - sorted_probs[:, 1]
low_conf_idx = np.where(margin < 0.20)[0]

lc_sampled = list(rng.choice(low_conf_idx, size=min(10, len(low_conf_idx)), replace=False))
for i in lc_sampled:
    add_row(i, "low_confidence")


# ══════════════════════════════════════════════════════════════════════════════
# Auto-generate notes
# ══════════════════════════════════════════════════════════════════════════════

NOTE_TEMPLATES = {
    ("priority_wrong", "too_short"):   "Almost no body text; model defaulted to majority class.",
    ("priority_wrong", "noisy_body"):  "Stack trace / code dominates; priority signal buried.",
    ("priority_wrong", "ambiguous"):   "Text doesn't clearly indicate urgency.",
    ("priority_wrong", "repo_jargon"): "Title uses repo-specific tag not seen at training scale.",
    ("priority_wrong", "rare_label"):  "Unusual pattern for this priority class.",
    ("label_fn",  "too_short"):        "Body too sparse; label not predictable from text alone.",
    ("label_fn",  "noisy_body"):       "Relevant label buried under boilerplate or code.",
    ("label_fn",  "rare_label"):       "Label appears rarely in training; model under-predicts it.",
    ("label_fn",  "repo_jargon"):      "Label tied to repo-specific workflow, not text content.",
    ("label_fn",  "ambiguous"):        "Issue text doesn't clearly signal the missing label.",
    ("label_fp",  "too_short"):        "Short ambiguous text caused over-prediction.",
    ("label_fp",  "noisy_body"):       "Noisy body triggered false label pattern.",
    ("label_fp",  "ambiguous"):        "Text superficially resembles this label's pattern.",
    ("label_fp",  "repo_jargon"):      "Repo-specific term caused label confusion.",
    ("label_fp",  "rare_label"):       "Rare label threshold too low at 0.15.",
    ("low_confidence", "too_short"):   "Sparse text; model uncertain between classes.",
    ("low_confidence", "noisy_body"):  "Long noisy body diffuses signal across classes.",
    ("low_confidence", "ambiguous"):   "Issue could plausibly be multiple priorities.",
    ("low_confidence", "repo_jargon"): "Repo jargon unfamiliar to model.",
    ("low_confidence", "rare_label"):  "Issue type is rare; model hedges.",
}

for r in rows:
    key = (r["error_type"], r["bucket"])
    r["note"] = NOTE_TEMPLATES.get(key, "See title for context.")


# ══════════════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════════════

df = pd.DataFrame(rows)

# ── Print summary ──
print(f"Total error rows: {len(df)}")
print("\nBy error_type:")
print(df["error_type"].value_counts().to_string())
print("\nBy bucket:")
print(df["bucket"].value_counts().to_string())
print("\nPriority wrong — true vs predicted:")
pwrong = df[df["error_type"] == "priority_wrong"]
print(pwrong.groupby(["true_priority", "pred_priority"]).size().to_string())

out = REPORTS_DIR / "error_analysis.csv"
df.to_csv(out, index=False)
print(f"\nSaved → {out}")
print(df[["error_type","bucket","repo","title","true_priority","pred_priority",
          "true_labels","pred_labels","note"]].to_string(max_colwidth=60))
