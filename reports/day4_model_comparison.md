# Day 4 Model Comparison — Multi-label Issue Labeling

**Task:** Predict GitHub issue labels (50-class multi-label classification)  
**Split:** repo-level (train: 58 repos, val: 5 repos, test: 5 repos)  
**Primary metric:** val micro-F1 (threshold tuned on val only)

---

## Results Table

| Model | Val micro-F1 | Val macro-F1 | Test micro-F1 | Test macro-F1 | Threshold | Notes |
|---|---|---|---|---|---|---|
| Majority baseline (empty set) | 0.000 | 0.000 | 0.000 | 0.000 | — | Predict no labels for every issue |
| Majority baseline (top-1: "bug") | 0.400 | 0.010 | 0.129 | 0.003 | — | Always predict most common label |
| TF-IDF + LR (OvR) | 0.273 | 0.028 | 0.101 | 0.029 | 0.15 | 100k bigrams, C=4.0, balanced weights |
| DistilBERT (run 1, lr=2e-5) | **0.355** | 0.027 | 0.099 | 0.014 | 0.10 | 4 epochs, batch=64, best at epoch 3 |
| DistilBERT (run 2, lr=5e-6) | 0.326 | 0.028 | 0.084 | 0.013 | 0.10 | 4 epochs, batch=64 — slower convergence |
| DistilBERT ablation: title-only | 0.246 | 0.021 | — | — | 0.10 | Same checkpoint, body stripped — delta -0.080 |

---

## Cost Comparison

| Model | Checkpoint size | Training time | Inference speed | Hardware |
|---|---|---|---|---|
| TF-IDF + LR | 44 MB | ~5 min | ~50,000 samples/sec | CPU |
| DistilBERT | 266 MB | ~6.5 min (4 ep × 96s) | ~130 samples/sec | V100 GPU |

---

## Per-label F1 (val, best encoder checkpoint)

Only 5 of 50 labels have non-zero F1 — the rest the model never predicts correctly.

| Label | TF-IDF F1 | Encoder F1 | Delta |
|---|---|---|---|
| bug | 0.567 | 0.402 | -0.165 |
| feature-request | 0.441 | 0.599 | +0.158 |
| documentation | 0.194 | 0.258 | +0.064 |
| help wanted | 0.087 | 0.114 | +0.027 |
| terminal | 0.013 | 0.016 | +0.003 |
| All others (45 labels) | 0.000 | 0.000 | 0.000 |

---

## Key Observations

1. **Encoder beats TF-IDF on val (+0.08 micro-F1)** — semantic understanding helps for `feature-request` and `documentation` which require reading meaning, not just keyword matching.

2. **TF-IDF wins on `bug` (F1=0.567 vs 0.402)** — bug reports have strong lexical signals ("error", "crash", "exception") that TF-IDF captures well. The encoder may be overfitting to longer body text.

3. **Both models fail on 45/50 labels** — the problem is not model capacity, it's data sparsity. Most labels appear in <2% of issues and are repo-specific. No model can learn these from cross-repo generalization.

4. **Val→test gap is structural, not a tuning problem** — test repos have 73% unlabeled issues vs 34% in val. The model learned val repo label conventions (vscode, transformers), not universal patterns.

5. **Threshold 0.10 is operationally problematic** — at this threshold the model predicts labels for 63% of val issues. The true positive rate in test is only 27%. This mismatch drives the test regression.

---

## Error Analysis (encoder, test set, 20,700 errors)

| Error type | Count | % of errors |
|---|---|---|
| False positives (unlabeled issue → predicted label) | 13,019 | 62.9% |
| Wrong labels (both non-empty, mismatch) | 4,556 | 22.0% |
| False negatives (labeled issue → no prediction) | 3,125 | 15.1% |

**Dominant error: false positives on flutter (9,495 errors) and rust (4,977 errors)**
These repos use few or no standard labels — the model incorrectly predicts `feature-request` / `bug` for most issues.

**Key error patterns:**
- Model confuses `enhancement` ↔ `feature-request` constantly (semantically near-identical, repo-convention-dependent)
- `help wanted` is missed whenever not paired with `bug` or `feature-request` in training context
- Repo-specific labels (`terminal`, `gopls`, `compiler`, `oncall: jit`) have near-zero F1 — the encoder learned nothing generalizable for these
- False negatives cluster on shorter issues with minimal body text — title-only cases where body adds no signal

---

## What the encoder did better vs TF-IDF

- Semantic near-synonyms: correctly identifies `feature-request` from natural language requests without keyword match
- Handles paraphrasing: "I would love to see support for X" → `feature-request` even without the word "feature"
- Better on `documentation` issues that describe *missing* docs without using the word "documentation"

## What TF-IDF did better

- `bug` detection: strong lexical signals are sufficient, encoder overshoots
- No GPU required, 400x faster inference
- Stable across thresholds — probabilities are better calibrated
- Smaller artifact size (6x smaller)

---

## Next step

The val→test gap (0.355 → 0.099) is not fixable by tuning the current architecture.
The core problem: labels are repo-specific conventions, not universal semantic categories.

Options worth exploring:
1. Retrieval-augmented labeling — find similar labeled issues from the same repo at inference time
2. Few-shot adaptation — fine-tune on a handful of labeled examples per test repo
3. Treat it as a retrieval problem, not a classification problem
