# Task Definition: Issue Triage Copilot

## Input Columns
| Column | Notes |
|--------|-------|
| `title` | Always present, 0% null. Primary signal. |
| `body` | 0.12% null + 0.08% empty string. Fill with `""`. |

`repo` is available but excluded from V1 inputs — we want the model to generalize from
text, not memorize per-repo label distributions.

## Dropped Columns
| Column | Reason |
|--------|--------|
| `id` | Arbitrary identifier, no predictive value |
| `repo` | Risk of repo-memorization masking poor text understanding; revisit in V2 |

---

## Task A: Multi-Label Issue Classification

**Target:** `labels`

**Input:** `title`, `body`

**Output:** one or more labels per issue

### Label Cleaning Rules
- Split on comma: `"bug,enhancement"` → `["bug", "enhancement"]`
- Strip whitespace from each token
- Lowercase all labels
- Drop labels that appear fewer than 50 times across the dataset (too sparse to learn)
- Drop meta/process labels that are assigned post-triage and would cause leakage:
  - `triaged`, `triaged-*`, `needs-triage`, `needsinvestigation`, `has reproducible steps`
  - Priority-encoded labels: `p0`, `p1`, `p2`, `p3`, `p4`
  - Team routing labels: `team-*`, `t-*`
- Keep the top N labels after cleaning (start with N=20, tune later)

**Why keep this task:** Labels are the most direct output of triage. Multi-label F1 is
a well-understood evaluation metric and gives the project clear measurability.

---

## Task B: Priority Prediction

**Target:** `priority`

**Input:** `title`, `body`

**Output:** one of `{low, medium, high}`

### Class Distribution
| Class | Count | % |
|-------|-------|---|
| low | 101,937 | 89.4% |
| medium | 10,132 | 8.9% |
| high | 2,004 | 1.8% |

### Cleaning Rules
- No missing values — use as-is
- Handle imbalance via class weighting during training (not resampling at this stage)

**Why keep this task:** Priority is the most actionable output for a maintainer.
Despite imbalance, all three classes are present in sufficient volume to train and evaluate.
Weighted F1 will be the primary metric to avoid the majority-class trap.

---

## Task C: Severity Prediction — DROPPED FOR V1

**Target:** `severity`

**Reason for dropping:**
- "Critical" accounts for 58% of rows — suspiciously high for a severity label
- Distribution likely reflects per-repo labeling conventions, not true severity
- Without a timestamp or repo-normalized baseline, it is not possible to distinguish
  signal from labeling noise

**Revisit condition:** If a per-repo severity audit shows consistent semantics across
repos, we can add it back in V2.

---

## Final V1 Task List

| # | Task | Type | Target | Metric |
|---|------|------|--------|--------|
| A | Issue label classification | Multi-label | `labels` (cleaned, top 20) | Micro-F1, Macro-F1 |
| B | Priority prediction | Multi-class | `priority` | Weighted F1 |

---

## Notes on Leakage
- `labels` values like `triaged`, `P2`, `Needs-Triage` are assigned during or after
  triage — never use them as input features
- `repo` correlates strongly with label distributions — excluded from V1 inputs for
  this reason
- Train/val/test split must be done before any label frequency counting or vocabulary
  building to prevent data leakage from the future into preprocessing
