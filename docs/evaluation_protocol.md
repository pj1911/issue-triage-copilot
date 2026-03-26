# Evaluation Protocol

## Split Type: Repo-Level

Issues are split by assigning entire repositories to one partition.
No repository appears in more than one split.

### Partition Assignments

| Split | Issues | % | Repos |
|-------|--------|---|-------|
| train | ~71,351 | 62.5% | all remaining repos |
| val | ~11,583 | 10.2% | vscode, deno, tauri, svelte, transformers |
| test | ~31,139 | 27.3% | flutter, rust, typescript, neovim, ollama |

### Why Repo-Level Split

A naive random split would allow the same repository's issues to appear in
both train and test. Because repos have strong label distribution signals
(e.g. pytorch always has `module:` labels, godot always has `engine`), a
random split would let the model learn "if it looks like a pytorch issue,
predict pytorch labels" rather than understanding the issue text itself.

Repo-level splitting forces the model to generalize across codebases —
which is the actual use case for an issue triage copilot.

### What Leakage This Avoids

1. **Label distribution leakage** — per-repo label conventions cannot be
   memorized; the model must learn from issue content.

2. **Vocabulary leakage** — label vocab and frequency counts are computed
   from train repos only, then applied to val/test. No frequency information
   from val/test repos influences which labels are kept.

3. **Post-triage label leakage** — labels assigned during/after triage
   (e.g. `triaged`, `needs-triage`, `confirmed`, `needsfix`, `awaiting more
   feedback`) are removed from all splits before training or evaluation.

### Test Repo Selection Rationale

Test repos were chosen to maximize domain diversity:
- **flutter** — mobile/UI, Dart ecosystem
- **rust** — systems/compiler, very different label taxonomy
- **typescript** — language tooling, Microsoft ecosystem
- **neovim** — editor tooling, C/Lua ecosystem
- **ollama** — LLM tooling, domain-relevant to this project

### Val Repo Selection Rationale

Val repos are used for hyperparameter tuning and early stopping:
- **vscode** — large editor repo, similar domain to neovim (test)
- **deno** — JavaScript runtime, overlaps with typescript (test)
- **tauri** — desktop framework, overlaps with flutter domain (test)
- **svelte** — frontend framework, overlaps with next.js (train)
- **transformers** — ML library, domain-relevant

---

## Evaluation Metrics

### Why Not Accuracy

Accuracy is misleading on imbalanced data.
- A model that always predicts `low` priority achieves 89% accuracy.
- A model that always predicts `bug` on multi-label achieves high recall on one label
  while ignoring all others.

Accuracy is not reported for either task.

---

## Task A: Multi-Label Classification (Labels)

### Evaluation Filter

Only rows with at least one label in the vocab are included in Task A evaluation.
Rows with empty `labels_clean` are excluded — they carry no ground truth signal
for the labels the model was trained to predict.

- Train rows with labels: ~46,094 / 71,351 (64.6%)
- Val rows with labels: ~7,668 / 11,583 (66.2%)
- Test rows with labels: ~8,289 / 31,139 (26.6%)

### Metrics

**Primary: Micro-F1**

Aggregates TP, FP, FN across all labels before computing precision and recall,
then combines into a single F1. Favors frequent labels (bug, enhancement) —
appropriate because these are the labels maintainers most need to get right.

```
Micro-F1 = 2 * (sum_TP) / (2 * sum_TP + sum_FP + sum_FN)
```

**Secondary: Macro-F1**

Computes F1 independently per label, then takes an unweighted average.
Gives equal weight to rare labels. Used to detect whether the model completely
ignores minority labels while doing well on the majority.

```
Macro-F1 = mean(F1_label_1, F1_label_2, ..., F1_label_N)
```

**Diagnostic: Per-Label F1 (top labels)**

Report F1 individually for the top 10 labels by train frequency.
This reveals which labels the model has learned vs. which it is skipping.
Logged to MLflow at each evaluation run.

**Decision threshold:** 0.5 per label (default). Tuned on val set if
macro-F1 on val is below 0.20 at baseline.

### Class Distribution Reported Alongside Metrics

Always report label frequency (train support) next to per-label F1.
A label with F1=0.60 and support=15,000 means something different than
F1=0.60 with support=200.

---

## Task B: Priority Prediction (3-class)

### Class Distribution

| Class | Train count | Train % |
|-------|-------------|---------|
| low | 63,639 | 89.2% |
| medium | 6,625 | 9.3% |
| high | 1,087 | 1.5% |

Always report this table alongside any metric so that the number is not
interpreted in isolation.

### Metrics

**Primary: Weighted F1**

Weights each class's F1 by its support before averaging.
Appropriate for imbalanced multi-class classification — penalizes models
that achieve high accuracy by predicting the majority class.

```
Weighted-F1 = sum(F1_class_i * support_i) / total_support
```

**Secondary: Macro-F1**

Unweighted average of per-class F1. Tracks whether the model has any
signal on `high` priority issues (support=1,087), which are rare but the
most actionable for a maintainer.

**Diagnostic: Confusion Matrix**

Logged as an artifact (not a scalar) at each evaluation run.
Key failure mode to watch: `high` predicted as `low` (dangerous miss).

---

## Reporting Standard

Every model evaluation must report:

| Field | Required |
|-------|----------|
| Split (val or test) | yes |
| Task A micro-F1 | yes |
| Task A macro-F1 | yes |
| Task A per-label F1 (top 10) | yes |
| Task B weighted-F1 | yes |
| Task B macro-F1 | yes |
| Task B confusion matrix | yes |
| Priority class distribution | yes |
| Label support alongside per-label F1 | yes |
| Model description / hyperparameters | yes |

Test set results are reported **once**, at the end of each model phase.
Val set is used during development. Do not repeatedly evaluate on test
and use results to guide decisions — that is test set leakage.

---

## Limitations

- Repo-level split reduces training data — 68 repos total leaves limited
  room for val/test diversity.
- Label vocab is train-only. Val/test repos may have native labels not in
  the vocab, which are silently dropped. This under-counts true positives
  for those repos.
- No timestamp available — cannot confirm that test issues are temporally
  later than train issues.
