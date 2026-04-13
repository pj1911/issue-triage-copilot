# Issue Triage Copilot

An end-to-end ML project for automated GitHub issue triage — predicting labels and priority for incoming issues across open-source repositories.

## Status

- [x] Dataset ingestion (114k issues, 63 repos)
- [x] EDA and label analysis
- [x] Data cleaning, label rules, leakage-safe repo-level split
- [x] Baseline classifiers (majority + TF-IDF + logistic regression)
- [x] Encoder fine-tuning (DistilBERT, multi-label)
- [x] Retrieval baseline (lexical, TF-IDF + cosine)
- [ ] Dense retrieval (encoder embeddings)
- [ ] Retrieval-augmented labeling
- [ ] Fine-tuned LLM
- [ ] Demo app

---

## Task

**Multi-label issue labeling** — given an issue title and body, predict which of 50 labels apply.

- 50-class multi-label classification
- Repo-level train/val/test split (no repo appears in more than one split)
- Primary metric: val micro-F1 with threshold tuned on val only

---

## Results so far

| Model | Val micro-F1 | Test micro-F1 | Notes |
|---|---|---|---|
| Majority baseline (empty set) | 0.000 | 0.000 | Predict nothing |
| Majority baseline (top-1 "bug") | 0.400 | 0.129 | Always predict most common label |
| TF-IDF + Logistic Regression | 0.273 | 0.101 | 100k bigrams, C=4.0 |
| DistilBERT (fine-tuned) | **0.355** | 0.099 | 4 epochs, batch=64, lr=2e-5 |

**Ablation:** body text contributes +0.080 micro-F1 over title-only.

**Key finding:** the val→test gap is structural — test repos have 73% unlabeled issues vs 34% in val. Better models don't close it; retrieval is the next step.

---

## Day 5 — Retrieval baseline

**Goal:** build a similar-issue retriever so the system can use repo-specific history instead of relying only on classifier generalization.

### Retrieval metrics (lexical: TF-IDF + cosine)

Eval set: 37 queries (25 val / 12 test), 3 gold relevant docs per query.

| Metric | Score |
|---|---|
| Recall@5 | 0.225 |
| Recall@10 | 0.351 |
| Recall@20 | 0.505 |
| MRR | 0.252 |

**By repo:**

| Repo | R@5 | R@10 | MRR |
|---|---|---|---|
| deno | 0.467 | 0.600 | 0.496 |
| neovim | 0.444 | 0.667 | 0.333 |
| vscode | 0.467 | 0.533 | 0.390 |
| svelte | 0.267 | 0.467 | 0.272 |
| ollama | 0.111 | 0.333 | 0.164 |
| TypeScript | 0.111 | 0.111 | 0.333 |
| tauri | 0.067 | 0.200 | 0.144 |
| flutter | 0.000 | 0.111 | 0.048 |
| transformers | 0.000 | 0.067 | 0.036 |

**Analysis:**
- **Strong repos (deno, neovim, vscode):** distinctive vocabulary maps directly to corpus tokens — lexical matching is sufficient
- **Weak repos (flutter, transformers):** repo-specific jargon and cross-repo label noise defeat keyword overlap; semantic matching needed
- **Failure modes** (from manual examples): label-only mismatch (query uses abstract language with no surface-form overlap) and cross-repo vocabulary gap (same label, completely different technical domain)

### Corpus

71,351 train issues — title + body + labels + repo per document. Only train issues included; no val/test contamination.

| | |
|---|---|
| Documents | 71,351 |
| Repos | 58 |
| Labeled | 46,094 (64.6%) |
| Vocab size | 150,000 features |

---

## Project structure

```
src/
  data/          # ingestion, cleaning, split, preprocessing
  models/        # encoder training, evaluation, ablations
  retrieval/     # corpus building, lexical retriever, eval, examples
  utils/         # shared paths
models/          # saved checkpoints and retrieval index
reports/         # metrics, error analysis, retrieval eval
configs/         # label cleaning rules
data/            # raw / interim / processed (gitignored)
```

---

## Quickstart

```bash
# setup
pip install -r requirements.txt

# data pipeline
python -m src.data.download
python -m src.data.split
python -m src.data.preprocess

# baselines
python -m src.baseline
python -m src.train_tfidf

# encoder (requires GPU — see train_encoder.slurm for HPC)
python -m src.models.train_encoder --max_train 2000   # dry run
python -m src.models.train_encoder                    # full run
python -m src.models.evaluate_encoder

# ablations
python -m src.models.sweep_threshold
python -m src.models.ablation_title_only

# retrieval baseline (CPU only)
python -m src.retrieval.build_corpus       # build retrieval corpus from train
python -m src.retrieval.lexical_retriever  # fit TF-IDF index + smoke test
python -m src.retrieval.build_eval         # build 37-query eval set
python -m src.retrieval.evaluate_retrieval # Recall@k + MRR
python -m src.retrieval.export_examples   # 10 good / 10 bad examples
python -m src.retrieval.save_outputs       # write report + CSV
```

---

## Data

114,073 GitHub issues from 63 repositories across diverse domains (systems, frontend, ML, tooling). Split by repo to test generalization across codebases:

- **Train:** 58 repos, 71,351 issues
- **Val:** 5 repos (vscode, deno, tauri, svelte, transformers), 11,583 issues
- **Test:** 5 repos (flutter, rust, typescript, neovim, ollama), 31,139 issues
