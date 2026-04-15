# Issue Triage Copilot

An end-to-end ML project for automated GitHub issue triage — predicting labels and priority for incoming issues across open-source repositories.

## Status

- [x] Dataset ingestion (114k issues, 63 repos)
- [x] EDA and label analysis
- [x] Data cleaning, label rules, leakage-safe repo-level split
- [x] Baseline classifiers (majority + TF-IDF + logistic regression)
- [x] Encoder fine-tuning (DistilBERT, multi-label)
- [x] Retrieval baseline (lexical, TF-IDF + cosine)
- [x] Dense retrieval (encoder embeddings)
- [x] Hybrid retrieval (RRF) + fairer human-annotated eval set
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

## Day 6 — Dense retrieval baseline

**Goal:** replace TF-IDF with BAAI/bge-small-en-v1.5 embeddings and measure whether semantic matching improves on the same 37-query eval set.

### Results (same eval set as Day 5)

| Metric | Lexical | Dense | Δ |
|---|---|---|---|
| Recall@5 | 0.225 | 0.045 | -0.180 |
| Recall@10 | 0.351 | 0.072 | -0.279 |
| Recall@20 | 0.505 | 0.081 | -0.424 |
| MRR | 0.252 | 0.099 | -0.153 |

**Hard repos (lexical was weakest):**

| Repo | Method | R@10 | MRR |
|---|---|---|---|
| flutter | Lexical | 0.111 | 0.048 |
| flutter | Dense | 0.111 | **0.167** |
| transformers | Lexical | 0.067 | 0.036 |
| transformers | Dense | **0.133** | **0.220** |

### Analysis

Overall dense underperforms — but the eval set gold docs were selected by TF-IDF cosine similarity, which structurally favors lexical retrieval. The signal that matters: on the two repos where lexical was near-zero (flutter, transformers), dense is strictly better.

Example exports (37 queries classified):
- **3 dense wins** — both on `transformers` with `documentation` labels; dense understood the concept where lexical had no surface overlap
- **10 dense failures** — `bug` reports on vscode/deno; keyword-heavy issues that TF-IDF handles exactly
- **5 both fail** — TypeScript `discussion` labels; abstract issues neither method anchors

**Conclusion:** lexical and dense are complementary. Lexical wins on concrete bug keywords; dense wins on semantic/conceptual labels. Neither handles abstract discussions. Next step: reciprocal rank fusion.

---

## Day 7 — Hybrid retrieval + fairer eval set

**Goal:** combine lexical and dense via Reciprocal Rank Fusion and measure on a human-annotated eval set that isn't biased toward either retrieval method.

### Why the Day 5/6 eval was biased

The original 37-query eval selected gold docs by TF-IDF cosine similarity. Dense retrieval scored 0.072 R@10 on that set — not because it retrieves poorly, but because its gold docs were literally chosen by the lexical method. A fair eval needs gold docs selected independently of both retrievers.

### Human-annotated eval set

- **27 queries** generated (3 per repo × 9 repos), candidates pooled from top-10 lexical + top-10 dense + 5 random label-matched docs per query
- **21 queries** retained after annotation (6 skipped: no cross-repo semantic matches found)
- **78 gold docs** annotated by relevance judgment (mean 3.7/query)
- Gold source: **71% dense-only**, 19% lex-only, 10% both — confirms the original eval was structurally suppressing dense retrieval

### Hybrid retrieval (RRF)

Reciprocal Rank Fusion over top-20 lexical + top-20 dense results:

```
score(d) = Σ  1 / (60 + rank_m(d))    for each method m that retrieved d
```

RRF constant k=60 (standard). Documents retrieved by both methods get boosted; documents missed by one method still appear if the other ranked them highly.

### Results (21 queries, human-annotated eval set)

| Method | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|
| Lexical (TF-IDF) | 0.193 | 0.412 | 0.460 | 0.253 |
| Dense (BGE-small) | **0.383** | **0.721** | 0.721 | **0.619** |
| Hybrid (RRF) | 0.356 | 0.549 | **1.000** | 0.578 |

**Hard repos (lexical was weakest on old eval):**

| Repo | Method | R@5 | R@10 | MRR |
|---|---|---|---|---|
| flutter | Lexical | 0.111 | 0.278 | 0.214 |
| flutter | Dense | 0.222 | **0.833** | **0.417** |
| flutter | Hybrid | 0.333 | 0.333 | 0.374 |
| transformers | Lexical | 0.133 | 0.308 | 0.370 |
| transformers | Dense | **0.418** | **0.825** | **0.833** |
| transformers | Hybrid | 0.357 | 0.552 | 0.667 |

### Analysis

- **Dense dominates on a fair eval** — R@10 0.721 vs 0.412 lexical. The Day 6 result (dense 0.072 R@10) was an artifact of biased gold selection.
- **Hybrid R@20 = 1.000** — the union of top-20 from each method captures every gold doc. The combined pool has full recall; the problem is ranking.
- **Equal-weight RRF underperforms dense** at R@5/R@10 — because 71% of gold docs are dense-only, lexical noise dilutes dense's strong signal at the top ranks. This is clearest on the Italian/French translation queries where dense ranks 5 gold docs in its top-5 but RRF interleaves irrelevant PyTorch results.
- **Hybrid wins when both methods agree weakly** — e.g. deno PTY query where the gold doc sat at lex:9 + dense:10 individually (outside top-5 for either), but RRF fusion promoted it to rank 3.
- **Next step**: weighted RRF (e.g. `w_dense=0.7, w_lex=0.3`) would likely recover the dense-beats-hybrid cases without losing the hybrid wins.

---

## Project structure

```
src/
  data/          # ingestion, cleaning, split, preprocessing
  models/        # encoder training, evaluation, ablations
  retrieval/     # corpus building, lexical retriever, eval, examples
    build_corpus.py            # build 71k-doc retrieval corpus
    lexical_retriever.py       # TF-IDF + cosine retriever
    dense_retriever.py         # BGE-small encoder + cosine retriever
    hybrid_retriever.py        # RRF fusion of lexical + dense        [Day 7]
    build_eval.py              # original biased eval set (37 queries)
    build_human_eval.py        # candidate pool for human annotation  [Day 7]
    finalize_human_eval.py     # convert annotations → eval set       [Day 7]
    evaluate_retrieval.py      # lexical eval (Recall@k, MRR)
    evaluate_dense.py          # dense eval
    evaluate_hybrid.py         # 3-way lexical/dense/hybrid eval      [Day 7]
    export_examples.py         # lexical win/fail examples
    export_dense_examples.py   # dense win/fail examples
    export_hybrid_examples.py  # hybrid win/fail examples             [Day 7]
  utils/         # shared paths
models/          # saved checkpoints and retrieval index
reports/         # metrics, error analysis, retrieval eval
  hybrid_metrics.json          # 3-way eval results                  [Day 7]
  hybrid_examples.json/txt     # hybrid win/fail case studies         [Day 7]
  human_eval_stats.json        # annotation statistics                [Day 7]
configs/         # label cleaning rules
data/
  processed/
    retrieval_eval_human.json  # 21-query human-annotated eval set    [Day 7]
    human_eval_candidates.json # annotated candidate pool             [Day 7]
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

# dense retrieval (requires GPU — see dense_retriever.slurm for HPC)
python -m src.retrieval.dense_retriever          # encode 71k docs, save index
python -m src.retrieval.evaluate_dense           # Recall@k + MRR vs lexical
python -m src.retrieval.export_dense_examples   # dense wins / failures / both-fail

# hybrid retrieval + fairer eval (Day 7)
python -m src.retrieval.build_human_eval         # generate candidate pool (lex+dense+random)
# annotate data/processed/human_eval_candidates.json, then:
python -m src.retrieval.finalize_human_eval      # produce retrieval_eval_human.json
python -m src.retrieval.hybrid_retriever         # smoke test RRF fusion
python -m src.retrieval.evaluate_hybrid          # 3-way comparison on human eval set
python -m src.retrieval.export_hybrid_examples   # hybrid wins / failures
```

---

## Data

114,073 GitHub issues from 63 repositories across diverse domains (systems, frontend, ML, tooling). Split by repo to test generalization across codebases:

- **Train:** 58 repos, 71,351 issues
- **Val:** 5 repos (vscode, deno, tauri, svelte, transformers), 11,583 issues
- **Test:** 5 repos (flutter, rust, typescript, neovim, ollama), 31,139 issues
