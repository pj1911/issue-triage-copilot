# Issue Triage Copilot

An end-to-end ML project for automated GitHub issue triage — predicting labels and priority for incoming issues across open-source repositories.

## Status

- [x] Dataset ingestion (114k issues, 63 repos)
- [x] EDA and label analysis
- [x] Data cleaning, label rules, leakage-safe repo-level split
- [x] Baseline classifiers (majority + TF-IDF + logistic regression)
- [x] Encoder fine-tuning (DistilBERT, multi-label)
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

## Project structure

```
src/
  data/          # ingestion, cleaning, split, preprocessing
  models/        # encoder training, evaluation, ablations
  utils/         # shared paths
models/          # saved checkpoints and configs
reports/         # metrics, error analysis, comparison tables
artifacts/
  encoder/       # all encoder experiment artifacts
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
```

---

## Data

114,073 GitHub issues from 63 repositories across diverse domains (systems, frontend, ML, tooling). Split by repo to test generalization across codebases:

- **Train:** 58 repos, 71,351 issues
- **Val:** 5 repos (vscode, deno, tauri, svelte, transformers), 11,583 issues
- **Test:** 5 repos (flutter, rust, typescript, neovim, ollama), 31,139 issues
