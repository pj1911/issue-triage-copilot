# Day 3 Notes

## Tasks trained
- priority (single-label: low / medium / high)
- labels (multi-label, 50 classes)

## Baselines
- majority baseline: always predict most common class / empty set
- TF-IDF + logistic regression (OvR for both tasks)

## Validation results
- labels: micro F1 = 0.2733
- labels: macro F1 = 0.0276
- priority: macro F1 = 0.3652
- priority: weighted F1 = 0.7679

## Best settings
- ngram_range = (1, 2)
- min_df = 3
- max_features = 100,000
- sublinear_tf = True
- C = 4.0, class_weight = balanced, solver = liblinear (OvR)
- threshold = 0.15 (tuned on val, applied once to test)

## Main failure modes
- `medium` and `high` priority almost always collapse to `low` (89% class imbalance)
- 46 of 50 labels have F1 = 0 on val; model only learns the top 2–3 most common labels
- Repo-specific jargon (`[folding]`, `[HTML]`, acronyms) not shared across repos
- Ambiguous/underspecified issues dominate errors — 32 of 50 sampled mistakes
- Val → test micro F1 drops from 0.273 → 0.101: label distribution shifts across repos

## Next step
Train a stronger encoder-based model and compare against this baseline.
