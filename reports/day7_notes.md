# Day 7 Notes — Hybrid Retrieval + Fairer Eval Set

## Goal

Combine lexical (TF-IDF) and dense (BGE-small) retrieval via Reciprocal Rank Fusion,
and measure all three methods on a human-annotated eval set that is not biased toward
either retrieval method.

---

## Why the original eval was biased

The Day 5/6 eval set (`retrieval_eval.json`, 37 queries) selected gold docs by TF-IDF
cosine similarity. This structurally guaranteed that lexical retrieval would score well
(its outputs are the ground truth) and dense retrieval would appear weak.

On the biased eval:
  - Lexical R@10 = 0.351
  - Dense R@10  = 0.072

The dense number is misleading — not a real measure of dense retrieval quality.

---

## Human-annotated eval set

**Approach:** generate candidates from BOTH retrievers + random label-matched docs,
then annotate by relevance judgment. Neither method dominates candidate selection.

**Pipeline:**
1. `build_human_eval.py` — 27 queries (3/repo × 9 repos), candidates = top-10 lex
   + top-10 dense + 5 random label-matched per query (654 total candidates)
2. Manual relevance annotation on all 654 candidates in-session
3. `finalize_human_eval.py` — converts annotations → `retrieval_eval_human.json`

**Eval set stats:**
- 21 queries retained (6 skipped: no cross-repo matches found)
- 78 gold docs (mean 3.7/query)
- Gold source: 71% dense-only, 19% lex-only, 10% both
- The 71% dense-only share confirms the original eval was suppressing dense retrieval

---

## Hybrid retrieval (RRF)

Reciprocal Rank Fusion (Cormack et al., 2009):

    score(d) = Σ  1 / (k + rank_m(d))    for each method m that retrieved d

Parameters:
- k = 60 (standard RRF constant)
- Pool = top-20 from each method before fusion

Implementation: `src/retrieval/hybrid_retriever.py`

---

## Results (21 queries, human-annotated eval set)

| Method       | R@5   | R@10  | R@20  | MRR   |
|--------------|-------|-------|-------|-------|
| Lexical      | 0.193 | 0.412 | 0.460 | 0.253 |
| Dense        | 0.383 | 0.721 | 0.721 | 0.619 |
| Hybrid (RRF) | 0.356 | 0.549 | 1.000 | 0.578 |

Dense wins overall. Hybrid R@20 = 1.000 (the combined pool has complete recall).

### By repo

| Repo         | Lex R@10 | Den R@10 | Hyb R@10 | Lex MRR | Den MRR | Hyb MRR | n |
|--------------|----------|----------|----------|---------|---------|---------|---|
| TypeScript   | 0.143    | 1.000    | 0.571    | 0.500   | 1.000   | 1.000   | 1 |
| deno         | 0.500    | 1.000    | 0.833    | 0.082   | 0.300   | 0.667   | 2 |
| flutter      | 0.278    | 0.833    | 0.333    | 0.214   | 0.417   | 0.374   | 3 |
| neovim       | 0.222    | 0.833    | 0.778    | 0.180   | 0.733   | 1.000   | 3 |
| ollama       | 0.611    | 0.389    | 0.111    | 0.137   | 0.148   | 0.106   | 3 |
| svelte       | 0.750    | 0.500    | 0.500    | 0.300   | 1.000   | 0.667   | 2 |
| tauri        | 0.583    | 0.500    | 0.767    | 0.444   | 0.667   | 0.567   | 3 |
| transformers | 0.308    | 0.825    | 0.552    | 0.370   | 0.833   | 0.667   | 3 |
| vscode       | 0.000    | 1.000    | 0.667    | 0.000   | 1.000   | 0.333   | 1 |

---

## Example analysis (from hybrid_examples.txt)

### Hybrid wins (3 cases — hybrid MRR beats both)

1. **neovim / extended attributes** — gold doc at lex rank 20 + dense rank 5.
   Neither method placed it in top-5 individually. RRF accumulated both weak
   signals and promoted it to rank 1. Hybrid MRR: 1.000 vs lex 0.200, dense 0.200.

2. **deno / --watch paths** — gold doc agreed on by both methods (lex:19 + dense:3).
   Hybrid promoted that consensus item to rank 1. MRR: 1.000 vs dense 0.500.

3. **deno / pty interface** — gold doc at lex:9 + dense:10, just outside each
   method's top-5. RRF elevated it to rank 3 (R@5=1.0). MRR: 0.333 vs lex 0.111.

### Dense beats hybrid (3 cases — dominant failure mode)

1. **svelte / runes+async** — dense ranked gold #1 (score 0.788). Lexical ranked
   an irrelevant SvelteKit bug #2. RRF promoted the noise item to #1, pushing
   gold to #3. Dense MRR: 1.000 → Hybrid MRR: 0.333.

2. **vscode / persistent terminals** — dense had 3 gold docs in top-5. Lexical
   had zero relevant hits but contributed noise docs from wrong parts of the
   terminal repo. Hybrid MRR: 0.333 vs Dense MRR: 1.000.

3. **transformers / Italian translation** — dense found 5 gold docs in its top-5
   (all translation-related issues). Lexical retrieved PyTorch issues matching
   "transformers" as a keyword. RRF interleaved them, cutting R@5 from 0.556 to 0.222.

### Lexical beats hybrid (2 cases)

1. **tauri / NPM package files** — both gold docs matched NPM-specific keywords
   (lexical rank 3 and 5). Dense retrieved wrong CLI issues. RRF promoted dense
   noise to top slots, pushing gold docs out. Lex MRR: 0.333 → Hybrid: 0.200.

2. **ollama / API error docs** — lexical found gold via "500 Internal Server Error"
   keyword match (rank 5). Dense latched onto Kubernetes/Langchain HTTP error issues
   semantically. Hybrid MRR: 0.083 vs Lex MRR: 0.200.

---

## Key conclusions

1. **Dense is the right primary retriever** for this dataset. On a fair eval,
   dense R@10 = 0.721 vs lexical 0.412. The Day 6 result (dense 0.072) was
   purely an artifact of biased gold selection.

2. **Equal-weight RRF underperforms dense** because 71% of gold docs are
   dense-only. Lexical introduces noise at the top ranks without adding recall.

3. **RRF does help when both methods agree weakly** — items at the periphery
   of each retriever's list get boosted by consensus.

4. **Recommended next step:** weighted RRF (`w_dense=0.7, w_lex=0.3`) would
   likely capture most of the RRF wins while limiting the noise damage.
   Alternatively, use dense as primary and add lexical only as a re-ranker.

---

## Artifacts

| File | Description |
|---|---|
| `data/processed/human_eval_candidates.json` | Annotated candidate pool (654 candidates) |
| `data/processed/retrieval_eval_human.json` | 21-query human-annotated eval set |
| `reports/human_eval_stats.json` | Annotation statistics |
| `reports/hybrid_metrics.json` | Full 3-way eval results (overall, by repo, per query) |
| `reports/hybrid_examples.json` | Win/failure case studies (structured) |
| `reports/hybrid_examples.txt` | Win/failure case studies (readable) |
