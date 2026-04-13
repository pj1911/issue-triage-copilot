"""
Build a retrieval eval set (~40 queries) from val/test issues.

Sampling strategy
-----------------
- Prefer queries that have at least one "specific" label — avoids queries
  whose only label is 'bug' (15k relevant docs) or 'feature-request' (2.7k),
  which would make precision metrics uninformative.
- Balance across repos: ~5 per val repo (25) + ~3 per test repo (15).
- Queries must be labeled (we need ground truth to judge relevance).

Relevance judgment
------------------
A corpus document is "relevant" to a query if it shares ≥1 label.
For each query we mark 1–3 "gold" relevant docs: the highest-cosine-similarity
corpus docs among all label-matched candidates.  These are the targets a good
retriever should surface.

Output
------
  data/processed/retrieval_eval.json    — full eval set
  reports/eval_set_stats.json           — summary stats
"""
import json
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.paths import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

# ── paths ──────────────────────────────────────────────────────────────────────
DATASET_PATH  = PROCESSED_DIR / "dataset.parquet"
CORPUS_PATH   = PROCESSED_DIR / "retrieval_corpus.parquet"
VEC_PATH      = MODELS_DIR    / "tfidf_retrieval_vectorizer.joblib"
MATRIX_PATH   = MODELS_DIR    / "tfidf_retrieval_matrix.npz"
EVAL_PATH     = PROCESSED_DIR / "retrieval_eval.json"
STATS_PATH    = REPORTS_DIR   / "eval_set_stats.json"

# Generic labels whose relevant sets are too large to be discriminative
_GENERIC_LABELS = {"bug", "feature-request", "help wanted", "good first issue",
                   "enhancement", "proposal"}

# Queries per repo
_VAL_PER_REPO  = 5
_TEST_PER_REPO = 3

RANDOM_SEED = 42


# ── helpers ────────────────────────────────────────────────────────────────────

def has_specific_label(labels: list[str]) -> bool:
    """True if at least one label is not in the generic set."""
    return any(l not in _GENERIC_LABELS for l in labels)


def relevant_corpus_ids(query_labels: list[str], corpus: pd.DataFrame) -> np.ndarray:
    """Return integer indices (into corpus) whose labels overlap with query_labels."""
    qlabels = set(query_labels)
    mask = corpus["labels_clean"].apply(lambda l: bool(set(l) & qlabels))
    return np.where(mask.values)[0]


def pick_gold(query_text: str,
              cand_indices: np.ndarray,
              matrix: sp.csr_matrix,
              vectorizer,
              corpus: pd.DataFrame,
              n: int = 3) -> list[dict]:
    """
    Among candidate corpus rows (label-matched), pick the top-n by
    cosine similarity to the query text.  These are the gold relevant docs.
    """
    if len(cand_indices) == 0:
        return []

    q_vec    = vectorizer.transform([query_text])           # (1, vocab)
    cand_mat = matrix[cand_indices]                         # (|cand|, vocab)
    scores   = cosine_similarity(q_vec, cand_mat).ravel()   # (|cand|,)

    top_n    = min(n, len(cand_indices))
    top_pos  = np.argpartition(scores, -top_n)[-top_n:]
    top_pos  = top_pos[np.argsort(scores[top_pos])[::-1]]

    results = []
    for pos in top_pos:
        idx = int(cand_indices[pos])
        row = corpus.iloc[idx]
        results.append({
            "corpus_id":    int(row["id"]),
            "corpus_repo":  row["repo"],
            "shared_labels": sorted(set(row["labels_clean"]) &
                                    set(corpus.iloc[idx]["labels_clean"])),
            "score":        round(float(scores[pos]), 4),
            "text_snippet": row["text"][:200],
        })
    return results


def sample_queries(df: pd.DataFrame, split: str, per_repo: int,
                   rng: random.Random) -> pd.DataFrame:
    """
    Sample up to per_repo queries per repo from the given split.
    Prefer queries with specific labels; fall back to all labeled if needed.
    """
    rows = []
    split_df = df[(df["split"] == split) & (df["labels_clean"].apply(len) > 0)]

    for repo, group in split_df.groupby("repo"):
        specific = group[group["labels_clean"].apply(has_specific_label)]
        pool     = specific if len(specific) >= per_repo else group
        chosen   = pool.sample(n=min(per_repo, len(pool)),
                               random_state=rng.randint(0, 99999))
        rows.append(chosen)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(RANDOM_SEED)

    print("Loading data and index …")
    df         = pd.read_parquet(DATASET_PATH)
    corpus     = pd.read_parquet(CORPUS_PATH)
    vectorizer = joblib.load(VEC_PATH)
    matrix     = sp.load_npz(str(MATRIX_PATH))        # (71k, vocab)

    # ── sample queries ──────────────────────────────────────────────────────
    val_queries  = sample_queries(df, "val",  _VAL_PER_REPO,  rng)
    test_queries = sample_queries(df, "test", _TEST_PER_REPO, rng)
    queries      = pd.concat([val_queries, test_queries], ignore_index=True)
    print(f"Sampled {len(queries)} queries  "
          f"({len(val_queries)} val / {len(test_queries)} test)")

    # ── build eval records ──────────────────────────────────────────────────
    eval_set = []
    for _, row in queries.iterrows():
        query_labels = list(row["labels_clean"])
        cand_idx     = relevant_corpus_ids(query_labels, corpus)
        gold         = pick_gold(row["text"], cand_idx, matrix,
                                 vectorizer, corpus, n=3)

        # Fix shared_labels post-hoc: use actual intersection per gold doc
        for g in gold:
            g["shared_labels"] = sorted(
                set(query_labels) &
                set(corpus[corpus["id"] == g["corpus_id"]].iloc[0]["labels_clean"])
            )

        eval_set.append({
            "query_id":         int(row["id"]),
            "query_repo":       row["repo"],
            "query_split":      row["split"],
            "query_labels":     query_labels,
            "query_text":       row["text"],
            "n_relevant_corpus": int(len(cand_idx)),
            "gold_relevant":    gold,       # 1–3 gold docs, highest cosine sim
        })

    with open(EVAL_PATH, "w") as f:
        json.dump(eval_set, f, indent=2)
    print(f"Saved eval set ({len(eval_set)} queries) → {EVAL_PATH}")

    # ── stats ───────────────────────────────────────────────────────────────
    n_with_gold   = sum(1 for e in eval_set if e["gold_relevant"])
    n_gold_total  = sum(len(e["gold_relevant"]) for e in eval_set)
    rel_sizes     = [e["n_relevant_corpus"] for e in eval_set]
    by_repo       = {}
    for e in eval_set:
        by_repo.setdefault(e["query_repo"], 0)
        by_repo[e["query_repo"]] += 1

    stats = {
        "n_queries":          len(eval_set),
        "n_val_queries":      int((queries["split"] == "val").sum()),
        "n_test_queries":     int((queries["split"] == "test").sum()),
        "n_with_gold":        n_with_gold,
        "n_queries_no_gold":  len(eval_set) - n_with_gold,
        "gold_per_query":     round(n_gold_total / len(eval_set), 2),
        "relevant_set_size": {
            "min":    int(min(rel_sizes)),
            "median": int(np.median(rel_sizes)),
            "max":    int(max(rel_sizes)),
        },
        "queries_per_repo":   dict(sorted(by_repo.items())),
        "label_breakdown":    {
            e["query_id"]: {
                "labels": e["query_labels"],
                "n_relevant": e["n_relevant_corpus"],
                "n_gold":     len(e["gold_relevant"]),
            }
            for e in eval_set
        },
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nEval set summary:")
    print(f"  queries         : {stats['n_queries']}")
    print(f"  val / test      : {stats['n_val_queries']} / {stats['n_test_queries']}")
    print(f"  with gold       : {stats['n_with_gold']}")
    print(f"  gold / query    : {stats['gold_per_query']}")
    print(f"  relevant size   : min={stats['relevant_set_size']['min']}  "
          f"median={stats['relevant_set_size']['median']}  "
          f"max={stats['relevant_set_size']['max']}")
    print(f"  by repo         : {stats['queries_per_repo']}")
    print(f"\nSaved stats → {STATS_PATH}")

    # ── print a few examples ─────────────────────────────────────────────────
    print("\n── Example eval entries ──")
    for entry in eval_set[:3]:
        print(f"\nQuery [{entry['query_repo']}]  labels={entry['query_labels']}")
        print(f"  {entry['query_text'].splitlines()[0][:80]}")
        print(f"  {entry['n_relevant_corpus']} relevant corpus docs")
        for g in entry["gold_relevant"]:
            print(f"  gold: [{g['corpus_repo']}] score={g['score']}  "
                  f"shared={g['shared_labels']}")
            print(f"        {g['text_snippet'].splitlines()[0][:80]}")


if __name__ == "__main__":
    main()
